import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from models.utils import TransformerEncoder
from collections import OrderedDict

eps = 1e-9


def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)


def recon_loss_function(recon_x, x):
    negLogLike = torch.sum(F.log_softmax(recon_x, 1) * x, -1) / x.sum(dim=-1)
    negLogLike = -torch.mean(negLogLike)
    return negLogLike


infonce_criterion = nn.CrossEntropyLoss()


def cl_loss_function(a, b, temp=0.2):
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    logits = torch.mm(a, b.T)
    logits /= temp
    labels = torch.arange(a.shape[0]).to(a.device)
    return infonce_criterion(logits, labels)


class HierachicalEncoder(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(HierachicalEncoder, self).__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.num_cate = self.conf["num_cates"]
        
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen, self.ic_graph = raw_graph
        print(self.ic_graph)
        self.attention_components = self.conf["attention"]

        self.content_feature, self.text_feature, self.cf_feature = features

        items_in_train = self.bi_graph_train.sum(axis=0, dtype=bool)
        self.warm_indices = torch.LongTensor(
            np.argwhere(items_in_train)[:, 1]).to(device)
        self.cold_indices = torch.LongTensor(
            np.argwhere(~items_in_train)[:, 1]).to(device)

        # MM >>>
        self.content_feature = nn.functional.normalize(
            self.content_feature, dim=-1)
        self.text_feature = nn.functional.normalize(self.text_feature, dim=-1)

        def dense(feature):
            module = nn.Sequential(OrderedDict([
                ('w1', nn.Linear(feature.shape[1], feature.shape[1])),
                ('act1', nn.ReLU()),
                ('w2', nn.Linear(feature.shape[1], 256)),
                ('act2', nn.ReLU()),
                ('w3', nn.Linear(256, 64)),
            ]))

            for m in module:
                init(m)
            return module

        # encoders for media feature
        self.c_encoder = dense(self.content_feature)
        self.t_encoder = dense(self.text_feature)

        self.multimodal_feature_dim = self.embedding_size
        # MM <<<

        # BI >>>
        self.item_embeddings = nn.Parameter(
            torch.FloatTensor(self.num_item, self.embedding_size))
        init(self.item_embeddings)
        self.multimodal_feature_dim += self.embedding_size
        # BI <<<

        # UI >>>
        self.cf_transformation = nn.Linear(
            self.embedding_size, self.embedding_size)
        init(self.cf_transformation)
        items_in_cf = self.ui_graph.sum(axis=0, dtype=bool)
        self.warm_indices_cf = torch.LongTensor(
            np.argwhere(items_in_cf)[:, 1]).to(device)
        self.cold_indices_cf = torch.LongTensor(
            np.argwhere(~items_in_cf)[:, 1]).to(device)
        self.multimodal_feature_dim += self.embedding_size
        # UI <<<

        # Multimodal Fusion:
        self.w_q = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_size, elementwise_affine=False)

    def selfAttention(self, features):
        # features: [bs, #modality, d]
        if "layernorm" in self.attention_components:
            features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        if "w_v" in self.attention_components:
            v = self.w_v(features)
        else:
            v = features
        # [bs, #modality, #modality]
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [bs, #modality, d]
        # average pooling
        y = features.mean(dim=-2)  # [bs, d]

        return y

    def forward_all(self):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        features = [mm_feature_full]
        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        features.append(cf_feature_full)

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]

        # multimodal fusion >>>
        final_feature = self.selfAttention(F.normalize(features, dim=-1))
        # multimodal fusion <<<

        return final_feature

    def forward(self, seq_modify, all=False):
        if all is True:
            return self.forward_all()

        modify_mask = seq_modify == self.num_item
        seq_modify.masked_fill_(modify_mask, 0)

        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        mm_feature = mm_feature_full[seq_modify]  # [bs, n_token, d]

        features = [mm_feature]
        bi_feature_full = self.item_embeddings
        bi_feature = bi_feature_full[seq_modify]
        features.append(bi_feature)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        cf_feature = cf_feature_full[seq_modify]
        features.append(cf_feature)

        features = torch.stack(features, dim=-2)  # [bs, n_token, #modality, d]
        bs, n_token, N_modal, d = features.shape

        # multimodal fusion >>>
        final_feature = self.selfAttention(
            F.normalize(features.view(-1, N_modal, d), dim=-1))
        final_feature = final_feature.view(bs, n_token, d)
        # multimodal fusion <<<

        return final_feature

    def generate_two_subs(self, dropout_ratio=0):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        # early-fusion
        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        features = [mm_feature_full]

        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        features.append(cf_feature_full)

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]
        size = features.shape[:2]  # (bs, #modality)

        def random_mask():
            random_tensor = torch.rand(size).to(features.device)
            mask_bool = random_tensor < dropout_ratio  # the remainders are true
            masked_feat = features.masked_fill(mask_bool.unsqueeze(-1), 0)

            # multimodal fusion >>>
            final_feature = self.selfAttention(
                F.normalize(masked_feat, dim=-1))
            # multimodal fusion <<<
            return final_feature

        return random_mask(), random_mask()


class CLHE(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.raw_graph = raw_graph
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.num_cate = self.conf["num_cates"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen, self.ic_graph= raw_graph
        self.item_augmentation = self.conf["item_augment"]

        self.encoder = HierachicalEncoder(conf, raw_graph, features)
        # decoder has the similar structure of the encoder
        self.decoder = HierachicalEncoder(conf, raw_graph, features)

        self.bundle_encode = TransformerEncoder(conf={
            "n_layer": conf["trans_layer"],
            "dim": 64,
            "num_token": 100,
            "device": self.device,
        }, data={"sp_graph": self.bi_graph_seen})

        self.cl_temp = conf['cl_temp']
        self.cl_alpha = conf['cl_alpha']
 
        self.bundle_cl_temp = conf['bundle_cl_temp']
        self.bundle_cl_alpha = conf['bundle_cl_alpha']
        self.cl_projector = nn.Linear(self.embedding_size, self.embedding_size)
        init(self.cl_projector)
        if self.item_augmentation in ["FD", "MD"]:
            self.dropout_rate = conf["dropout_rate"]
            self.dropout = nn.Dropout(p=self.dropout_rate)
        elif self.item_augmentation in ["FN"]:
            self.noise_weight = conf['noise_weight']

    def forward(self, batch):
        idx, full, seq_full, modify, seq_modify = batch  # x: [bs, #items]
        mask = seq_full == self.num_item
        feat_bundle_view = self.encoder(seq_full)  # [bs, n_token, d]

        # bundle feature construction >>>
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        feat_retrival_view = self.decoder(batch, all=True)

        # compute loss >>>
        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)
        loss = recon_loss_function(logits, full)  # main_loss 

        # # item-level contrastive learning >>>
        items_in_batch = torch.argwhere(full.sum(dim=0)).squeeze()
        item_loss = torch.tensor(0).to(self.device)
        if self.cl_alpha > 0:
            if self.item_augmentation == "FD":
                item_features = self.encoder(batch, all=True)[items_in_batch]
                #make it go through Light-GCN
                sub1 = self.cl_projector(self.dropout(item_features))
                sub2 = self.cl_projector(self.dropout(item_features))
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), sub2.view(-1, self.embedding_size), self.cl_temp)
            elif self.item_augmentation == "NA":
                item_features = self.encoder(batch, all=True)[items_in_batch]
                item_loss = self.cl_alpha * cl_loss_function(
                item_features.view(-1, self.embedding_size), item_features.view(-1, self.embedding_size), self.cl_temp)
            elif self.item_augmentation == "FN":
                item_ft = self.encoder(batch, all=True)
                tmp = BunCa(self.conf, self.raw_graph, item_ft,self.device).propagate()
                # print(tmp.shape)
                # print(item_ft.shape)
                # print(items_in_batch.long)
                item_features = tmp[items_in_batch]
                # item_f = item_ft[items_in_batch]
                # print(item_features.shape)
                # print(item_f.shape)
                sub1 = self.cl_projector(
                    self.noise_weight * torch.randn_like(item_features) + item_features)
                sub2 = self.cl_projector(
                    self.noise_weight * torch.randn_like(item_features) + item_features)
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), sub2.view(-1, self.embedding_size), self.cl_temp)
            elif self.item_augmentation == "MD":
                sub1, sub2 = self.encoder.generate_two_subs(self.dropout_rate)
                bunca_model1 = BunCa(self.conf, self.raw_graph, sub1)
                bunca_model2 = BunCa(self.conf, self.raw_graph,sub2)
                items1_feature = bunca_model1.propagate()
                items2_feature = bunca_model2.propagate()
                
                sub1 = items1_feature[items_in_batch]
                sub2 = items2_feature[items_in_batch]
                # Process features as needed
                # combined_feature = torch.cat([bundles_feature[0], items_feature[0]], dim=-1)
                # Pass through additional layers
                # output = self.fc(combined_feature)


                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), sub2.view(-1, self.embedding_size), self.cl_temp)
        # # item-level contrastive learning <<<

        # bundle-level contrastive learning >>>
        bundle_loss = torch.tensor(0).to(self.device)
        if self.bundle_cl_alpha > 0:
            feat_bundle_view2 = self.encoder(seq_modify)  # [bs, n_token, d]
            bundle_feature2 = self.bundle_encode(feat_bundle_view2, mask=mask)
            bundle_loss = self.bundle_cl_alpha * cl_loss_function(
                bundle_feature.view(-1, self.embedding_size), bundle_feature2.view(-1, self.embedding_size), self.bundle_cl_temp)
        # bundle-level contrastive learning <<<

        return {
            'loss': loss + item_loss + bundle_loss,
            'item_loss': item_loss.detach(),
            'bundle_loss': bundle_loss.detach()
        }

    def evaluate(self, _, batch):
        idx, x, seq_x = batch
        mask = seq_x == self.num_item
        feat_bundle_view = self.encoder(seq_x)

        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        feat_retrival_view = self.decoder(
            (idx, x, seq_x, None, None), all=True)

        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)

        return logits

    def propagate(self, test=False):
        return None


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1 - dropout_ratio])
    values = mask * values
    return values

#Idea is to reconstruct the part getting item embedding learning from connection U-C C-I => U-I, B-U, U-I
#Getting item rep after using self attention => Information from User and Bundle and Category
class BunCa(nn.Module):
    def __init__(self,conf, raw_graph, items_feat,device):
        super().__init__()
        self.conf = conf
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_cates = conf["num_cates"]
        self.items_feature  = items_feat


        self.ui_graph, self.bi_graph_train, self.bi_graph_seen, self.ic_graph= raw_graph
        self.bc_graph = self.bi_graph_train@self.ic_graph
        self.num_layers = self.conf["num_layers"]
        self.init_emb()
        self.init_md_dropouts()
        self.get_item_agg_graph()
        self.get_item_agg_graph()

    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(0.2, True)
        self.bundle_level_dropout = nn.Dropout(0.2, True)
        self.bundle_agg_dropout = nn.Dropout(0.2, True)

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        # self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        # nn.init.xavier_normal_(self.items_feature)
        self.cates_feature = nn.Parameter(torch.FloatTensor(self.num_cates, self.embedding_size))
        nn.init.xavier_normal_(self.cates_feature)

    def get_cate_level_graph(self):
        bc_graph = self.bc_graph
        ic_graph = self.ic_graph
        device = self.device
        cate_level_graph = sp.bmat([[sp.csr_matrix((bc_graph.shape[0], bc_graph.shape[0])),bc_graph],
                                    [bc_graph.T, sp.csr_matrix((bc_graph.shape[1],bc_graph.shape[1]))]])
        ic_propagate_graph = sp.bmat([[sp.csr_matrix((ic_graph.shape[0], ic_graph.shape[0])), ic_graph],
                                      [ic_graph.T, sp.csr_matrix((ic_graph.shape[1], ic_graph.shape[1]))]])
        self.ic_propagate_graph_ori = to_tensor(laplace_transform(ic_propagate_graph)).to(device)
        self.cate_level_graph = to_tensor(laplace_transform(cate_level_graph)).to(device)
        self.ic_propagate_graph = to_tensor(laplace_transform(ic_propagate_graph)).to(device)

    def get_item_level_graph(self, threshold=0):
        bi_graph = self.bi_graph_train
        device = self.device
        print(bi_graph.shape)

        bb_graph = (bi_graph @ bi_graph.T) >threshold
        ii_graph = (bi_graph.T @ bi_graph) > threshold
        print(ii_graph.shape)
        print(bb_graph.shape)
        item_level_graph = sp.bmat([[bb_graph, bi_graph],
                                    [bi_graph.T, ii_graph]])
        
        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)
        return self.item_level_graph

    def get_item_agg_graph(self):
        ic_graph = self.ic_graph
        device = self.device

        item_size  = ic_graph.sum(axis=1) +1e-8
        ic_graph = sp.diags(1/item_size.A.ravel())@ ic_graph
        self.item_agg_graph = to_tensor(ic_graph).to(device)

    def get_bundle_agg_graph(self):
        bc_graph  = self.bc_graph
        device = self.device

        bundle_size = bc_graph.sum(axis = 1) + 1e-8
        bc_graph = sp.diags(1/bundle_size.A.ravel())@bc_graph
        self.bundle_agg_graph = to_tensor(bc_graph).to(device)

    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test, coefs=None):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features).to(self.device)


            features = features / (i + 2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        if coefs is not None:
            all_features = all_features * coefs
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    def get_CL_item_rep(self, CL_cates_feature,test):
        if test:
            CL_items_feature = torch.matmul(self.item_agg_graph, CL_cates_feature)
        else:
            CL_items_feature = torch.matmul(self.item_agg_graph, CL_cates_feature)
        # simple embedding dropout on bundle embeddings
        # if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
        #    CL_items_feature = self.item_agg_dropout(CL_items_feature)
        return CL_items_feature
    
    
    def propagate(self, test = False):
        #______________CATE level _______________
        # if test:
        #     CL_bundles_feature, CL_cates_feature = self.one_propagate(self.cate_level_graph, self.bundles_feature, self.cates_feature, self.item_level_dropout, test)
        # else:
        self.get_cate_level_graph()
        CL_bundles_feature, CL_cates_feature = self.one_propagate(self.cate_level_graph, self.bundles_feature, self.cates_feature, self.item_level_dropout, test)
        
        CL_items_feature = self.get_CL_item_rep(CL_cates_feature,test)

        #_____________ITEM level_________________
        # if test:
        #     IL_bundles_feature, IL_item_features = self.one_propagate(self.get_item_level_graph_ori, self.bundles_feature, self.items_feature, self.bundle_level_dropout, test)
        # else:
        self.get_item_level_graph()
        IL_bundles_feature, IL_item_features = self.one_propagate(self.item_level_graph, self.bundles_feature, self.items_feature, self.bundle_level_dropout, test)
        
        # bundles_feature = [CL_bundles_feature, IL_bundles_feature]
        # items_feature  = [CL_items_feature, IL_item_features]

        # bundles_feature = CL_bundles_feature @ W_bundle1 + IL_bundles_feature @ W_bundle2
        items_feature = CL_items_feature * 0.6 + IL_item_features * 0.4
        return items_feature