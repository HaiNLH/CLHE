import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder

class Cross_Attn(nn.Module):
    def __init__(self):
        """
        Construct a cross transformer
        """
        super(Cross_Attn, self).__init__()
        self.orig_d_t, self.orig_d_m, self.orig_d_c = 64, 64, 64
        self.d_t, self.d_m, self.d_c = 64, 64, 64
        #3 modality: text, media, user-item:c
        self.t_only = True   # Use only text 
        self.m_only = True   # Use only media
        self.c_only = True   # Use only user-item (cf) 
        self.num_heads = 4     
        self.layers = 2       
        self.attn_dropout = 0.1      
        self.attn_dropout_t = 0.1    
        self.attn_dropout_m = 0.1    
        self.attn_dropout_c = 0.1    
        self.relu_dropout = 0.1      
        self.res_dropout = 0.1       
        self.out_dropout = 0.1     
        self.embed_dropout = 0.1   
        self.attn_mask = None

        combined_dim = self.d_t + self.d_m + self.d_c

        self.parital_mode = self.t_only + self.m_only + self.c_only

        if self.parital_mode == 1:
            combined_dim = 2 * self.d_t # assume d_t == d_m == d_c
        else:
            combined_dim = 2*(self.d_t + self.d_m + self.d_c)
        
        output_dim  = 64
        # 1. Temporal convolution layers get all presentation of 3 modality

        self.proj_t = nn.Conv1d(self.orig_d_t, self.d_t, kernel_size = 1, padding = 0, bias = False) 
        self.proj_m = nn.Conv1d(self.orig_d_m, self.d_m, kernel_size = 1, padding = 0, bias = False)
        self.proj_c = nn.Conv1d(self.orig_d_c, self.d_c, kernel_size = 1, padding = 0, bias = False)
        

        # 2. Cross-modal attention
        if self.t_only:
            self.trans_t_with_m = self.get_network(self_type = 'tm')
            self.trans_t_with_c = self.get_network(self_type = 'tc')
        if self.m_only:
            self.trans_m_with_t = self.get_network(self_type = 'mt')
            self.trans_m_with_c = self.get_network(self_type = 'mc')
        if self.c_only:
            self.trans_c_with_t = self.get_network(self_type = 'ct')
            self.trans_c_with_m = self.get_network(self_type = 'cm')

        # 3. Self-attention (others optoin: LSTMs, GRUs,...)

        self.trans_t_mem = self.get_network(self_type = 't_mem', layers = 3)
        self.trans_m_mem = self.get_network(self_type = 'm_mem', layers = 3) 
        self.trans_c_mem = self.get_network(self_type = 'c_mem', layers = 3)

        #proj layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim*3)

    def get_network(self, self_type ='t', layers = 1):
        if self_type in ['t', 'mt', 'ct']:
            embed_dim, attn_dropout = self.d_t, self.attn_dropout
        elif self_type in ['m','tm','cm'] :
            embed_dim, attn_dropout = self.d_m, self.attn_dropout_m
        elif self_type in ['c','tc','mc']:
            embed_dim, attn_dropout = self.d_c, self.attn_dropout_c
        elif self_type == 't_mem':
            embed_dim, attn_dropout = 2*self.d_t, self.attn_dropout
        elif self_type == 'm_mem':
            embed_dim, attn_dropout = 2*self.d_m, self.attn_dropout
        elif self_type == 'c_mem':
            embed_dim, attn_dropout = 2* self.d_c, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim = embed_dim,
                                  num_heads = self.num_heads,
                                  layers = max(self.layers, layers),
                                  attn_dropout = attn_dropout,
                                  relu_dropout = self.relu_dropout,
                                  res_dropout = self.res_dropout,
                                  embed_dropout = self.embed_dropout,
                                  attn_mask = self.attn_mask)
    
    def forward(self, x_t, x_m, x_c):
        """
        text, media, content should have dimension [batch_size, seq_len, n_features]
        """
        if x_t.dim() < 3:
            x_t = x_t.unsqueeze(1)
        if x_m.dim() < 3:
            x_m = x_m.unsqueeze(1)
        if x_c.dim() < 3:
            x_c = x_c.unsqueeze(1)

        x_t = F.dropout(x_t.transpose(1,2), p = self.embed_dropout, training = self.training)
        x_m = x_m.transpose(1,2)
        x_c = x_c.transpose(1,2)

        #Proj the textual/media/content
        proj_x_t = x_t if self.orig_d_t == self.d_t else self.proj_t(x_t)
        proj_x_m = x_m if self.orig_d_m == self.d_m else self.proj_m(x_m)
        proj_x_c = x_c if self.orig_d_c == self.d_c else self.proj_c(x_c)
        proj_x_t = proj_x_t.permute(2,0,1)
        proj_x_m = proj_x_m.permute(2,0,1)
        proj_x_c = proj_x_c.permute(2,0,1)
        

        if self.t_only:
            #(M,C) --> T
            h_t_with_ms = self.trans_t_with_m(proj_x_t, proj_x_m, proj_x_m) #Dim (T, N ,d_t)
            h_t_with_cs = self.trans_t_with_c(proj_x_t, proj_x_c, proj_x_c) #Dim (T, N, d_t)
            h_ts = torch.cat([h_t_with_ms, h_t_with_cs], dim = 2)
            h_ts = self.trans_t_mem(h_ts)

            if type(h_ts) == tuple:
                h_ts = h_ts[0]
            last_h_t = last_hs = h_ts[-1] # take the last output for prediction

        if self.m_only:
            #(T,C) --> M
            h_m_with_ts = self.trans_m_with_t(proj_x_m, proj_x_t, proj_x_t) #Dim (T, N ,d_t)
            h_m_with_cs = self.trans_m_with_c(proj_x_m, proj_x_c, proj_x_c) #Dim (T, N, d_t)
            h_ms = torch.cat([h_m_with_ts, h_m_with_cs], dim = 2)
            h_ms = self.trans_m_mem(h_ms)

            if type(h_ms) == tuple:
                h_ms = h_ms[0]
            last_h_m = last_hs = h_ms[-1] # take the last output for prediction
        if self.c_only:
            #(T,M) --> C
            h_c_with_ts = self.trans_c_with_t(proj_x_c, proj_x_t, proj_x_t) #Dim (T, N ,d_t)
            h_c_with_ms = self.trans_c_with_m(proj_x_c, proj_x_m, proj_x_m) #Dim (T, N, d_t)
            h_cs = torch.cat([h_c_with_ts, h_c_with_ms], dim = 2)
            h_cs = self.trans_c_mem(h_cs)

            if type(h_cs) == tuple:
                h_cs = h_cs[0]
            last_h_c = last_hs = h_cs[-1] # take the last output for prediction
        
        if self.parital_mode == 3:
            last_hs = torch.cat([last_h_t, last_h_m, last_h_c], dim = 1)

        #residual block
        last_hs_proj = self.proj2(F.dropout(
            F.relu(self.proj1(last_hs)),
            p =self.out_dropout,
            training = self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        #item*64*3
        return output, last_hs
    