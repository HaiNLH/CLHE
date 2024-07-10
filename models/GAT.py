import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.optim as optim


#Getting the data
pairs = []

with open("/kaggle/input/clhe-pog/pog/ui_full.txt", 'r') as f:
    for line in f:
        # Split the line into bundle ID and item IDs
        ids = list(map(int, line.strip().split(',')))
        user_id = ids[0]
        item_ids = ids[1:]

        # Create pairs of bundle_id and each item_id
        for item_id in item_ids:
            pairs.append([user_id, item_id])
#     pairs = list(set(pairs))
#     Extract the first two integers from each line for indices
    indices = np.array([[pair[0], pair[1]] for pair in pairs], dtype=np.int32)
    values = np.ones(len(pairs), dtype=np.float32)

    u_i_graph = sp.coo_matrix(
        (values, (indices[:, 0], indices[:, 1])), shape=(x, y)).tocsr()

u_i_pairs =[]
x= 17449
y = 48676
def get_ui():
    with open("/kaggle/input/clhe-pog/pog/ui_full.txt", 'r') as f:
        for line in f:
            # Split the line into bundle ID and item IDs
            ids = list(map(int, line.strip().split(',')))
            user_id = ids[0]
            item_ids = ids[1:]

            # Create pairs of bundle_id and each item_id
            for item_id in item_ids:
                u_i_pairs.append((user_id, item_id))
        pairs = list(set(u_i_pairs))
    #     Extract the first two integers from each line for indices
        indices = np.array([[pair[0], pair[1]] for pair in u_i_pairs], dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)

    u_i_graph = sp.coo_matrix(
        (values, (indices[:, 0], indices[:, 1])), shape=(x, y)).tocsr()

    return u_i_pairs, u_i_graph
u_i_pairs, ui_graph = get_ui()

       
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

def get_item_level_graph_ori():
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph],
                                    [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        item_level_graph_ori = to_tensor(laplace_transform(item_level_graph))
        return item_level_graph_ori

graph = graph.coalesce()
uigraph = to_tensor(ui_graph)
uigraph = uigraph.coalesce()
uigraph.shape

ui_tensor = torch.sparse_coo_tensor(uigraph.indices(), uigraph.values(), uigraph.shape)
ui_tensor = ui_tensor.to_dense()
uigraph.values().shape

from torch_geometric.data import Data, DataLoader
import torch.optim as optim
# model = GAT(in_channels=64, out_channels=64, heads=2,concat=False)
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super().__init__()
        self.GAT = GATConv(in_channels=64, out_channels=64, heads=2)
        self.users_feature = nn.Parameter(torch.FloatTensor(x, 64))
        nn.init.xavier_normal_(users_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(y, 64))
        nn.init.xavier_normal_(items_feature)
        
        
    def forward(self, edge_index):
        ui_feat = torch.cat([self.users_feature,self.items_feature], dim = 0)
        return self.GAT(ui_feat, edge_index)
    def get_ui(self):
        return self.users_feature, self.items_feature

criterion = torch.nn.MSELoss()
model = Model(in_channels=64, out_channels=64, heads=2)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

uigraph.indices().max()

def train():
    model.train()
    total_loss = 0 
#     for batch in loader:
    optimizer.zero_grad()
    print(ui_feat.shape, uigraph.indices().shape)
    print(uigraph.indices())
    out = model(torch.tensor(uigraph.indices()))
    # Compute the loss using only the edges in the edge_index
    loss = criterion(out[uigraph.indices()[1]], uigraph.values().view(-1, 1))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    return total_loss
#     return total_loss / len(loader)

# Train the model for a number of epochs
num_epochs = 200
for epoch in range(1, num_epochs + 1):
    loss = train()
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')


### Running GAT in kaggle
class GAT(MessagePassing):
    def __init__ (
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True, 
        negative_slope: float =0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor,str] = 'mean',
        bias: bool = True,
        extra_layer=False,
        **kwargs
    ):
        super().__init__(node_dim = 0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.extra_layer = extra_layer
        if self.extra_layer:
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.extra_layer:
            self.lin_l.reset_parameters()
            self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        H, C = self.heads, self.out_channels
        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            if self.extra_layer:
                x_l = self.lin_l(x).view(-1, H, C)
                x_r = self.lin_r(x).view(-1, H, C)
            else:
                x = x.expand(self.heads, x.shape[0], x.shape[1]).transpose(0, 1)
                x_l = x_r = x.view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j



# Instantiate the model
model = GAT(in_channels=64, out_channels=64, heads=1)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.MSELoss()
def train():
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        # Compute the loss using only the edges in the edge_index
        loss = criterion(out[batch.edge_index[1]], batch.edge_attr.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Train the model for a number of epochs
num_epochs = 200
for epoch in range(1, num_epochs + 1):
    loss = train()
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Extract item representations after training
model.eval()
with torch.no_grad():
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        item_representations = out[num_users:]  # Extract item representations

print("Item Representations:")
print(item_representations)
