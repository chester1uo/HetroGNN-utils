import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv

class CustomHeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes, dropout_rate=0.5, use_bn=True, residual=True, layer_type='linear', num_heads=1):
        super(CustomHeteroRGCNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        self.residual = residual

        # Initializing different types of layers based on the layer_type parameter
        if layer_type == 'linear':
            self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size, bias=False) for name in etypes
            })
        elif layer_type == 'conv':
            self.weight = nn.ModuleDict({
                name: GraphConv(in_size, out_size, allow_zero_in_degree=True) for name in etypes
            })
        elif layer_type == 'gat':
            self.weight = nn.ModuleDict({
                name: GATConv(in_size, out_size, num_heads=num_heads, allow_zero_in_degree=True) for name in etypes
            })

        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_size * num_heads if layer_type == 'gat' else out_size)

        if self.residual:
            res_size = out_size * num_heads if layer_type == 'gat' else out_size
            if in_size != res_size:
                self.res_fc = nn.Linear(in_size, res_size, bias=False)
            else:
                self.res_fc = nn.Identity()

    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                Wh = self.dropout(Wh)
                if isinstance(self.weight[etype], GATConv):
                    Wh = Wh.flatten(1)  # Flatten the output from GATConv
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')

        result = {}
        for ntype in G.ntypes:
            if 'h' in G.nodes[ntype].data:
                h = G.nodes[ntype].data.pop('h')
                if self.use_bn:
                    h = self.bn(h)
                if self.residual:
                    res = self.res_fc(feat_dict[ntype])
                    h += res
                result[ntype] = h
        return result

class CustomHeteroRGCN(nn.Module):
    def __init__(self, config):
        super(CustomHeteroRGCN, self).__init__()
        self.config = config
        # Initialize embeddings for each node type
        self.embed = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(num_nodes, config['embedding_size']), requires_grad=True)
            for ntype, num_nodes in config['ntype_dict'].items()
        })
        # Proper initialization using Xavier Uniform
        for _, param in self.embed.items():
            nn.init.xavier_uniform_(param.data)

        self.layers = nn.ModuleList()
        layer_params = {
            'etypes': config['etypes'],
            'dropout_rate': config['dropout_rate'],
            'use_bn': config['use_bn'],
            'residual': config['use_residual'],
            'layer_type': config['layer_type'],
            'num_heads': config.get('num_heads', 1)  # defaulting to 1 if not specified
        }
        self.layers.append(CustomHeteroRGCNLayer(config['embedding_size'], config['hidden_size'], **layer_params))
        for _ in range(1, config['n_layers']):
            self.layers.append(CustomHeteroRGCNLayer(config['hidden_size'], config['hidden_size'], **layer_params))

        self.final_fc = nn.Linear(config['hidden_size'] * config.get('num_heads', 1) if config['layer_type'] == 'gat' else config['hidden_size'], config['out_size'])

    def forward(self, g, features):
        # Extract embeddings for each node type into a dictionary of tensors
        h_dict = {ntype: self.embed[ntype] for ntype in self.embed.keys()}
        # Additional handling for 'target' features if separate input features for 'target' nodes are provided
        if 'target' in features:
            h_dict['target'] = features['target']

        for i, layer in enumerate(self.layers):
            h_dict = layer(g, h_dict)
            if i < len(self.layers) - 1:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}

        return self.final_fc(h_dict['target'])
    
def build_model_from_config(path_to_config):
    with open(path_to_config, 'r') as file:
        config = yaml.safe_load(file)
    model = CustomHeteroRGCN(config)
    return model

# Example usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = build_model_from_config('../config/arch.yaml')
# model.to(device)
# print(model)