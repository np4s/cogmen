import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import TransformerConv, RGCNConv

class SeqContext(nn.Module):
    def __init__(self, D_m, D_e, drop_rate=0.5, nhead=1, nlayer=1, no_cuda=False):
        super(SeqContext, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_m,
            nhead=nhead,
            dropout=drop_rate,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayer)
        self.transformer_out = torch.nn.Linear(D_m, D_e, bias=True)

    def forward(self, x):
        rnn_out = self.transformer_encoder(x)
        rnn_out = self.transformer_out(rnn_out)
        return rnn_out
    
class GNN(nn.Module):
    def __init__(self, num_features, num_relations, D_h, gnn_nhead=1, no_cuda=False):
        super(GNN, self).__init__()
        self.conv1 = RGCNConv(num_features, D_h, num_relations)
        self.conv2 = TransformerConv(D_h, D_h, heads=gnn_nhead, concat=True)
        self.bn = nn.BatchNorm1d(D_h*gnn_nhead)
        
    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index)
        x = self.bn(x)
        x = F.leaky_relu(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        # print(input_dim, hidden_size, n_classes)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.nll_loss = nn.NLLLoss()
        
    def get_prob(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        log_prob = F.log_softmax(x, dim=-1)
        return log_prob
    
    def get_loss(self, x, labels):
        log_prob = self.get_prob(x)
        loss = self.nll_loss(log_prob, labels)
        return log_prob, loss

def edge_perms(length, window_past, window_future):
    all_perms = set()
    array = np.arange(length)
    for i in range(length):
        perms = set()
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(length, i+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, i-window_past):]
        else:
            eff_array = array[max(0, i-window_past):min(length, i+window_future+1)]

        for item in eff_array:
            perms.add((i, item))
        all_perms = all_perms.union(perms)

    return list(all_perms)


def batch_graphify(features, lengths, speakers, window_past, window_future, edge_type_mapping, no_cuda=False):
    node_features, edge_index, edge_type = [], [], []
    batch_size = len(lengths)
    length_sum = 0

    for j in range(batch_size):
        cur_len = lengths[j]
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, window_past, window_future)
        perms_rec = [(item[0]+length_sum, item[1]+length_sum)
                     for item in perms]
        length_sum += cur_len
        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            speaker0 = (speakers[item[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (speakers[item[1], j, :] == 1).nonzero()[0][0].tolist()
            c = '0' if item[0] < item[1] else '1'
            edge_type.append(
                edge_type_mapping[str(speaker0) + str(speaker1) + c])

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_type = torch.tensor(edge_type).long()

    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_type = edge_type.cuda()

    return node_features, edge_index, edge_type

class COGMEN(nn.Module):
    def __init__(self, modal, D_A, D_V, D_L, D_e, D_h, n_speakers, window_past, window_future, seqcontext_nlayer=1, gnn_nhead=1, n_classes=6, dropout=0.1, no_cuda=False):
        """
        Args:
            modal (_type_): _description_
            D_A (_type_): audio feature dimension
            D_V (_type_): visual feature dimension
            D_L (_type_): text feature dimension
            D_e (_type_): feature dimension after seqcontext
            D_h (_type_): hidden dimension of GNN
            n_speakers (_type_): _description_
            max_seq_len (_type_): _description_
            window_past (_type_): _description_
            window_future (_type_): _description_
            n_classes (int, optional): _description_. Defaults to 7.
            dropout (float, optional): _description_. Defaults to 0.5.
            no_cuda (bool, optional): _description_. Defaults to False.
        """
        super(COGMEN, self).__init__()
        self.modal = modal
        self.no_cuda = no_cuda

        if modal == 'avl':
            D_m = D_A + D_V + D_L
        num_relations = 2 * n_speakers ** 2

        self.window_past = window_past
        self.window_future = window_future

        self.seqcontext = SeqContext(
            D_m, D_e, drop_rate=dropout, nlayer=seqcontext_nlayer, no_cuda=no_cuda)
        self.gnn = GNN(D_e, num_relations, D_h, gnn_nhead=gnn_nhead, no_cuda=no_cuda)
        self.classifier = Classifier(D_h*gnn_nhead, D_h, n_classes, dropout=dropout)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) +
                                  '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) +
                                  '1'] = len(edge_type_mapping)
        self.edge_type_mapping = edge_type_mapping
        
    def forward(self, textf, lengths, speakers, labels=None, acouf=None, visuf=None):
        """
        Args:
            x (_type_): seq_len x batch_size x feature_dim
            lengths (_type_): batch_size x seq_len
            speakers (_type_): batch_size x seq_len
            labels (_type_): batch_size x seq_len
        """
        x = torch.cat((textf, acouf, visuf), dim=-1)
        x = torch.permute(x, (1, 0, 2))
        # print(x.shape)
        
        x = self.seqcontext(x)
        x, edge_index, edge_type = batch_graphify(
            x, lengths, speakers, self.window_past, self.window_future, self.edge_type_mapping, no_cuda=self.no_cuda)
        x = self.gnn(x, edge_index, edge_type)
        
        if labels is None:
            log_prob = self.classifier.get_prob(x)
            return log_prob
        
        log_prob, loss = self.classifier.get_loss(x, labels)
        return log_prob, loss
