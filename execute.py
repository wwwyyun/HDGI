import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import DGI, LogReg
from utils import process
import argparse
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
#from torchsummary import summary
import time


parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')

parser.add_argument('--dataset', type=str, default="cora", help='name of dataset (default: cora)')
parser.add_argument('--k', type=int, default=7, help='num of communitues')
parser.add_argument('--dim_embed', type=int, default=64, help='dim of output embedding')
parser.add_argument('--lr_embed', type=float, default=0.01, help='learning rate for node embedding')
parser.add_argument('--lr_node_class', type=float, default=0.01, help='learning rate for node classification')
parser.add_argument('--weight_decay_embed', type=float, default=0.0, help='weight decay for node embedding')
parser.add_argument('--weight_decay_node_class', type=float, default=0.0, help='weight decay for node classification')
parser.add_argument('--nb_epochs_embed', type=int, default=10000, help='num of epochs for graph embedding')
parser.add_argument('--nb_epochs_node_class', type=int, default=100, help='num of epochs for node classification')
parser.add_argument('--patience', type=int, default=20, help='for early stopping')
parser.add_argument('--sparse', type=bool, default=True, help='sparse or not')
parser.add_argument('--nonlinearity', type=str, default='prelu', help='activation function in GNN layers')
parser.add_argument('--alpha_topo', type=float, default='1.0', help='hyperparameter for topology smoothness loss')
parser.add_argument('--device', type=str, default='cuda:0', help='specify which gpu to use')
parser.add_argument('--hard', type=bool, default=True, help='hard assignment or soft assignment')

def entropy(x):
    p = torch.div(x, torch.sum(x)+1e-8)
    b = p * (torch.log(p+1e-8))
    b = -1.0 * b.sum()
    return b

def prune(edge_index, y):
    row, col = edge_index
    y_r, y_c = y[row], y[col]
    dot_sim = torch.bmm(
        y_r.view(y_r.size(0), 1, y_r.size(1)), y_c.view(y_c.size(0), y_c.size(1), 1)
    ).view(edge_index.size(1))
    return dot_sim

def train(args):
    dataset = args.dataset
    k = args.k
    dim_embed = args.dim_embed  
    lr_embed = args.lr_embed
    weight_decay_embed = args.weight_decay_embed
    nb_epochs_embed = args.nb_epochs_embed
    patience = args.patience
    sparse = args.sparse
    nonlinearity = args.nonlinearity
    alpha_topo = args.alpha_topo
    device = args.device
    hard = args.hard
    # adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    adj, features, _, _, _, _ = process.load_data(dataset)
    features, _ = process.preprocess_features(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]   
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()
    
    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    
    model = DGI(ft_size, dim_embed, nonlinearity, k, device, hard, nb_nodes)
    print(model)
#    for name, param in model.named_parameters():
#        if param.requires_grad:
#                print(name, param.data)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr_embed, weight_decay=weight_decay_embed)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.to(device)
        features = features.to(device)
        if sparse:
            sp_adj = sp_adj.to(device)
            edge_list = sp_adj._indices()
        else:
            adj = adj.to(device)
            edge_list = adj[indices]
            
    b_xent = nn.BCEWithLogitsLoss().to(device)
    l2_loss = nn.MSELoss(reduction='mean').to(device)
    logsoftmax = nn.LogSoftmax(dim=1).to(device)
    softmax = nn.Softmax(dim=1).to(device)
    kl_loss = torch.nn.KLDivLoss().to(device)
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs_embed):
        model.train()
        optimiser.zero_grad()
    
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
    
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
    
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.to(device)
            lbl = lbl.to(device)
        
        logits, c, assignmat, node_embed = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None) 
        disc_loss = b_xent(logits, lbl)
        assign = prune(edge_list, assignmat)
        topo_smoothness_loss = l2_loss(assign, torch.ones(assign.shape).to(device))
        # node_embed helps community_embed
#        with torch.no_grad():
#            community_embed_ = torch.div(torch.mm(torch.t(assignmat), torch.squeeze(node_embed)),\
#                torch.sum(torch.t(assignmat), dim=1, keepdim=True)+1e-8)
        community_embed_ = torch.div(torch.mm(torch.t(assignmat.detach()), torch.squeeze(node_embed)),\
            torch.sum(torch.t(assignmat), dim=1, keepdim=True)+1e-8)
        y = softmax(community_embed_)
        print('community_embed_ in exe: ', community_embed_.requires_grad)
        x = logsoftmax(c)
        # n2c_loss = l2_loss(c, community_embed_)
        n2c_loss = kl_loss(x, y)
        count = torch.sum(assignmat, dim=0)
        print('count: ', count)
        entropy_loss = entropy(count)
        loss = disc_loss + 3*alpha_topo*topo_smoothness_loss  - 1*entropy_loss + 1.*n2c_loss 
        print("Epoch {0}: disc = {1}, topo = {2}, loss_n2c = {4}, entropy_loss = {5}, loss = {3}".\
            format(epoch, disc_loss, topo_smoothness_loss, loss, n2c_loss, entropy_loss))
    
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1
    
        if cnt_wait == patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    print('c: ', c)
    model.load_state_dict(torch.load('best_dgi.pkl'))
    #get node embedding
    model.eval()
    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    return embeds

def node_classcification(embeds, args):
    dataset = args.dataset
    dim_embed = args.dim_embed
    weight_decay_node_class = args.weight_decay_node_class
    nb_epochs_node_class = args.nb_epochs_node_class
    lr_node_class = args.lr_node_class
    device = args.device
    _, _, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]
    
    if torch.cuda.is_available():
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)
    
    nb_classes = labels.shape[1]
    tot = torch.zeros(1)
    tot = tot.to(device)
    
    accs = []
    
    xent = nn.CrossEntropyLoss()
    for _ in range(50):
        log = LogReg(dim_embed, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.to(device)
    
        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.to(device)
        for _ in range(nb_epochs_node_class):
            log.train()
            opt.zero_grad()
    
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            
            loss.backward()
            opt.step()
    
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        #print(acc)
        tot += acc
    
    print('Average accuracy:', tot / 50)
    
    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())

def tsne(embed, k, dataset):
    embed = torch.squeeze(embed).cpu().numpy()
    embed_tsne = TSNE(learning_rate=100).fit_transform(embed)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(embed.shape)
    print(embed_tsne.shape)
    _, _, labels, _, _, _ = process.load_data(dataset)
    target = [np.where(r==1)[0][0] for r in labels]
    #print(target.shape)
    ax.scatter(embed_tsne[:,0], embed_tsne[:,1], c=target)
    fig.savefig('tsne.png')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    embed = train(args)
    node_classcification(embed, args)
    #tsne(embed, args.k, args.dataset)
