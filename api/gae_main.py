from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
import torchvision
from torch import optim
from scipy import sparse
import pandas as pd
import os

from .model import GCNModelVAE
from .facemesh_mediapipe import mediapipe_facemesh
from .optimizer import loss_function
from .utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
# from facemesh_mediapipe import mediapipe_facemesh
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=234, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=117, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=64, help='Number of units in hidden layer 3.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', default='cora', help='type of dataset.')

args, unknown = parser.parse_known_args()
# print("epoch=",args.hidden1)
# torch.manual_seed(1)

def gae_for(path):
    # Calling mediapipe for adjacency matrix and feature matrix
    # adj: Type=scipy.sparse._csr.csr_matrix, Shape=(468,468)
    # Features: Type=torch.Tensor, Shape=(468,468)
    adj,features, image=mediapipe_facemesh(path)
    # print(adj)
    # adj = mediapipe_facemesh(path)
    # FEATURE_MATRIX=np.zeros((468,468))
    # for i in range(468):
    #     FEATURE_MATRIX[i][i]=1
    # features = torch.from_numpy(FEATURE_MATRIX)
    # print("adj=",adj)
    # print("features=",features)
    # n_nodes= 468, feat_dim=468
    n_nodes, feat_dim = features.shape 
    adj_orig = adj
    # adj_orig.diagonal() = all diagonal elements. Eg: [0,0,0,.....0], Shape = (1,468), 1 Dimensional
    # adj_orig.diagonal()[np.newaxis, :] = [[0,0,0,....,0]], Shape = (1,468), 2 Dimensional
    # sp.dia_matrix = sparse matrix with diagonal storage
    # sp.dia_matrix((data,offsets),shape) : data = [[0,0,0,...,0]], offsets = [0], and shape = (468,468)
    # subtractiong adj matrix with 468*468 matrix of zeros. ( Remove diagonal elements )
    # print(adj_orig.shape)
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape) 
    # eliminating zeros
    
    adj_orig.eliminate_zeros()
    # calling mask_test_edges() function, present in utils
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj) 
    # 198 edges and lower triangular values are removed from original matrix. 
    # adj = adj_train
    adj_train=adj
    # print("val_edges_false=",len(test_edges_false))
    # print(train_edges.shape,test_edges,test_edges_false)
    # normalize adj matrix : Tilda A = (adj + feature) * pow(D,.5) * pow(D,.5)
    adj_norm = preprocess_graph(adj)
    # adj + features
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # cerating array of tensor type
    adj_label = torch.FloatTensor(adj_label.toarray())
    # 468 * 468 - 2248 / 2248
    # pos_weight= tensor([96.4306])
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    # 468 * 468 / ((468 * 468) - 2248) * 2
    # norm= 0.5051850758386537
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    # calling GCNModelVAE, passing shape, hidden1 hidden2, dropout
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    # print("model=",model)
    # predefined in torch
    # Implements Adam algorithm
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(args.epochs):
        # This method returns the time as a floating point number expressed in seconds
        t = time.time()
        # predefined in torch
        model.train()
        # predefined in torch
        # Sets the gradients of all optimized torch.Tensor s to zero.
        optimizer.zero_grad()
        recovered, mu, logvar, z = model(features, adj_norm)
        # print("shape=",recovered.shape)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr, emb = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        # print("roc=",roc_curr)
    # print("Optimization Finished!")
    
    roc_score, ap_score, emb = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
    #     "val_ap=", "{:.5f}".format(ap_curr),
    #     "time=", "{:.5f}".format(time.time() - t)
    #     )
    # print("roc=",roc_score)
    # print("epoch=",args.epochs)
    # print("emb",hidden_emb.shape)
    # print(recovered.shape,mu.shape,logvar.shape)
    return(z, adj, features, image)
# path="https://mymodernmet.com/wp/wp-content/uploads/2019/09/100k-ai-faces-6.jpg"
# gae_for(path)
