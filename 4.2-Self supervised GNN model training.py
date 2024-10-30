#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import time
import pickle
import argparse
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


# In[2]:


version = '0919'


# ### graph partition

# In[3]:


with open("model_0919/layered_graph_hetero.pkl", "rb") as f:   
    data = pickle.load(f)


# In[5]:


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
data = data.to(device)


# In[6]:


from torch_geometric.loader import NeighborLoader


# In[7]:


loader = NeighborLoader(data, num_neighbors=[30, 30], batch_size=128, 
                        input_nodes=('patient', None), shuffle=True) #input_nodes=('paper', data['paper'].train_mask)


# In[8]:


total_num_nodes = 0
for step, sub_data in enumerate(loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
    print(sub_data)
    print()
    total_num_nodes += sub_data.num_nodes

print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')



# ### model training

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from GNN import GNNLayer


# In[10]:


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_input)

        self.dec_1 = Linear(n_input, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, z):
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar


# In[11]:


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_input) # full connection layer，softmax
        self.gnn_5 = GNNLayer(n_input, n_z) # full connection layer，softmax

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v
        
    def forward(self, data):
        # Graph
        homogeneous_data = data.to_homogeneous()
        adj = torch.sparse_coo_tensor(homogeneous_data.edge_index,torch.tensor(np.ones(homogeneous_data.num_edges, dtype=np.float32)).to(device),(homogeneous_data.num_nodes, homogeneous_data.num_nodes))
        adj = adj.to(device)

        # cluster parameter initiate; initial embeddings
        event_num = data['event'].x.shape[0]
        visit_num = data['visit'].x.shape[0]
        patient_num = data['patient'].x.shape[0]
        x = torch.cat([data['event'].x, data['visit'].x, data['patient'].x], 0)
        data_ev = torch.cat([data['event'].x, data['visit'].x], 0)

        # GCN Module
        z = self.gnn_1(x, adj)
        z = self.gnn_2(z, adj)
        z = self.gnn_3(z, adj)
        z = self.gnn_4(z, adj)
        
        # Step 2: gnn reconstruction
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        # loss F.binary_cross_entropy_with_logits(adj_rec, adj)
        
        # get patient nodes
        z_s = z[:event_num+visit_num].float()
        z_p = z[event_num+visit_num:].float()
        
        # Step 3: decoder for events, visits
        x_bar = self.ae(z_s)
        
        # Step 4 and 5
        predict = self.gnn_5(z_p, adj, active=False, last=True)
        q = 1.0 / (1.0 + torch.sum(torch.pow(predict.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v) 
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        predict = F.softmax(predict, dim=1)

        return data_ev, adj, adj_rec, z_s, z_p, x_bar, q, predict


# In[12]:


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


# In[10]:


def train_sdcn(epoch):
    for step, sub_data in enumerate(loader):
        data_ev, adj_sparse, adj_rec, z_s, z_p, x_bar, q, pred = model(sub_data)
        tmp_q = q.data
        p = target_distribution(tmp_q)

        #print('kl_loss')
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean') #clu_loss
        #print('rec_s_loss')
        rec_s_loss = F.mse_loss(x_bar, data_ev) # reconstruction loss of nodes
        adj_dense = adj_sparse.to_dense() #adj.half()
        #print('rec_g_loss')
        dimension = adj_rec.size()[0]
        temp = 0
        for i in range(dimension):
            temp += F.binary_cross_entropy_with_logits(adj_rec[i], adj_dense[i])# graph reconstruction loss
        rec_g_loss = temp/dimension

        #lr = 1e-3 * (0.1 ** (epoch // 20)) #adjust_learning_rate

        loss = kl_loss + 0.1 * rec_g_loss + rec_s_loss
        print('epoch{} step{} kl_loss: {}; rec_g_loss: {}; rec_s_loss: {}'.format(epoch, step, kl_loss, rec_g_loss * 0.1, rec_s_loss))
        print('epoch{} step{} loss: {}'.format(epoch, step, loss)) 

        optimizer.zero_grad()
        #print('loss backward')
        loss.backward()
        #print('optimizer')
        optimizer.step()

        del data_ev, adj_sparse, adj_rec, z_s, z_p, x_bar, q, pred
        start = time.time()
        


# In[14]:


torch.backends.cudnn.benchmark = True
lr = 1e-3
k = None #graph KNN
n_clusters = 10 #classes
n_z = 10
n_input = 100 #dimension
pretrain_path = 'data/ev_pretrain.pkl'

model = SDCN(500, 500, 1000, 1000, 500, 500,
            n_input=n_input,
            n_z=n_z,
            n_clusters=n_clusters,
            v=n_clusters-1).to(device)
optimizer = Adam(model.parameters(), lr=lr)


# In[15]:


try:
    model.load_state_dict(torch.load('model_0919/hetero_model.pkl'))
    print(model.eval())
except:
    pass


# In[17]:


start_time = time.time()
for epoch in range(200): #epoch        
    '''
        res1 = tmp_q.cpu().numpy().argmax(1)       #Q
        res2 = pred.data.cpu().numpy().argmax(1)   #Z
        res3 = p.data.cpu().numpy().argmax(1)      #P
    '''
    start = time.time()
    result = train_sdcn(epoch)
    print('time for one turn:' + str(time.time()-start) + 's')
print('time for whole model:' + str(time.time()-start_time) + 's')

torch.save(model.state_dict(), 'model/hetero_model.pkl')


# In[16]:


with torch.no_grad():
    data_ev, adj_sparse, adj_rec, z_s, z_p, x_bar, q, pred = model(data)


# In[19]:


with open('model_'+version+'/x_bar.pkl', "wb") as f:
    pickle.dump(x_bar, f)
with open('model_'+version+'/z_p.pkl', "wb") as f:
    pickle.dump(z_p, f)
with open('model_'+version+'/pred.pkl', "wb") as f:
    pickle.dump(pred, f)


# In[17]:


class_count = defaultdict(int)
for patient in pred:
    temp_class = 0
    max_prob = patient[0]
    for i in range(1,len(patient)):
        if patient[i] > max_prob:
            max_prob = patient[i]
            temp_class = i
    class_count[temp_class] += 1
class_count

