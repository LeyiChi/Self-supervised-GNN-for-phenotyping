#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from tqdm.notebook import tqdm
from collections import defaultdict


# In[2]:



import torch
from torch_geometric.data import Data


# In[3]:


version = '0706'


# ### Load data

# In[4]:


df = pd.read_csv("data/data_"+version+".csv")
df.drop_duplicates(inplace = True)
df.head()


# In[5]:


print('patient: ', len(set(df['PERSON_ID'])))
print('visit: ', len(set(df['VISIT_OCCURRENCE_ID'])))
print('event: ', len(set(df['EVENT'])))


# ### edge construction

# In[6]:


df_patient_visit = df[['PERSON_ID', 'VISIT_OCCURRENCE_ID']].drop_duplicates()
df_visit_event = df[['VISIT_OCCURRENCE_ID', 'EVENT']].drop_duplicates()


# In[7]:


# record index for events, visits and patients
index = 0
event_index = {}
for event in set(df['EVENT']):
    event_index[event] = index
    index += 1


# In[8]:


visit_index = {}
for visit in set(df['VISIT_OCCURRENCE_ID']):
    visit_index[visit] = index
    index += 1


# In[9]:


patient_index = {}
for patient in set(df['PERSON_ID']):
    patient_index[patient] = index
    index += 1


# In[10]:


overall_index = {**event_index, **visit_index, **patient_index}


# In[11]:


with open('model_'+version+'/overall_index.pkl', "wb") as f:
    pickle.dump(overall_index, f)


# In[11]:


person_num = len(set(df['PERSON_ID']))
visit_num = len(set(df['VISIT_OCCURRENCE_ID']))
event_num = len(set(df['EVENT']))


# In[12]:


visit_patient_edge = []
for ind in df_patient_visit.index:
    temp = list(df_patient_visit.loc[ind])
    p = patient_index[temp[0]]
    v = visit_index[temp[1]]
    visit_patient_edge.append([v-event_num,p-event_num-visit_num])


# In[13]:


event_visit_edge = []
for ind in df_visit_event.index:
    temp = list(df_visit_event.loc[ind])
    v = visit_index[temp[0]]
    e = event_index[temp[1]]
    event_visit_edge.append([e,v-event_num])


# ### node embeddings

# In[14]:


from gensim.models import Word2Vec
from sklearn.preprocessing import normalize


# In[15]:


# load embeddings of events, visits and patients
cbow_model = Word2Vec.load("model_0919/cbow.model")
with open("model_"+version+"/visit_embeddings_enhanced.pkl", "rb") as tf:
    visit_embeddings = pickle.load(tf)
with open("model_"+version+"/patient_embeddings.pkl", "rb") as tf:
    patient_embeddings = pickle.load(tf)


# In[16]:


event_list = []
for key in event_index:
    event_list.append(cbow_model.wv[key])
    
visit_list = []
for key in visit_index:
    visit_list.append(visit_embeddings[key])
    
patient_list = []
for key in patient_index:
    patient_list.append(patient_embeddings[key])


# In[17]:


# normalization
x = torch.tensor(event_list + visit_list + patient_list)
x = torch.tensor(normalize(x,axis=1), dtype=torch.float)


# In[18]:


event_tensor, visit_tensor, patient_tensor = torch.split(x, [event_num, visit_num, person_num], dim=0)


# ### graph construction

# In[20]:


from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


# In[21]:


data = HeteroData()


# In[22]:


data['event'].x = event_tensor
data['visit'].x = visit_tensor
data['patient'].x = patient_tensor


# In[23]:


event_visit_index = torch.tensor(event_visit_edge)
visit_patient_index = torch.tensor(visit_patient_edge)


# In[24]:


data['event', 'event_visit', 'visit'].edge_index = event_visit_index.t().contiguous()
data['visit', 'visit_patient', 'patient'].edge_index = visit_patient_index.t().contiguous()


# In[25]:


data = T.ToUndirected()(data)


# In[27]:


with open('model_'+version+'/layered_graph_hetero.pkl', "wb") as f:
    pickle.dump(data, f)


# ### graph partition

# In[28]:


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
data = data.to(device)


# In[29]:


from torch_geometric.loader import NeighborLoader


# In[30]:


loader = NeighborLoader(data, num_neighbors=[30, 30], batch_size=128, 
                        input_nodes=('patient', None), shuffle=True) #input_nodes=('paper', data['paper'].train_mask)


# In[31]:


total_num_nodes = 0
for step, sub_data in enumerate(loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
    print(sub_data)
    print()
    total_num_nodes += sub_data.num_nodes

print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')



# ### model validation on the testing data

# In[32]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from GNN import GNNLayer
from torch.optim import Adam


# In[33]:


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


# In[34]:


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
        #loss F.binary_cross_entropy_with_logits(adj_rec, adj)
        
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


# In[35]:


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


# In[36]:


try:
    model.load_state_dict(torch.load('model_0919/hetero_model.pkl'))
    print(model.eval())
except:
    pass


# In[37]:


with torch.no_grad():
    data_ev, adj_sparse, adj_rec, z_s, z_p, x_bar, q, pred = model(data)


# In[38]:


class_count = defaultdict(int)
for patient in pred:
    temp_class = 0
    max_prob = patient[0]
    for i in range(1,len(patient)):
        if patient[i] > max_prob:
            max_prob = patient[i]
            temp_class = i
    class_count[temp_class] += 1


# In[39]:


with open("model_0919/x_bar.pkl", "rb") as tf:
    x_bar_0919 = pickle.load(tf)
with open("model_0919/z_p.pkl", "rb") as tf:
    z_p_0919 = pickle.load(tf)
with open("model_0919/pred.pkl", "rb") as tf:
    pred_0919 = pickle.load(tf)


# ### Visualization

# In[40]:


import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from utils import *


# In[41]:


event_num = data['event'].x.shape[0]
visit_num = data['visit'].x.shape[0]
patient_num = data['patient'].x.shape[0]


# In[42]:


df = pd.read_csv("data/data_"+version+".csv")
df.head()


# In[43]:


event_type = df[['EVENT','TYPE']].drop_duplicates(ignore_index=True)
visit_list = list(set(df['VISIT_OCCURRENCE_ID']))
person_list = list(set(df['PERSON_ID']))
with open("model_"+version+"/overall_index.pkl", "rb") as tf:
    overall_index = pickle.load(tf)


# In[51]:


tsnePatient = TSNE(perplexity=25, n_iter=250).fit_transform(z_p)
tsnePatient_0919 = TSNE(perplexity=25, n_iter=250).fit_transform(z_p_0919)


# In[44]:


patient_class = {}
for i in range(len(pred)):
    temp_class = 0
    max_prob = pred[i][0]
    for j in range(1,len(pred[i])):
        if pred[i][j] > max_prob:
            max_prob = pred[i][j]
            temp_class = j
    patient_class[i] = temp_class


# In[45]:


patient_class_temp = {}
for i in range(len(pred_0919)):
    temp_class = 0
    max_prob = pred_0919[i][0]
    for j in range(1,len(pred_0919[i])):
        if pred_0919[i][j] > max_prob:
            max_prob = pred_0919[i][j]
            temp_class = j
    patient_class_temp[i] = temp_class



# In[56]:


header = ['PID','CLUSTER']
event_list = list(set(df['EVENT']))
for event in event_list:
    header.append(event)


# In[57]:


pidEventMap = defaultdict(list)
for i in tqdm(range(len(df))):
    pid = df.loc[i,'PERSON_ID']
    event = df.loc[i,'EVENT']
    pidEventMap[pid].append(event)


# In[58]:


with open ('model_'+version+'/pid_cluster_shier.csv','w',encoding='utf-8',newline='') as f:
    writer =csv.writer(f)
    writer.writerow(header)

    for i in tqdm(range(len(person_list))):
        event_multihot = []
        for event in event_list:
            if event in pidEventMap[person_list[i]]:
                event_multihot.append(1)
            else:
                event_multihot.append(0)
        row = [person_list[i], patient_class[i]]+event_multihot
        writer.writerows([row])


# ### UMAP

# In[46]:


#import umap
import umap.umap_ as umap


# In[57]:


umapPatient = umap.UMAP(n_neighbors=30,min_dist=0.5,metric='correlation').fit_transform(z_p)


# In[48]:


umapPatient_0919 = umap.UMAP().fit_transform(z_p_0919)


# In[65]:


import os
print(os.__file__)


# In[61]:


#color = ["black",'tan',"green","violet",'#1E90FF',"darkgray",'#FFB451','#1E90FF','yellow','pink']
color = ['#1E90FF','#FFB451']
phenotype = {4:1, 6:2, 7:1, 3:1}


# In[62]:


patient_class_list_0919 = list(patient_class_temp.values())
patient_class_list_0919 = list(map(lambda x:phenotype[x], patient_class_list_0919))
patient_class_list_0919 = np.array(patient_class_list_0919)
patient_class_list_shier = list(patient_class.values())
patient_class_list_shier = list(map(lambda x:phenotype[x], patient_class_list_shier))
patient_class_list_shier = np.array(patient_class_list_shier)


# In[65]:


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
font = {'family':'Arial','size':5,}

plt.figure(figsize=(4,2))

ax = plt.subplot(121)
for i in set(patient_class_list_0919):
    point_color = color[i-1]
    x = umapPatient_0919[patient_class_list_0919==i]
    sc = ax.scatter(x[:,0], x[:,1], c = point_color, edgecolors = 'none', label='Subphenotype '+ str(i), s = 0.5)

plt.title('A', loc='left', fontsize =  6)
plt.xticks([])
plt.yticks([])
plt.legend(loc=2, prop=font, frameon=False, markerscale = 3)


ax = plt.subplot(122)
for i in set(patient_class_list_shier):
    point_color = color[i-1]
    x = umapPatient[patient_class_list_shier==i]
    sc = ax.scatter(x[:,0], x[:,1], c = point_color, edgecolors = 'none', label='Subphenotype '+ str(i), s = 0.5)
plt.title('B', loc='left', fontsize =  6)
plt.xticks([])
plt.yticks([])
plt.legend(loc=2, prop=font, frameon=False, markerscale = 3)


plt.savefig('model_0706/patient_c2.tiff', bbox_inches='tight', pad_inches=0.01)
plt.savefig('model_0706/patient_c2.png', bbox_inches='tight', pad_inches=0.01)

plt.show()

