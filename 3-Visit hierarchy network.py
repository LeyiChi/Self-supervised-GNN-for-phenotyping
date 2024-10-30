#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[2]:


import torch
from torch_geometric.data import Data


# In[3]:


version = '0919'


# ### Load data

# In[11]:


df = pd.read_csv("data/data_"+version+".csv")
df.drop_duplicates(inplace = True)
df = df.reset_index(drop=True)
df.head()


# In[12]:


print('patient: ', len(set(df['PERSON_ID'])))
print('visit: ', len(set(df['VISIT_OCCURRENCE_ID'])))
print('event: ', len(set(df['EVENT'])))


# In[13]:


from collections import defaultdict
from tqdm.notebook import tqdm


# In[14]:


vidEventMap = defaultdict(list)
for i in tqdm(range(len(df))):
    pid = df.loc[i,'PERSON_ID']
    vid = df.loc[i,'VISIT_OCCURRENCE_ID']
    event = df.loc[i,'EVENT']
    vidEventMap[vid].append(event)


# In[15]:


pidVidMap = defaultdict(list)
for i in tqdm(range(len(df))):
    pid = df.loc[i,'PERSON_ID']
    vid = df.loc[i,'VISIT_OCCURRENCE_ID']
    event = df.loc[i,'EVENT']
    pidVidMap[pid].append(vid)



# ### edge construction

# In[91]:


df_patient_visit = df[['PERSON_ID', 'VISIT_OCCURRENCE_ID']].drop_duplicates()
df_visit_event = df[['VISIT_OCCURRENCE_ID', 'EVENT']].drop_duplicates()


# In[92]:


# record index for events, visits and patients
index = 0
event_index = {}
for event in set(df['EVENT']):
    event_index[event] = index
    index += 1


# In[93]:


visit_index = {}
for visit in set(df['VISIT_OCCURRENCE_ID']):
    visit_index[visit] = index
    index += 1


# In[94]:


patient_index = {}
for patient in set(df['PERSON_ID']):
    patient_index[patient] = index
    index += 1


# In[95]:


overall_index = {**event_index, **visit_index, **patient_index}


# In[96]:


with open('model_'+version+'/overall_index.pkl', "wb") as f:
    pickle.dump(overall_index, f)



# In[97]:


person_num = len(set(df['PERSON_ID']))
visit_num = len(set(df['VISIT_OCCURRENCE_ID']))
event_num = len(set(df['EVENT']))


# In[98]:


visit_patient_edge = []
for ind in df_patient_visit.index:
    temp = list(df_patient_visit.loc[ind])
    p = patient_index[temp[0]]
    v = visit_index[temp[1]]
    visit_patient_edge.append([v-event_num,p-event_num-visit_num])


# In[99]:


event_visit_edge = []
for ind in df_visit_event.index:
    temp = list(df_visit_event.loc[ind])
    v = visit_index[temp[0]]
    e = event_index[temp[1]]
    event_visit_edge.append([e,v-event_num])



# ### node embeddings

# In[100]:


from gensim.models import Word2Vec
from sklearn.preprocessing import normalize


# In[101]:


# load embeddings of events, visits and patients
cbow_model = Word2Vec.load("model_"+version+"/cbow.model")
with open("model_"+version+"/visit_embeddings_enhanced.pkl", "rb") as tf:
    visit_embeddings = pickle.load(tf)
with open("model_"+version+"/patient_embeddings.pkl", "rb") as tf:
    patient_embeddings = pickle.load(tf)



# In[102]:


event_list = []
for key in event_index:
    event_list.append(cbow_model.wv[key])
    
visit_list = []
for key in visit_index:
    visit_list.append(visit_embeddings[key])
    
patient_list = []
for key in patient_index:
    patient_list.append(patient_embeddings[key])


# In[103]:


# normalization
x = torch.tensor(event_list + visit_list + patient_list)
x = torch.tensor(normalize(x,axis=1), dtype=torch.float)


# In[104]:


event_tensor, visit_tensor, patient_tensor = torch.split(x, [event_num, visit_num, person_num], dim=0)



# ### graph construction

# In[106]:


from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


# In[107]:


data = HeteroData()


# In[108]:


data['event'].x = event_tensor
data['visit'].x = visit_tensor
data['patient'].x = patient_tensor



# In[109]:


event_visit_index = torch.tensor(event_visit_edge)
visit_patient_index = torch.tensor(visit_patient_edge)


# In[110]:


data['event', 'event_visit', 'visit'].edge_index = event_visit_index.t().contiguous()
data['visit', 'visit_patient', 'patient'].edge_index = visit_patient_index.t().contiguous()


# In[111]:


data = T.ToUndirected()(data)



# In[113]:


with open('model_'+version+'/layered_graph_hetero.pkl', "wb") as f:
    pickle.dump(data, f)



