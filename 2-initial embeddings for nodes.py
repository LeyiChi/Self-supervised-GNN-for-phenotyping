#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import time
import pickle


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# ## Load data

# In[3]:


df = pd.read_csv("data/data_0919.csv")
df.head()


# In[4]:


vidEventMap = {}
for i in tqdm(range(len(df))):
    vid = df.loc[i,'VISIT_OCCURRENCE_ID']
    event = df.loc[i,'EVENT']
    if vid in vidEventMap: 
        vidEventMap[vid].append(event)
    else: 
        vidEventMap[vid] = [event]


# In[5]:


pidVidMap = {}
for i in tqdm(range(len(df))):
    pid = df.loc[i,'PERSON_ID']
    vid = df.loc[i,'VISIT_OCCURRENCE_ID']
    if pid in pidVidMap: 
        pidVidMap[pid].append(vid)
    else: 
        pidVidMap[pid] = [vid]
        
for key in pidVidMap:
    pidVidMap[key] = list(set(pidVidMap[key]))




# ### word embedding for events using CBOW

# In[6]:


from gensim.test.utils import common_texts
from gensim.models import Word2Vec


# In[7]:


text = list(vidEventMap.values())


# In[8]:


try:
    model = Word2Vec.load("model_0919/cbow.model")
except:
    model = Word2Vec(sentences=text, vector_size=100, window=5, min_count=1, workers=4, epochs=100)
    model.save("model_0919/cbow.model")


# In[11]:


vidEmbedMap = {}
for visit in vidEventMap:
    embed = []
    for event in vidEventMap[visit]:
        embed.append(model.wv[event])
    vidEmbedMap[visit] = np.array(embed)





# ### initial embeddings for the visited nodes using an LSTM autoencoder model. 

# In[12]:


from LSTMAE import RecurrentAutoencoder
import random


# In[3]:


device = ('cuda' if torch.cuda.is_available() else 'cpu')


# In[14]:


INPUT_FEATURES_NUM = 100
OUTPUT_FEATURES_NUM = 100


# In[15]:


def train_event_lstm_enhanced(model):
    max_epochs = 18
    total_visit = 0
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    start_time = time.time()
    for epoch in range(max_epochs):
        batch_count = 0
        total_loss = 0
        loss = 0
        key_ls = list(vidEventMap.keys())
        random.shuffle(key_ls)
        for visit in key_ls:
            optimizer.zero_grad()
            for i in range(4):
                input_x = vidEmbedMap[visit]
                if i != 0:
                    random.shuffle(input_x)
                seq_len = len(input_x)
                input_x = torch.from_numpy(input_x.reshape(seq_len,1,INPUT_FEATURES_NUM))
                input_x = input_x.to(device)
                output_c, output = model(input_x)
                loss += loss_function(output, input_x)
                batch_count += 1
                total_visit += 1
            
            if total_visit % 10000 == 0:
                print('Visits {}, Recent Loss: {:.5f}'.format(total_visit, loss.item()))
            if batch_count % 128 == 0:
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                loss = 0

        lr = 1e-2 * (0.1 ** (epoch // 3)) #adjust_learning_rate
        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Total Loss: {:.5f}'.format(epoch+1, max_epochs, total_loss))
            print("The loss value is reached")
            break
        
        print('Epoch [{}/{}], Total Loss: {:.5f}'.format(epoch+1, max_epochs, total_loss))
        print('Time elapsed: {} seconds'.format(time.time() - start_time))
        
    return model


# In[16]:


try:
    lstm_model_enhanced = torch.load("model_0919/event_lstm_enhanced.pkl")
except:
    lstm_model_enhanced = RecurrentAutoencoder(INPUT_FEATURES_NUM, OUTPUT_FEATURES_NUM).to(device)
    lstm_model_enhanced = train_event_lstm_enhanced(lstm_model_enhanced)
    print('LSTM model:', lstm_model_enhanced)
    torch.save(lstm_model_enhanced, 'model_0919/event_lstm_enhanced.pkl')


# In[17]:


with torch.no_grad():
    visit_embeddings_enhanced = {}
    for visit in tqdm(vidEmbedMap):
        event_emb = vidEmbedMap[visit]
        temp = torch.from_numpy(event_emb.reshape(len(event_emb),1,INPUT_FEATURES_NUM)).to(device)
        output_c, output = lstm_model_enhanced(temp)
        visit_embeddings_enhanced[visit] = np.array(output_c[-1].view(-1).cpu())


# In[78]:


with open('model_0919/visit_embeddings_enhanced.pkl', "wb") as f:
    pickle.dump(visit_embeddings_enhanced, f)





# ### initial embeddings for the patient nodes using an LSTM autoencoder model.

# In[18]:


def train(model):
    max_epochs = 30
    total_visit = 0
    lr = 1e-3
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    for epoch in range(max_epochs):
        batch_count = 0
        total_loss = 0
        loss = 0
        key_ls = list(pidVidMap.keys())
        random.shuffle(key_ls)
        for patient in key_ls:
            optimizer.zero_grad()
            input_x = []
            for visit in pidVidMap[patient]:
                input_x.append(visit_embeddings_enhanced[visit])
            seq_len = len(input_x)
            input_x = torch.from_numpy(np.array(input_x).reshape(seq_len,1,INPUT_FEATURES_NUM))
            input_x = input_x.to(device)
            output_c, output = model(input_x)
            loss += loss_function(output, input_x)
            batch_count += 1
            total_visit += 1
            if batch_count % 32 == 0:
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                loss = 0

        lr = 1e-3 * (0.1 ** (epoch // 3)) #adjust_learning_rate
        print('Epoch [{}/{}], Total Loss: {:.5f}'.format(epoch+1, max_epochs, total_loss))
        print('Time elapsed: {} seconds'.format(time.time() - start_time))
        
    return model


# In[19]:


try:
    visit_lstm_model = torch.load("model_0919/visit_lstm.pkl")
except:
    visit_lstm_model = RecurrentAutoencoder(INPUT_FEATURES_NUM, OUTPUT_FEATURES_NUM).to(device)
    visit_lstm_model = train(visit_lstm_model)
    print('LSTM model:', visit_lstm_model)
    torch.save(visit_lstm_model, 'model_0919/visit_lstm.pkl')


# In[20]:


with torch.no_grad():
    patient_embeddings = {}
    for patient in tqdm(pidVidMap):
        input_x = []
        for visit in pidVidMap[patient]:
            input_x.append(visit_embeddings_enhanced[visit])
        seq_len = len(input_x)
        input_x = torch.from_numpy(np.array(input_x).reshape(seq_len,1,INPUT_FEATURES_NUM))
        input_x = input_x.to(device)
        output_c, output = visit_lstm_model(input_x)
        patient_embeddings[patient] = np.array(output_c[-1].view(-1).cpu())


# In[82]:


with open('model_0919/patient_embeddings.pkl', "wb") as f:
    pickle.dump(patient_embeddings, f)


# In[ ]:





# ### visualization of the distribution of nodes

# In[21]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


# In[22]:


count = 0
event_index = {}   
event_embed = []
for event in set(df['EVENT']):
    event_embed.append(model.wv[event])
    event_index[event] = count
    count += 1
event_embed = np.array(event_embed)


# In[23]:


count = 0
visit_index = {}
for visit in visit_embeddings_enhanced:
    visit_index[visit] = count
    count += 1


# In[24]:


count = 0
patient_index = {}
for patient in patient_embeddings:
    patient_index[patient] = count
    count += 1



# #### patient embeddings

# In[131]:


import csv


# In[110]:


df_result = pd.read_csv("data/patient_label.csv")


# In[111]:


print('outcomes：', set(df_result['outcome']))
result_dic = {}
num = 0
for e in set(df_result['outcome']):
    result_dic[e] = num
    num += 1


# In[112]:


M = np.array(list(patient_embeddings.values()))
tsnePatient = TSNE().fit_transform(M)


# In[114]:


f = plt.figure(figsize=(10,10))
ax = plt.subplot(aspect='equal')
for pid in tqdm(patient_index):
    ind = patient_index[pid]
    try:
        result = list(df_result[df_result['PERSON_ID'] == pid]['outcome'])[0]
    except:
        pass
    if result == 'death':
        point_color = 'red'
        sc = ax.scatter(tsnePatient[ind,0], tsnePatient[ind,1], c = point_color, s = 4) #, label='survive'+ str(lasted_yrs // 3 * 3) + 'years'
    else:
        point_color = 'black'
        sc = ax.scatter(tsnePatient[ind,0], tsnePatient[ind,1], c = point_color, s = 1)
    
plt.xlim(-50,50)
plt.ylim(-50,50)
ax.axis('tight')
plt.show()


