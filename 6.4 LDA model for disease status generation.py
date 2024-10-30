#!/usr/bin/env python
# coding: utf-8

# In[141]:


import warnings
warnings.filterwarnings('ignore')


# In[142]:


import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import time
from tableone import TableOne
import graphviz


# ### data merging

# In[143]:


# visit label
df = pd.read_csv("vid_cluster_zheyi.csv")
df.columns = ['VISIT_OCCURRENCE_ID','GNN_LABEL']


# In[144]:


# visit time
df_visit = pd.read_csv("../data/LXY_CKD_VISIT_0810.csv")
df_visit = df_visit[['PERSON_ID','VISIT_OCCURRENCE_ID','VISIT_START_DATE','VISIT_END_DATE']]
df_visit['VISIT_START_DATE'] = pd.to_datetime(df_visit['VISIT_START_DATE'], errors = 'coerce')
df_visit['VISIT_START_DATE'] = df_visit['VISIT_START_DATE'].dt.date
df_visit['VISIT_END_DATE'] = pd.to_datetime(df_visit['VISIT_END_DATE'], errors = 'coerce')
df_visit['VISIT_END_DATE'] = df_visit['VISIT_END_DATE'].dt.date


# In[145]:


# patient label
df_pid = pd.read_csv("pid_cluster_zheyi.csv")
df_pid = df_pid[['PID','CLUSTER']]
df_pid.columns = ['PERSON_ID', 'SUBTYPE']
df_pid.loc[df_pid['SUBTYPE'].isin([4,7]), 'SUBTYPE'] = 1
df_pid.loc[df_pid['SUBTYPE']==6, 'SUBTYPE'] = 2


# In[146]:


df = df.merge(df_visit, how='left')
df = df.merge(df_pid, how='left')


# In[147]:


# event logs
df_logs = pd.read_csv("zheyi_data_logs_combined.csv")


# In[150]:


# merge events in a visit
df_text = df_logs.groupby(['PERSON_ID','VISIT_OCCURRENCE_ID']).agg({'EVENT':' '.join}).reset_index()


# In[151]:


df = df.merge(df_text, how='left')
df = df[df['EVENT']==df['EVENT']] # drop event is null


# In[152]:


df = df[['PERSON_ID', 'SUBTYPE',
         'VISIT_OCCURRENCE_ID', 'EVENT', 'VISIT_START_DATE', 'VISIT_END_DATE', 'GNN_LABEL']]
df.head()


# ### LDA model

# In[153]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[154]:


data = list(df['EVENT'])


# In[155]:


# Prepare Stopwords
stop_words = []


# In[156]:


# Create the Document-Word matrix
vectorizer = CountVectorizer(stop_words = stop_words,
                             lowercase = False)
data_vectorized = vectorizer.fit_transform(data)
data_vectorized.shape


# In[157]:


vectorizer.get_feature_names()


# In[158]:


# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")


# In[159]:


def get_keywords_dict(model):
    # Topic-word Probalility Matrix
    w = model.components_
    proba = np.apply_along_axis(lambda x: x/x.sum(),1,w)
    df_topic_word_proba = pd.DataFrame(proba.T)
        
    # Assign Column and Index
    topicnames = ['Topic' + str(i) for i in range(1,model.n_components+1)]
    df_topic_word_proba.columns = topicnames
    df_topic_word_proba.index = vectorizer.get_feature_names()

    # Keywords
    df_topic_keywords = pd.DataFrame(columns=['Topic', 'No', 'Word', 'Proba', 'Cumsum'])
    for k in range(1,model.n_components+1):
        topic = 'Topic' + str(k)
        n = 0
        cumsum = 0
        while cumsum < 0.8:
            word_idx = (-df_topic_word_proba[topic]).argsort()[n]
            word = df_topic_word_proba.index[word_idx]
            proba = df_topic_word_proba.loc[word,topic]
            # if proba < 0.01:
            #     break
            cumsum +=  proba
            df_topic_keywords.loc[len(df_topic_keywords)] = [topic, n+1, word, proba, cumsum]
            n += 1
    # print(df_topic_keywords)
    
    # Save to Dict
    keywords_dict = {}
    for k in range(1,model.n_components+1):
        topic = 'Topic' + str(k)
        keywords_dict[topic] = list(df_topic_keywords.loc[(df_topic_keywords['Topic']==topic),'Word'])
    
    return keywords_dict

def get_document_topic(output):
    # Column names
    topicnames = ['Topic' + str(i) for i in range(1,output.shape[1]+1)]

    # Index names
    docnames = ['Doc' + str(i) for i in range(1,output.shape[0]+1)]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(output, columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1) + 1
    df_document_topic['dominant_topic'] = dominant_topic
    # print(df_document_topic)
    
    return df_document_topic


# In[160]:


def consistency(model_list):
    keywords_dict_0 = get_keywords_dict(model_list[0])
    consistency_list = []
    for k_0 in keywords_dict_0:
        set_0 = set(keywords_dict_0[k_0])
        
        intersection_list = []
        for i in range(1,len(model_list)):
            keywords_dict_i = get_keywords_dict(model_list[i])
            max_intersection = 0
            for k_i in keywords_dict_i:
                set_k = set(keywords_dict_i[k_i])
                intersection = len(set_0 & set_k)
                if intersection > max_intersection:
                    max_intersection = intersection
            intersection_list.append(max_intersection/len(set_0))
        
        topic_consistency = np.mean(intersection_list)
        consistency_list.append(topic_consistency)
    
    return np.mean(consistency_list)


def redundancy(model):
    keywords_dict = get_keywords_dict(model)
    redundancy_list = []
    for t in keywords_dict:
        set_t = set(keywords_dict[t])
        
        max_intersection = 0
        for k in keywords_dict:
            if k != t:
                set_k = set(keywords_dict[k])
                intersection = len(set_t & set_k)
                if intersection > max_intersection:
                    max_intersection = intersection
        redundancy_list.append(max_intersection/len(set_t))
        
    return np.mean(redundancy_list)

def importance(model, output):
    df_document_topic = get_document_topic(output)
    topic_importance_list = []
    for i in range(1,model.n_components+1):
        D = df_document_topic[df_document_topic['dominant_topic'] == i]
        document_importance_list = []
        for d in D.index:
            proba_list = list(D.loc[D.index[0]])[:-1]
            proba_T = max(proba_list)
            proba_list.remove(proba_T)
            proba_k = max(proba_list)
            document_importance = proba_T - proba_k
            document_importance_list.append(document_importance)
            
        topic_importance = np.mean(document_importance_list)
        topic_importance_list.append(np.mean(topic_importance))
    
    return np.nanmean(topic_importance_list) 


# In[ ]:


df_eval = pd.DataFrame(columns=['K', 'Iteration', 'Consistency', 'Redundancy', 'Importance', 'Perplexity'])

for k in tqdm(range(3,10)): 
    model_list = []
    C = None
    for i in range(1,10): 
        time1 = time.time()
        lda_model = LatentDirichletAllocation(n_components=k, random_state=i)
        lda_output = lda_model.fit_transform(data_vectorized)
        model_list.append(lda_model)
        R = redundancy(lda_model)
        I = importance(lda_model, lda_output)
        P = lda_model.perplexity(data_vectorized)
        df_eval.loc[len(df_eval)] = [k, i, C, R, I, P]
        time2 = time.time()
        timess = time2 - time1
        print('%.3fs'%timess, [k, i, C, R, I, P])
    C = consistency(model_list)
    df_eval_k = df_eval[df_eval['K'] == k]
    avg = list(df_eval_k.mean()) 
    df_eval.loc[len(df_eval)] = [k, None, C, avg[3], avg[4], avg[5]]
    print([k, None, C, avg[3], avg[4], avg[5]])


# ### best model

# In[180]:


# Build LDA Model  
lda_model = LatentDirichletAllocation(n_components=3, random_state=5)
lda_output = lda_model.fit_transform(data_vectorized)


# In[181]:


# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))


# In[183]:


# Topic-document Probalility Matrix
docnames = ['Doc' + str(i) for i in range(1,lda_output.shape[0]+1)]
topicnames = ['Topic' + str(i) for i in range(1,lda_output.shape[1]+1)]
df_document_topic = pd.DataFrame(lda_output, index=docnames, columns=topicnames)


# In[185]:


# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1) + 1
df_document_topic['dominant_topic'] = dominant_topic


# In[187]:


df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic', 'Num Documents']


# In[189]:


# Topic-word Weight Matrix
topicnames = ["Topic" + str(i) for i in range(1,lda_model.n_components+1)]
featurenames = vectorizer.get_feature_names()
df_topic_word = pd.DataFrame(lda_model.components_, index=topicnames, columns=featurenames)


# In[190]:


# Topic-word Probalility Matrix
featurenames = vectorizer.get_feature_names()
topicnames = ["Topic" + str(i) for i in range(1,lda_model.n_components+1)]
w = lda_model.components_
proba = np.apply_along_axis(lambda x: x/x.sum(),1,w)
df_topic_word_proba = pd.DataFrame(proba.T, index=featurenames, columns=topicnames)


# In[191]:


# Keywords
df_topic_keywords = pd.DataFrame(columns=['Topic', 'No', 'Word', 'Proba', 'Cumsum'])
for k in range(1,lda_model.n_components+1):
    topic = 'Topic' + str(k)
    n = 0
    cumsum = 0
    while cumsum < 0.80:
        word_idx = (-df_topic_word_proba[topic]).argsort()[n]
        word = df_topic_word_proba.index[word_idx]
        proba = df_topic_word_proba.loc[word,topic]
        if proba < 0.005:
            break
        cumsum +=  proba
        df_topic_keywords.loc[len(df_topic_keywords)] = [topic, n+1, word, proba, cumsum]
        n += 1


# In[193]:


df_topic_keywords['Text'] = [df_topic_keywords.loc[i,'Word'] + str(round(df_topic_keywords.loc[i,'Proba'],3)) for i in df_topic_keywords.index]
df_agg = df_topic_keywords.groupby(['Topic']).agg({'Text':' '.join}).reset_index()


# In[194]:


for i in df_agg.index:
    topic = df_agg.loc[i,'Topic']
    text = df_agg.loc[i,'Text']
    n = int(df_topic_distribution.loc[df_topic_distribution['Topic']==i+1,'Num Documents'])
    print(topic, n, ': ', text, '\n')



