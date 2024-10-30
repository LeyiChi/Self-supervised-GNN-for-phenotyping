#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from tqdm import tqdm_notebook as tqdm


# ### data preprocess

# In[3]:


# visit label
df = pd.read_csv("visit_label_tm.csv")
df['VISIT_START_DATE'] = pd.to_datetime(df['VISIT_START_DATE'],)
df['VISIT_END_DATE'] = pd.to_datetime(df['VISIT_END_DATE'])


# In[4]:


df_fst_visit = df.groupby(['PERSON_ID']).agg({'VISIT_START_DATE':'min'}).reset_index()
df_fst_visit.columns = ['PERSON_ID', 'FST_VISIT_DATE']


# In[5]:


df = df.merge(df_fst_visit, how='left')
df.head()


# In[6]:


# time interval
df['DIFF_DAYS'] = df['VISIT_START_DATE'] - df['FST_VISIT_DATE']
df['DIFF_DAYS'] = [x.days for x in df['DIFF_DAYS']]
df['YRS'] = [math.ceil(x/365.25) for x in df['DIFF_DAYS']]
df.loc[df['YRS']==0, 'YRS'] = 1


# ### disease status transformation

# In[7]:


df_s1 = df[df['SUBTYPE'] == 1]


# In[8]:


df_yrs_status = df_s1.groupby(['PERSON_ID', 'YRS'])['TM_LABEL'].apply(list).reset_index()

get_last_from_list = lambda x: x[-1]
df_yrs_status['STATUS'] = df_yrs_status['TM_LABEL'].apply(get_last_from_list)


# In[9]:


df_yrs_status = df_yrs_status[['PERSON_ID', 'YRS', 'STATUS']]
df_yrs_status = df_yrs_status[df_yrs_status['YRS'] <= 10]


# In[10]:


unique_person_ids = df_yrs_status['PERSON_ID'].unique()
df_all_years = pd.DataFrame(list(itertools.product(unique_person_ids, list(range(1, 11)))), columns=['PERSON_ID', 'YRS'])

df_all_yrs_status = df_all_years.merge(df_yrs_status, on=['PERSON_ID', 'YRS'], how='left')

df_all_yrs_status['STATUS'] = df_all_yrs_status.groupby('PERSON_ID')['STATUS'].ffill()


# In[11]:


df_all_yrs_status['PREV_STATUS'] = df_all_yrs_status.groupby('PERSON_ID')['STATUS'].shift(1)
df_all_yrs_status['TRANSITION'] = df_all_yrs_status.apply(lambda row: str(int(row['PREV_STATUS'])) + '-' + str(int(row['STATUS'])) if not pd.isnull(row['PREV_STATUS']) else None, axis=1)


# In[13]:


df_all_yrs_status_cnt = df_all_yrs_status.groupby(['YRS', 'TRANSITION']).size().reset_index(name='CNT')


# In[14]:


# 计算每年发生状态转变的次数
year_counts = df_all_yrs_status.groupby(['YRS'])['STATUS'].count().reset_index(name='YEAR_CNT')


# In[15]:


# 计算每年发生状态转变的概率
status_transitions_count = df_all_yrs_status_cnt.merge(year_counts)
# status_transitions_count['PROBABILITY'] = status_transitions_count['CNT'] / status_transitions_count['YEAR_CNT']
status_transitions_count['PROBABILITY'] = status_transitions_count['CNT'] / 3168


# In[16]:


status_transitions_count['FROM'] = [int(x[0]) for x in status_transitions_count['TRANSITION']]
status_transitions_count['TO'] = [int(x[2]) for x in status_transitions_count['TRANSITION']]


# In[17]:


temp = status_transitions_count[status_transitions_count['FROM']!=status_transitions_count['TO']]
max(temp['PROBABILITY'])


# In[18]:


def heatmap_k(status_transitions_count, k):
    # 检查 k 是否在 2 到 10 的范围内
    if k < 2 or k > 10:
        raise ValueError("k should be in the range of 2 to 10.")
        
    # 第k年数据
    status_transitions_count_k = status_transitions_count[status_transitions_count['YRS'] == k]
    status_transitions_count_k = status_transitions_count_k[['FROM','TO','PROBABILITY']]

    # 创建空的转移矩阵
    index = pd.Index(range(1,4), name='FROM')
    columns = pd.Index(range(1,4), name='TO')
    transition_matrix = pd.DataFrame(index=index, columns=columns)

    # 将转移概率填入转移矩阵
    for _, row in status_transitions_count_k.iterrows():
        from_value = row['FROM']
        to_value = row['TO']
        probability = row['PROBABILITY']
        transition_matrix.at[from_value, to_value] = probability

    # 将空值填充为0
    transition_matrix = transition_matrix.fillna(0)
    transition_matrix = transition_matrix + np.diag(1 - transition_matrix.sum(axis=0))
  
    return transition_matrix


# In[19]:


sns.set(style='whitegrid', font_scale=1)
fig, ax = plt.subplots(4, 2, constrained_layout=True, figsize=(8, 12), dpi=300)
# titles = ['A. Visit status from 1st year to 2nd year', 'B. Visit status from 2nd year to 3rd year',
#           'C. Visit status from 3rd year to 4th year', 'D. Visit status from 4th year to 5th year',
#           'E. Visit status from 5th year to 6th year', 'F. Visit status from 6th year to 7th year',
#           'G. Visit status from 7th year to 8th year', 'H. Visit status from 8th year to 9th year']
titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

for i in range(2,10):
    transition_matrix = heatmap_k(status_transitions_count, i)
    sns_plot = sns.heatmap(transition_matrix, cmap="Blues", ax=ax[i//2-1, i%2],
                          vmin = 0, vmax = 0.035)
    sns_plot.set_title(titles[i - 2], fontweight='bold', loc='left')
    if i==2:
        sns_plot.set_xlabel("Disease status at 2nd year", fontweight='bold')
        sns_plot.set_ylabel("Disease status at 1st year", fontweight='bold')
    elif i==3:
        sns_plot.set_xlabel("Disease status at 3rd year", fontweight='bold')
        sns_plot.set_ylabel("Disease status at 2nd year", fontweight='bold')
    elif i==4:
        sns_plot.set_xlabel("Disease status at 4th year", fontweight='bold')
        sns_plot.set_ylabel("Disease status at 3rd year", fontweight='bold')
    else:
        sns_plot.set_xlabel("Disease status at " + str(i) + "th year", fontweight='bold')
        sns_plot.set_ylabel("Disease status at " + str(i-1) + "th year", fontweight='bold')
        


# In[20]:


fig.savefig('./3.4. visit_transform_tm_1.tiff')
fig.savefig('./3.4. visit_transform_tm_1.png')


