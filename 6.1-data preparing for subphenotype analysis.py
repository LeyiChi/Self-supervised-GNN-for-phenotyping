#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from tableone import TableOne
import pickle

# ### events

# In[3]:


df_tr = pd.read_csv("tr_data_logs.csv")
df_tr.head()


# In[4]:


print('#events：', len(df_tr))
print('#ptients：', df_tr['PERSON_ID'].nunique())
print('#visits：', df_tr['VISIT_OCCURRENCE_ID'].nunique())
print('#entities：', df_tr['EVENT'].nunique())


# In[5]:


tr_pid_list = df_tr['PERSON_ID'].unique()

df_tr[['EVENT','TYPE']].drop_duplicates()['TYPE'].value_counts()


# In[11]:


df_pivot_tr = df_tr.groupby(['EVENT', 'TYPE']).agg({'VISIT_OCCURRENCE_ID':'count'}).reset_index()
df_pivot_tr = df_pivot_tr.rename(columns = {'VISIT_OCCURRENCE_ID':'CNT'})



# In[13]:


df_temp = df_pivot_tr[df_pivot_tr['TYPE']=='diagnosis']
df_temp = df_temp.sort_values(by=['CNT'], ascending=False)
print(list(df_temp['EVENT']))


# In[14]:


df_temp = df_pivot_tr[df_pivot_tr['TYPE']=='labtests']
df_temp = df_temp.sort_values(by=['CNT'], ascending=False)
print(list(df_temp['EVENT']))


# In[15]:


df_temp = df_pivot_tr[df_pivot_tr['TYPE']=='examination']
df_temp = df_temp.sort_values(by=['CNT'], ascending=False)
print(list(df_temp['EVENT']))


# In[16]:


df_temp = df_pivot_tr[df_pivot_tr['TYPE']=='surgery']
df_temp = df_temp.sort_values(by=['CNT'], ascending=False)
print(list(df_temp['EVENT']))


# In[17]:


df_temp = df_pivot_tr[df_pivot_tr['TYPE']=='medication']
df_temp = df_temp.sort_values(by=['CNT'], ascending=False)
print(list(df_temp['EVENT']))


# #### Number of events per visit

# In[18]:


df_event_stat_tr = df_tr.groupby(['PERSON_ID', 'VISIT_OCCURRENCE_ID']).agg({'EVENT':'count'}).reset_index()
df_event_stat_tr = df_event_stat_tr.rename(columns = {'EVENT':'CNT'})
df_event_stat_tr.head()


# #### The median time for the first occurrence of ALTRs 

# In[26]:


df_m_tr = df_tr[df_tr['TYPE']=='labtests']
df_m_tr.head()


# In[27]:


df_fst_abn_tr = df_m_tr.groupby(['PERSON_ID']).agg({'DATE':'min'}).reset_index()
df_fst_abn_tr.head()


# In[28]:


df_fst_abn_tr.columns = ['PERSON_ID', 'dateOfALTRs']


# In[30]:


df_fst_abn_tr.to_csv('tr_fst_abn.csv', index=False, encoding='utf-8-sig')


# ### visits

# In[29]:


df_visit_tr = df_tr.groupby(['PERSON_ID','VISIT_OCCURRENCE_ID']).agg({'EVENT':' '.join}).reset_index()
df_visit_tr.columns = df_visit_tr.columns.get_level_values(0)
df_visit_tr.head()


# In[30]:


df_visit_stat_tr = df_visit_tr.groupby(['PERSON_ID']).agg({'VISIT_OCCURRENCE_ID':'count'}).reset_index()
df_visit_stat_tr = df_visit_stat_tr.rename(columns = {'VISIT_OCCURRENCE_ID':'CNT'})
df_visit_stat_tr.head()


# ### Demographics

# In[36]:


df_basic_tr = pd.read_csv("../data/LXY_CKD_BASIC_0810.csv")
df_basic_tr.head()


# In[41]:


df_cohort_tr = pd.read_csv("../data/LXY_CKD_COHORT_WITH_ENDPOINT.csv")
df_cohort_tr.head()


# In[43]:


df_cohort_tr = df_cohort_tr[['PERSON_ID', 'time', 'status']]
df_cohort_tr.head()


# In[44]:


df_patient_tr = df_basic_tr.merge(df_cohort_tr, on='PERSON_ID')
df_patient_tr = df_patient_tr[df_patient_tr['PERSON_ID'].isin(tr_pid_list)]
df_patient_tr.head()


# ### data preparing for subphenotype analysis

# In[53]:


df_cluster_tr = pd.read_csv("pid_cluster_tr.csv")
df_cluster_tr.head()


# In[55]:


df_cluster_tr = df_cluster_tr.rename(columns = {'PID':'PERSON_ID'})


# In[56]:


df_cluster_tr = df_patient_tr.merge(df_cluster_tr, how='left', on='PERSON_ID')
df_cluster_tr = df_cluster_tr.merge(df_fst_abn_tr, how='left', on='PERSON_ID')
df_cluster_tr.head()


# In[57]:


df_cluster_tr.to_csv('tr_subtype_data.csv', index=False, encoding='utf-8-sig')


# ### validation cohort

# In[59]:


df_cluster_test = pd.read_csv("pid_cluster_test.csv")
df_cluster_test.head()


# In[61]:


df_cluster_test = df_cluster_test[df_cluster_test['CLUSTER']!=3]
df_cluster_test.head()


# In[62]:


df_cluster_test = df_cluster_test.rename(columns = {'PID':'PERSON_ID'})


# In[63]:


test_pid_list = df_cluster_test['PERSON_ID'].unique()


# ### event logs

# In[64]:


df_test = pd.read_csv("test_data_logs.csv")
df_test = df_test[df_test['PERSON_ID'].isin(test_pid_list)]
df_test.head()


# In[65]:


print('#events：', len(df_test))
print('#patients：', df_test['PERSON_ID'].nunique())
print('#visits：', df_test['VISIT_OCCURRENCE_ID'].nunique())
print('#entities：', df_test['EVENT'].nunique())


# In[66]:


test_pid_list = df_test['PERSON_ID'].unique()


# In[70]:


df_pivot_test = df_test.groupby(['EVENT', 'TYPE']).agg({'VISIT_OCCURRENCE_ID':'count'}).reset_index()
df_pivot_test = df_pivot_test.rename(columns = {'VISIT_OCCURRENCE_ID':'CNT'})
df_pivot_test.head()


# In[71]:


df_temp = df_pivot_test[df_pivot_test['TYPE']=='diagnosis']
df_temp = df_temp.sort_values(by=['CNT'], ascending=False)
print(list(df_temp['EVENT']))


# In[72]:


df_temp = df_pivot_test[df_pivot_test['TYPE']=='labtests']
df_temp = df_temp.sort_values(by=['CNT'], ascending=False)
print(list(df_temp['EVENT']))


# In[73]:


df_temp = df_pivot_test[df_pivot_test['TYPE']=='medication']
df_temp = df_temp.sort_values(by=['CNT'], ascending=False)
print(list(df_temp['EVENT']))


# #### Number of events per visit

# In[74]:


df_event_stat_test = df_test.groupby(['PERSON_ID', 'VISIT_OCCURRENCE_ID']).agg({'EVENT':'count'}).reset_index()
df_event_stat_test = df_event_stat_test.rename(columns = {'EVENT':'CNT'})


# #### The median time for the first occurrence of ALTRs

# In[80]:


df_m_test = df_test[df_test['TYPE']=='labtests']


# In[81]:


df_fst_abn_test = df_m_test.groupby(['PERSON_ID']).agg({'DATE':'min'}).reset_index()


# In[82]:


df_fst_abn_test.columns = ['PERSON_ID', 'dataOfALTRs']


# In[83]:


df_fst_abn_test.to_csv('test_fst_abn.csv', index=False, encoding='utf-8-sig')


# ### visits

# In[84]:


df_visit_test = df_test.groupby(['PERSON_ID','VISIT_OCCURRENCE_ID']).agg({'EVENT':' '.join}).reset_index()
df_visit_test.columns = df_visit_test.columns.get_level_values(0)


# In[85]:


df_visit_stat_test = df_visit_test.groupby(['PERSON_ID']).agg({'VISIT_OCCURRENCE_ID':'count'}).reset_index()
df_visit_stat_test = df_visit_stat_test.rename(columns = {'VISIT_OCCURRENCE_ID':'CNT'})


# ### Demographics

# In[92]:


df_basic_test = pd.read_csv("data/LXY_CKD_BASIC.csv")


# In[93]:


df_basic_test = df_basic_test[df_basic_test['PERSON_ID'].isin(test_pid_list)]


# In[94]:


df_fst_visit_test = df_test.groupby(['PERSON_ID']).agg({'DATE':'min'}).reset_index()


# In[95]:


df_patient_test = df_basic_test.merge(df_fst_visit_test, how = 'left', on = 'PERSON_ID')


# In[96]:


df_patient_test['DOB'] = pd.to_datetime(df_patient_test['DOB'] )
df_patient_test['DATE'] = pd.to_datetime(df_patient_test['DATE'] )
df_patient_test['DIFF'] = df_patient_test['DATE'] - df_patient_test['DOB'] 
df_patient_test['AGE'] = [x.days / 365.25 for x in df_patient_test['DIFF']]


# ### data preparing for subphenotype analysis

# In[102]:


df_cluster_test = df_patient_test.merge(df_cluster_test, how='left', on='PERSON_ID')
df_cluster_test = df_cluster_test.merge(df_fst_abn_test, how='left', on='PERSON_ID')


# In[104]:


df_cluster_test.to_csv('test_subtype_data.csv', index=False, encoding='utf-8-sig')


# ### Visualization

# In[107]:


import seaborn as sns
import matplotlib.pyplot as plt



# In[111]:


sns.set(style='whitegrid', font_scale=1)
fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(12,6), dpi=300)

# development cohort
sns_plot = sns.distplot(x=df_visit_stat_tr['CNT'], ax=ax[0,0])
sns_plot.set_title('a', fontweight='bold', loc='left')
sns_plot.set_xlabel("Number of visits per patient", fontweight='bold')
sns_plot.set_ylabel("Density", fontweight='bold')
ax[0,0].axvline(np.median(df_visit_stat_tr['CNT']), c='r', ls='--')

sns_plot = sns.distplot(x=df_event_stat_tr['CNT'], ax=ax[0,1])
sns_plot.set_title('b', fontweight='bold', loc='left')
sns_plot.set_xlabel("Number of events per visit", fontweight='bold')
sns_plot.set_ylabel("Density", fontweight='bold')
ax[0,1].axvline(np.median(df_event_stat_tr['CNT']), c='r', ls='--')

sns_plot = sns.distplot(x=df_pivot_tr['CNT'], bins=50, ax=ax[0,2])
sns_plot.set_title('c', fontweight='bold', loc='left')
sns_plot.set_xlabel("Number of event occurrences", fontweight='bold')
sns_plot.set_ylabel("Density", fontweight='bold')
ax[0,2].axvline(np.median(df_pivot_tr['CNT']), c='r', ls='--')

# validation cohort
sns_plot = sns.distplot(x=df_visit_stat_test['CNT'], ax=ax[1,0])
sns_plot.set_title('d', fontweight='bold', loc='left')
sns_plot.set_xlabel("Number of visits per patient", fontweight='bold')
sns_plot.set_ylabel("Density", fontweight='bold')
ax[1,0].axvline(np.median(df_visit_stat_test['CNT']), c='r', ls='--')

sns_plot = sns.distplot(x=df_event_stat_test['CNT'], ax=ax[1,1])
sns_plot.set_title('e', fontweight='bold', loc='left')
sns_plot.set_xlabel("Number of events per visit", fontweight='bold')
sns_plot.set_ylabel("Density", fontweight='bold')
ax[1,1].axvline(np.median(df_event_stat_test['CNT']), c='r', ls='--')

sns_plot = sns.distplot(x=df_pivot_test['CNT'], bins=50, ax=ax[1,2])
sns_plot.set_title('f', fontweight='bold', loc='left')
sns_plot.set_xlabel("Number of event occurrences", fontweight='bold')
sns_plot.set_ylabel("Density", fontweight='bold')
ax[1,2].axvline(np.median(df_pivot_test['CNT']), c='r', ls='--')


# In[112]:


fig = sns_plot.get_figure()
fig.savefig('./data_statistics.tiff')



