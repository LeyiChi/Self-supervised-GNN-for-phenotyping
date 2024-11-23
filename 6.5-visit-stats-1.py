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
import matplotlib.gridspec as gridspec
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


# ### progression

# In[7]:


lable_color = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E'}
lable_annotation = {1:'Topic 1',
                    2:'Topic 2',
                    3:'Topic 3',
                    }  


# ### subphenotype 1

# In[8]:


df_s1 = df[df['SUBTYPE'] == 1]


# In[9]:


data_time = []
n_gap = 9
bins = list(range(1,n_gap+1))

for c in range(1, 4): # grouped bar plot version
    target_df = df_s1[df_s1['TM_LABEL'] == c]
    n_c = len(target_df)
    distribution = plt.hist(target_df['YRS'].values, bins=bins)
    data_time.append(distribution[0])   
    
data_time = np.array(data_time)
data_time = data_time / data_time.sum(axis=0)
total_dist = plt.hist(df_s1['YRS'].values, bins=bins)[0]
total_dist = total_dist / len(df_s1)
max(total_dist)


# In[10]:


fig = plt.figure(figsize=[8, 6])
fig.subplots_adjust(left=0.08, right=0.8, top=0.95, bottom=0.25, hspace=0.1)
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 0.5])

ax1 = fig.add_subplot(spec[0, 0])
ax1.grid(False)
ax1.bar(range(1, n_gap), data_time[0], color=lable_color[1], label=lable_annotation[1], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[1], bottom=data_time[0], color=lable_color[2], label=lable_annotation[2], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[2], bottom=data_time[0]+data_time[1], color=lable_color[3], label=lable_annotation[3], width=1, edgecolor='white', linewidth=1)
ax1.set_xticks([])
ax1.set_ylim(0, 1)
ax1.set_yticks([0, .25, .5, .75, 1])
ax1.set_yticklabels([0, 25, 50, 75, 100])
ax1.legend(bbox_to_anchor = (1.275, 0.36), loc='upper right', fontsize=11)
ax1.set_ylabel('Proportion within year (%)', fontsize=12)

ax2 = fig.add_subplot(spec[1, 0])
ax2.grid(False)
ax2.bar(range(1, n_gap), total_dist, color='gray', width=1, edgecolor='white', linewidth=1)
ax2.set_ylim(0, .3)
ax2.set_xticks(range(1, n_gap))
ax2.set_xticklabels(['%d' % i for i in range(1, n_gap)])
ax2.set_yticks([0, .25, .5, .75, 1])
ax2.set_yticklabels([0, 25, 50, 75, 100])
ax2.set_ylabel('Proportion (%)', fontsize=12)
ax2.set_xlabel('Time to first dialysis (years)', fontsize=12)


# In[12]:


fig.savefig('./3.6. proportion_within_year_1.tiff')
fig.savefig('./3.6. proportion_within_year_1.png')


# ### subphenotype 2

# In[13]:


df_s2 = df[df['SUBTYPE'] == 2]


# In[14]:


data_time_s2 = []
n_gap = 9
bins = list(range(1,n_gap+1))

for c in range(1, 4): # grouped bar plot version
    target_df = df_s2[df_s2['TM_LABEL'] == c]
    n_c = len(target_df)
    distribution = plt.hist(target_df['YRS'].values, bins=bins)
    data_time_s2.append(distribution[0])   
    
data_time_s2 = np.array(data_time_s2)
data_time_s2 = data_time_s2 / data_time_s2.sum(axis=0)
total_dist_s2 = plt.hist(df_s2['YRS'].values, bins=bins)[0]
total_dist_s2 = total_dist_s2 / len(df_s2)
max(total_dist_s2)


# In[15]:


fig = plt.figure(figsize=[8, 6])
fig.subplots_adjust(left=0.08, right=0.8, top=0.95, bottom=0.25, hspace=0.1)
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 0.5])

ax1 = fig.add_subplot(spec[0, 0])
ax1.grid(False)
ax1.bar(range(1, n_gap), data_time_s2[0], color=lable_color[1], label=lable_annotation[1], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time_s2[1], bottom=data_time_s2[0], color=lable_color[2], label=lable_annotation[2], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time_s2[2], bottom=data_time_s2[0]+data_time_s2[1], color=lable_color[3], label=lable_annotation[3], width=1, edgecolor='white', linewidth=1)
ax1.set_xticks([])
ax1.set_ylim(0, 1)
ax1.set_yticks([0, .25, .5, .75, 1])
ax1.set_yticklabels([0, 25, 50, 75, 100])
ax1.legend(bbox_to_anchor = (1.275, 0.36), loc='upper right', fontsize=11)
ax1.set_ylabel('Proportion within year (%)', fontsize=12)

ax2 = fig.add_subplot(spec[1, 0])
ax2.grid(False)
ax2.bar(range(1, n_gap), total_dist_s2, color='gray', width=1, edgecolor='white', linewidth=1)
ax2.set_ylim(0, .3)
ax2.set_xticks(range(1, n_gap))
ax2.set_xticklabels(['%d' % i for i in range(1, n_gap)])
ax2.set_yticks([0, .25, .5, .75, 1])
ax2.set_yticklabels([0, 25, 50, 75, 100])
ax2.set_ylabel('Proportion (%)', fontsize=12)
ax2.set_xlabel('Time to first dialysis (years)', fontsize=12)


# In[18]:


fig.savefig('./3.7. proportion_within_year_2.tiff')
fig.savefig('./3.7. proportion_within_year_2.png')


# ### figure merge

# In[19]:


fig = plt.figure(figsize=[16, 8])
main_spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

# 子图1
spec = gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=2, subplot_spec=main_spec[0, 0], height_ratios=[1, 0.5])
ax1 = fig.add_subplot(spec[0, 0])
ax1.text(0, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top') 
ax1.grid(False)
ax1.bar(range(1, n_gap), data_time[0], color=lable_color[1], label=lable_annotation[1], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[1], bottom=data_time[0], color=lable_color[2], label=lable_annotation[2], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[2], bottom=data_time[0]+data_time[1], color=lable_color[3], label=lable_annotation[3], width=1, edgecolor='white', linewidth=1)
ax1.set_xticks([])
ax1.set_ylim(0, 1)
ax1.set_yticks([0, .25, .5, .75, 1])
ax1.set_yticklabels([0, 25, 50, 75, 100], fontweight='bold')
# ax1.legend(bbox_to_anchor = (1.275, 0.36), loc='upper right', fontsize=11)
ax1.set_ylabel('Proportion within year (%)', fontsize=12, fontweight='bold')

ax2 = fig.add_subplot(spec[1, 0])
ax2.grid(False)
ax2.bar(range(1, n_gap), total_dist, color='gray', width=1, edgecolor='white', linewidth=1)
ax2.set_ylim(0, .3)
ax2.set_xticks(range(1, n_gap))
ax2.set_xticklabels(['%d' % i for i in range(1, n_gap)], fontweight='bold')
ax2.set_yticks([0, .25, .5, .75, 1])
ax2.set_yticklabels([0, 25, 50, 75, 100], fontweight='bold')
ax2.set_ylabel('Proportion (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time to first dialysis (years)', fontsize=12, fontweight='bold')


# 子图2
spec = gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=2, subplot_spec=main_spec[0, 1], height_ratios=[1, 0.5])
ax1 = fig.add_subplot(spec[0, 0])
ax1.text(0, 1.1, 'B', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')  
ax1.grid(False)
ax1.bar(range(1, n_gap), data_time_s2[0], color=lable_color[1], label=lable_annotation[1], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time_s2[1], bottom=data_time_s2[0], color=lable_color[2], label=lable_annotation[2], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time_s2[2], bottom=data_time_s2[0]+data_time_s2[1], color=lable_color[3], label=lable_annotation[3], width=1, edgecolor='white', linewidth=1)
ax1.set_xticks([])
ax1.set_ylim(0, 1)
ax1.set_yticks([0, .25, .5, .75, 1])
ax1.set_yticklabels([0, 25, 50, 75, 100], fontweight='bold')
ax1.legend(bbox_to_anchor = (1.275, 0.36), loc='upper right', fontsize=11)
ax1.set_ylabel('Proportion within year (%)', fontsize=12, fontweight='bold')

ax2 = fig.add_subplot(spec[1, 0])
ax2.grid(False)
ax2.bar(range(1, n_gap), total_dist_s2, color='gray', width=1, edgecolor='white', linewidth=1)
ax2.set_ylim(0, .3)
ax2.set_xticks(range(1, n_gap))
ax2.set_xticklabels(['%d' % i for i in range(1, n_gap)], fontweight='bold')
ax2.set_yticks([0, .25, .5, .75, 1])
ax2.set_yticklabels([0, 25, 50, 75, 100], fontweight='bold')
ax2.set_ylabel('Proportion (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time to first dialysis (years)', fontsize=12, fontweight='bold')

plt.show()


# In[20]:


fig.savefig('./3.8. proportion_within_year.tiff')
fig.savefig('./3.8. proportion_within_year.png')


# In[ ]:




