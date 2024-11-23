#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.notebook import tqdm


# ### Load data

df = pd.read_csv("data/data_logs_0919.csv")
df.sort_values(['PERSON_ID', 'DATE'], inplace = True)
df = df.reset_index(drop=True)


print('patient: ', len(set(df['PERSON_ID'])))
print('visits: ', len(set(df['VISIT_OCCURRENCE_ID'])))
print('events: ', len(set(df['EVENT'])))


# ### data screening

# data formation: {visit: events}
vidEventMap = defaultdict(list)
for i in tqdm(range(len(df))):
    pid = df.loc[i,'PERSON_ID']
    vid = df.loc[i,'VISIT_OCCURRENCE_ID']
    event = df.loc[i,'EVENT']
    vidEventMap[vid].append(event)


# remove visits whose events are no more than 2
del_list = []
for vid in vidEventMap:
    if len(vidEventMap[vid]) <= 2:
        del_list.append(vid)
for vid in del_list:
    del vidEventMap[vid]



df_2 = df[df['VISIT_OCCURRENCE_ID'].isin(list(vidEventMap.keys()))]
print('patient: ', len(set(df_2['PERSON_ID'])))
print('visit: ', len(set(df_2['VISIT_OCCURRENCE_ID'])))
print('event: ', len(set(df_2['EVENT'])))



# data formation: {patient: visits}
df_2 = df_2.reset_index(drop=True)
pidVidMap = defaultdict(list)
for i in tqdm(range(len(df_2))):
    pid = df_2.loc[i,'PERSON_ID']
    vid = df_2.loc[i,'VISIT_OCCURRENCE_ID']
    event = df_2.loc[i,'EVENT']
    pidVidMap[pid].append(vid)

for key in pidVidMap:
    pidVidMap[key] = list(set(pidVidMap[key]))



# remove patients whose visits are no more than 2
del_list_2 = []
for pid in pidVidMap:
    if len(pidVidMap[pid]) <= 2:
        del_list_2.append(pid)
for pid in del_list_2:
    del pidVidMap[pid]



df_3 = df_2[df_2['PERSON_ID'].isin(list(pidVidMap.keys()))]



