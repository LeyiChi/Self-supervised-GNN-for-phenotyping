#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from tableone import TableOne
from scipy.stats import mannwhitneyu


# In[3]:


df_tr = pd.read_csv("tr_subtype_combined_data_R.csv", encoding='gb18030')


# In[4]:


df_tr['COHORT'] = 'Development cohort'


# In[5]:


df_test = pd.read_csv("test_subtype_combined_data_R.csv", encoding='gb18030')


# In[6]:


df_test['COHORT'] = 'Validation cohort'


# ### statistics of the cohorts

# In[7]:


df = pd.concat([df_tr, df_test])
df = df.reset_index(drop=True)


# In[9]:


columns = ['性别', '年龄', '身高', '体重', 
           '吸烟', '饮酒', '文化程度','对疾病认识', '药物依赖.药瘾.吸毒',
           '观察窗', '结局事件','首次指标异常时间',
           '高血压','贫血','糖尿病','甲状旁腺功能亢进','骨矿物质代谢异常','心力衰竭',
           '冠心病','胃肠道疾病','睡眠障碍','高脂血症','水电解质平衡失调',
           '血常规异常','肝功能异常','肾功能异常','脂异常',
           '电解质异常','血糖异常','血气分析异常','甲状腺功能异常',
           '超敏c反应蛋白异常','血栓指标异常','凝血功能异常',
           '心衰指标异常','肿瘤标志物异常','贫血指标异常']


# In[10]:


continuous = ['年龄','首次指标异常时间', '身高', '体重', '观察窗']
categorical = list(set(columns) - set(continuous))


# In[11]:


for var in categorical:
    df[var] = pd.Categorical(df[var])
    df[var] = df[var].cat.codes 


# In[12]:


table = TableOne(df, columns=columns, categorical=categorical, groupby="COHORT", pval=True,
                 nonnormal=continuous)


# In[13]:


table.to_excel('baseline characteristics.xlsx', encoding='utf-8-sig')


# ### statistics of subphenotype distribution

# In[14]:


df_dev = df[df['COHORT']=='Development cohort']
table = TableOne(df_dev, columns=columns, categorical=categorical, groupby="CLUSTER", pval=True,
                 nonnormal=continuous)
table.to_excel('baseline characteristics on the development cohort.xlsx', encoding='utf-8-sig')


# In[15]:


df_val = df[df['COHORT']=='Validation cohort']
table = TableOne(df_val, columns=columns, categorical=categorical, groupby="CLUSTER", pval=True,
                 nonnormal=continuous)
table.to_excel('baseline characteristics on the validation cohort.xlsx', encoding='utf-8-sig')


# ## 事件统计

# In[16]:


diag_list = ['慢性肾脏病', '血液透析', '高血压',
       '骨矿物质代谢异常', '贫血', '糖尿病', '动静脉瘘', '肾炎', '甲状旁腺功能亢进', '心力衰竭', '胃肠道疾病',
       '冠心病', '糖尿病肾病', '多囊肾', '肺部感染', '周围血管疾病', '睡眠障碍', '上呼吸道感染', '乙型病毒性肝炎',
       '高脂血症', '骨髓瘤', '感染', '水电解质平衡失调', '脑梗死', '骨折', '肝囊肿', '肾功能检查的异常结果', '便秘',
       '胆囊结石', '肾病综合征', '前列腺增生', '甲状腺结节', '痛风', '阿尔茨海默病', '房颤', '胸腔积液', '发热','神经病变', '甲状腺功能减退', '高尿酸血症', '慢性肾脏病性心肌病', '肾癌', '支气管病变', '营养不良',
       '肾移植状态']

measurement_list = ['血常规异常', '肝功能异常', '肾功能异常', '脂异常', '电解质异常', '血糖异常', '血气分析异常',
       '甲状腺功能异常', '超敏c反应蛋白异常', '血栓指标异常', '凝血功能异常', '心衰指标异常', '肿瘤标志物异常',
       '贫血指标异常', '乙肝病毒指标异常', '心脑血管指标异常', '心肌损伤指标异常', '抗生素敏感性指标异常', '感染指标异常',
       '乳酸异常', '脱氢酶异常', '血沉偏高', '甘脯二肽氨基肽酶偏低']

drug_list = ['左卡尼汀', '钙离子通道阻滞剂', '钙补充剂',
       '维生素', '骨化三醇', 'β受体阻滞剂', 'ARB', '抗血小板药物', '质子泵抑制剂', '复方α酮酸', '氨基酸',
       '尿激酶', '利多卡因', '利尿剂', '硝酸酯类药物', '司维拉姆', '人血白蛋白', '多磺酸粘多糖霜', '阿司匹林','碳酸镧', '多巴胺', 'ACEI', '磷酸钠盐', 'α受体阻滞剂', '米多君', '丙泊酚', '吗替麦考酚酯',
       '羟苯磺酸钙', '生长抑素', '鲑降钙素', '多巴酚丁胺', '硫酸鱼精蛋白', '甲氧氯普胺', '血凝酶', '腹膜透析液',
       '高渗枸橼酸盐嘌呤', '左甲状腺素钠', '塞来昔布', '其他抗高血压', '降血糖药', '调血脂药', '抗凝血药',
       '电解质补充药物', '抗生素', '非甾体抗炎药', '肠胃药', '精神类药物', '抗贫血药', '抗心绞痛药', '止咳平喘药',
       '营养剂', '泻药', '止泻药', '抗感冒药', '组胺受体阻断药', '抗心衰药', '抗心律失常药', '镇痛药', '抗病毒药',
       '碘造影剂', '拟肾上腺素药']
all_event_list = diag_list + measurement_list + drug_list


# In[17]:


event_type_list = ['diagnosis', 'labtests', 'medication']
event_list_list = [diag_list, measurement_list, drug_list]


# ### event distribution and p-value

# In[18]:


table = TableOne(df_dev, columns=all_event_list, categorical=all_event_list, groupby="CLUSTER", pval=True)
table.to_excel('subphenotype_stats_dev_ALL.xlsx', encoding='utf-8-sig')


# In[19]:


df_tr_dist_pval = pd.read_excel("subphenotype_stats_dev_ALL.xlsx")
df_tr_dist_pval = df_tr_dist_pval[~df_tr_dist_pval['Unnamed: 0'].isna()]
df_tr_dist_pval = df_tr_dist_pval.iloc[1:, [0, -1]]
df_tr_dist_pval.columns = ['event','P-Value']
df_tr_dist_pval['event'] = df_tr_dist_pval['event'].replace(', n \(%\)', '', regex=True)


# In[20]:


test_list = set(df_test.columns)
diff = [x for x in all_event_list if x not in test_list]
print(diff)


# In[21]:


event_list_test = set(all_event_list) - set(diff)
event_list_test = list(event_list_test)


# In[22]:


table = TableOne(df_val, columns=event_list_test, categorical=event_list_test, groupby="CLUSTER", pval=True)
table.to_excel('subphenotype_stats_val_ALL.xlsx', encoding='utf-8-sig')


# In[23]:


df_test_dist_pval = pd.read_excel("subphenotype_stats_val_ALL.xlsx")
df_test_dist_pval = df_test_dist_pval[~df_test_dist_pval['Unnamed: 0'].isna()]
df_test_dist_pval = df_test_dist_pval.iloc[1:, [0, -1]]
df_test_dist_pval.columns = ['event','P-Value']
df_test_dist_pval['event'] = df_test_dist_pval['event'].replace(', n \(%\)', '', regex=True)


# ### the first occurrence time of events

# #### development cohort

# In[24]:


df_data_logs = pd.read_csv("tr_data_logs_combined.csv")
df_data_logs = df_data_logs.drop_duplicates()


# In[25]:


df_sorted = df_data_logs.sort_values(['PERSON_ID', 'EVENT'])

def get_first_occurrence(group):
    return group.iloc[0]

df_first_occurrence = df_sorted.groupby(['PERSON_ID', 'EVENT']).apply(get_first_occurrence)

df_first_occurrence = df_first_occurrence.reset_index(drop=True)


# In[26]:


df_tr_2 = pd.read_csv("../data/LXY_CKD_COHORT_0810.csv")
df_first_occurrence_days = df_first_occurrence.merge(df_tr_2[['PERSON_ID','startTimeOfObserve']])
df_first_occurrence_days = df_first_occurrence_days.merge(df_tr[['PERSON_ID','CLUSTER']])


# In[27]:


df_first_occurrence_days['DATE'] = pd.to_datetime(df_first_occurrence_days['DATE'] )
df_first_occurrence_days['观察起点'] = pd.to_datetime(df_first_occurrence_days['观察起点']).dt.date
df_first_occurrence_days['观察起点'] = pd.to_datetime(df_first_occurrence_days['观察起点'])
df_first_occurrence_days['时间间隔'] = df_first_occurrence_days['DATE'] - df_first_occurrence_days['观察起点'] 
df_first_occurrence_days['时间间隔'] = [x.days for x in df_first_occurrence_days['时间间隔']]


# In[28]:


# 开发队列
df_cluster_1 = df_tr[df_tr['CLUSTER'] == 1]
df_cluster_2 = df_tr[df_tr['CLUSTER'] == 2]
n_all = len(df_tr)
n_cluster_1 = len(df_cluster_1)
n_cluster_2 = len(df_cluster_2)

res_all = pd.DataFrame(columns=['事件类别', '事件',
                                '人数1', '内部频率1', '整体频率1', '时间间隔1',
                                '人数2', '内部频率2', '整体频率2', '时间间隔2',
                                '分布P-Value', '时间P-Value'])
for i in range(3):
    event_type = event_type_list[i]
    event_list = event_list_list[i]
    res = pd.DataFrame(columns=['事件类别', '事件',
                                '人数1', '内部频率1', '整体频率1', '时间间隔1',
                                '人数2', '内部频率2', '整体频率2', '时间间隔2',
                                '分布P-Value', '时间P-Value'])
    for event in event_list:
        # 事件频率
        n_1 = int(df_cluster_1[[event]].sum())
        freq_in_1 = n_1/n_cluster_1
        freq_out_1 = n_1/n_all
        n_2 = int(df_cluster_2[[event]].sum())
        freq_in_2 = n_2/n_cluster_2
        freq_out_2 = n_2/n_all
        
        # 事件分布P-Value
        dist_pval = df_tr_dist_pval.loc[df_tr_dist_pval['事件'] == event, 'P-Value'].item()
        
        # 时间间隔
        df_event = df_first_occurrence_days[df_first_occurrence_days['EVENT'] == event]
        grouped = df_event.groupby('CLUSTER')
        
        median = grouped['时间间隔'].median()
        q1 = grouped['时间间隔'].quantile(0.25)
        q3 = grouped['时间间隔'].quantile(0.75)
        days_1 = str(median[1]) + ' [' + str(q1[1]) + ',' + str(q3[1]) + ']'
        days_2 = str(median[2]) + ' [' + str(q1[2]) + ',' + str(q3[2]) + ']'
        
        # 时间间隔P-Value
        cluster1_data = df_event[df_event['CLUSTER'] == 1]['时间间隔']
        cluster2_data = df_event[df_event['CLUSTER'] == 2]['时间间隔']
        statistic, time_pval = mannwhitneyu(cluster1_data, cluster2_data) # Mann-Whitney U检验（也称为Wilcoxon秩和检验）
        if time_pval < 0.001:
            time_pval = '<0.001'
        else:
            time_pval = "{:.3f}".format(time_pval)
        
        res.loc[len(res)] = [event_type, event, 
                             n_1, freq_in_1, freq_out_1, median[1],
                             n_2, freq_in_2, freq_out_2, median[2],
                             dist_pval, time_pval]
    
    res = res.sort_values(by =['内部频率2','整体频率2'], ascending=False)
    res_all = pd.concat([res_all, res])


# In[30]:


df_tr_event = res_all.copy()


# In[31]:


df_tr_event.to_excel('event_stats_dev.xlsx', index=False, encoding='utf-8-sig')


# In[32]:


import matplotlib as mpl
from matplotlib.font_manager import FontProperties

font_path = 'SimHei.ttf'  
font_prop = FontProperties(fname=font_path)


# In[33]:


df_tr_event_pval = df_tr_event.copy()
df_tr_event_pval['分布P-Value'] = pd.to_numeric(df_tr_event_pval['分布P-Value'], errors='coerce')
df_tr_event_pval['时间P-Value'] = pd.to_numeric(df_tr_event_pval['时间P-Value'], errors='coerce')


# In[36]:


# 特定事件
dist_event_list = ['高血压','贫血','睡眠障碍','骨矿物质代谢异常','心力衰竭',
                   '甲状旁腺功能亢进','糖尿病', '冠心病',
                   '肾功能异常', '电解质异常','甲状腺功能异常','贫血指标异常','脂异常', 
                   '凝血功能异常','血糖异常','超敏c反应蛋白异常','血气分析异常',
                   '心脑血管指标异常','血沉偏高',
                   '抗凝血药','抗贫血药','钙补充剂','钙离子通道阻滞剂','骨化三醇',
                   '降血糖药','抗血小板药物','ARB','调血脂药','尿激酶',
                   'ACEI']
dist_event_list_eng = ['Hypertension','Anemia','Sleep disorder','Metabolic bone disease','Heart failure',
                   'Hyperparathyroidism','Diabetes','Coronary artery disease',
                   'Abnormal renal function', 'Electrolyte imbalance','Abnormal thyroid function',
                   'Abnormal anemia markers','Dyslipidemia', 'Abnormal coagulation function',
                   'Abnormal blood glucose','Abnormal hs-CRP','Abnormal arterial blood gas',
                   'High homocysteine', 'High erythrocyte sedimentation rate',
                   'Anticoagulants','Antianemia drugs','Calcium supplements','CCB','Calcitriol',
                   'Antihyperglycemic drugs','Antiplatelet drugs','ARB','Antihyperlipidemic drugs','Urokinase',
                   'ACEI']
df_tr_event_select = df_tr_event_pval[df_tr_event_pval['event'].isin(dist_event_list)]
df_tr_event_select['Event'] = dist_event_list_eng 


# ##### Overall freq

# In[37]:


# 选择事件
df_plot = df_tr_event_select.copy()

# 计算整体频率的平均值并进行排序
# df_plot['整体频率均值'] = df_plot[['整体频率1', '整体频率2']].mean(axis=1)
df_plot = df_plot.sort_values('整体频率2')

# 设置图形大小
plt.figure(figsize=(12, 16))
plt.subplots_adjust(left=0.4) 

# 绘制水平柱状图
bar_height = 0.3
index = np.arange(len(df_plot))
labels = df_plot['Event']
frequency1 = df_plot['整体频率1']
frequency2 = df_plot['整体频率2']

plt.barh(index, frequency1, bar_height, label='Subphenotype 1', color='#82B0D2')
plt.barh(index + bar_height, frequency2, bar_height, label='Subphenotype 2', color='#FFBE7A')

# 添加标签和标题
# plt.xlabel('Totel Freq', fontproperties=font_prop)
# plt.ylabel('Event', fontproperties=font_prop)
# plt.title('事件的整体频率', fontproperties=font_prop)
plt.xlabel('Total frequency', fontsize=14, weight='bold')
# plt.ylabel('Event', fontsize=16)

# 添加刻度标签（处理事件名称换行显示）
# wrapped_labels = [textwrap.fill(label, 25) for label in labels]  
plt.yticks(index + bar_height/2, labels, fontsize=14, weight='bold')
# plt.xticks(fontsize=14, weight='bold')
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=14, weight='bold')

# 添加图例
# plt.legend(prop=font_prop)
plt.legend(fontsize=14)

# 保存图像
plt.savefig('事件统计/开发队列整体频率柱状图_0815.tiff', dpi=300)


# #### validation cohort

# In[40]:


df_data_logs = pd.read_csv("test_data_logs_combined.csv")
df_data_logs = df_data_logs.drop_duplicates()


# In[41]:


# 按照PERSON_ID和EVENT排序数据框
df_sorted = df_data_logs.sort_values(['PERSON_ID', 'EVENT'])

# 定义一个自定义函数，用于获取每个组的第一行
def get_first_occurrence(group):
    return group.iloc[0]

# 按照PERSON_ID和EVENT分组，并应用自定义函数获取每个组的第一行
df_first_occurrence = df_sorted.groupby(['PERSON_ID', 'EVENT']).apply(get_first_occurrence)

# 重置索引
df_first_occurrence = df_first_occurrence.reset_index(drop=True)


# In[42]:


# 计算患者观察起点
df_test_2 = pd.read_csv("市二/LXY_CKD_VISIT_2.csv")
df_test_2['VISIT_START_DATE'] = pd.to_datetime(df_test_2['VISIT_START_DATE']).dt.date
df_fst_date = df_test_2.groupby('PERSON_ID')['VISIT_START_DATE'].min().reset_index()


# In[43]:


# 增加患者观察起点
df_fst_date = df_fst_date.rename(columns = {'VISIT_START_DATE':'观察起点'})
df_first_occurrence_days = df_first_occurrence.merge(df_fst_date)

# 增加患者亚型
df_first_occurrence_days = df_first_occurrence_days.merge(df_test[['PERSON_ID','CLUSTER']])


# In[44]:


# 计算时间间隔
df_first_occurrence_days['DATE'] = pd.to_datetime(df_first_occurrence_days['DATE'] )
df_first_occurrence_days['观察起点'] = pd.to_datetime(df_first_occurrence_days['观察起点']).dt.date
df_first_occurrence_days['观察起点'] = pd.to_datetime(df_first_occurrence_days['观察起点'])
df_first_occurrence_days['时间间隔'] = df_first_occurrence_days['DATE'] - df_first_occurrence_days['观察起点'] 
df_first_occurrence_days['时间间隔'] = [x.days for x in df_first_occurrence_days['时间间隔']]


# In[45]:


# 验证队列
df_cluster_1 = df_test[df_test['CLUSTER'] == 1]
df_cluster_2 = df_test[df_test['CLUSTER'] == 2]
n_all = len(df_test)
n_cluster_1 = len(df_cluster_1)
n_cluster_2 = len(df_cluster_2)

res_all = pd.DataFrame(columns=['事件类别', '事件',
                                '人数1', '内部频率1', '整体频率1', '时间间隔1',
                                '人数2', '内部频率2', '整体频率2', '时间间隔2',
                                '分布P-Value', '时间P-Value'])
for i in range(3):
    event_type = event_type_list[i]
    event_list = event_list_list[i]
    res = pd.DataFrame(columns=['事件类别', '事件',
                                '人数1', '内部频率1', '整体频率1', '时间间隔1',
                                '人数2', '内部频率2', '整体频率2', '时间间隔2',
                                '分布P-Value', '时间P-Value'])
    for event in event_list:
        if event in event_list_test:
            # 事件频率
            n_1 = int(df_cluster_1[[event]].sum())
            freq_in_1 = n_1/n_cluster_1
            freq_out_1 = n_1/n_all
            n_2 = int(df_cluster_2[[event]].sum())
            freq_in_2 = n_2/n_cluster_2
            freq_out_2 = n_2/n_all

            # 事件分布P-Value
            dist_pval = df_test_dist_pval.loc[df_test_dist_pval['事件'] == event, 'P-Value'].item()

            # 时间间隔
            df_event = df_first_occurrence_days[df_first_occurrence_days['EVENT'] == event]
            grouped = df_event.groupby('CLUSTER')

            median = grouped['时间间隔'].median()
#             q1 = grouped['时间间隔'].quantile(0.25)
#             q3 = grouped['时间间隔'].quantile(0.75)
#             days_1 = str(median[1]) + ' [' + str(q1[1]) + ',' + str(q3[1]) + ']'
#             days_2 = str(median[2]) + ' [' + str(q1[2]) + ',' + str(q3[2]) + ']'

            if len(median) == 1:
                if median.index == 1:
                    median[2] = 0
                else:
                    median[1] = 0
                time_pval = '-'
            else:    
                # 时间间隔P-Value
                cluster1_data = df_event[df_event['CLUSTER'] == 1]['时间间隔']
                cluster2_data = df_event[df_event['CLUSTER'] == 2]['时间间隔']
                statistic, time_pval = mannwhitneyu(cluster1_data, cluster2_data) # Mann-Whitney U检验（也称为Wilcoxon秩和检验）
                if time_pval < 0.001:
                    time_pval = '<0.001'
                else:
                    time_pval = "{:.3f}".format(time_pval)

            # 存入dataframe
#             res.loc[len(res)] = [event_type, event, 
#                                  n_1, freq_in_1, freq_out_1, days_1,
#                                  n_2, freq_in_2, freq_out_2, days_2,
#                                  p_value]
            res.loc[len(res)] = [event_type, event, 
                                 n_1, freq_in_1, freq_out_1, median[1],
                                 n_2, freq_in_2, freq_out_2, median[2],
                                 dist_pval, time_pval]
    
    res = res.sort_values(by =['内部频率2','整体频率2'], ascending=False)
    res_all = pd.concat([res_all, res])



# In[47]:


df_test_event = res_all.copy()
df_test_event


# In[48]:


df_test_event.to_excel('事件统计/验证队列事件统计.xlsx', index=False, encoding='utf-8-sig')


# In[49]:


df_test_event_pval = df_test_event.copy()
df_test_event_pval['分布P-Value'] = pd.to_numeric(df_test_event_pval['分布P-Value'], errors='coerce')
df_test_event_pval['时间P-Value'] = pd.to_numeric(df_test_event_pval['时间P-Value'], errors='coerce')


# ##### 事件选择

# In[50]:


# 分布P-Value<0.05
df_test_event_dist = df_test_event_pval.loc[(df_test_event_pval['分布P-Value'] <= 0.05) | (pd.isna(df_test_event_pval['分布P-Value']))]
len(df_test_event_dist)


# In[51]:


# 事件P-Value<0.05
df_test_event_time = df_test_event_pval.loc[(df_test_event_pval['时间P-Value'] <= 0.05) | (pd.isna(df_test_event_pval['时间P-Value']))]
len(df_test_event_time)


# In[52]:


# 特定事件
df_test_event_select = df_test_event_pval[df_test_event_pval['event'].isin(dist_event_list)]


# In[53]:


# 特定事件
df_test_event_select = df_test_event_pval[df_test_event_pval['event'].isin(dist_event_list)]
df_test_event_select['Event'] = dist_event_list_eng


# ##### 整体频率

# In[54]:


# 选择事件
df_plot = df_test_event_select.copy()

# 计算整体频率的平均值并进行排序
# df_plot['整体频率均值'] = df_plot[['整体频率1', '整体频率2']].mean(axis=1)
df_plot = df_plot.sort_values('整体频率2')

# 设置图形大小
plt.figure(figsize=(12, 16))
plt.subplots_adjust(left=0.4) 

# 绘制水平柱状图
bar_height = 0.3
index = np.arange(len(df_plot))
labels = df_plot['Event']
frequency1 = df_plot['整体频率1']
frequency2 = df_plot['整体频率2']

plt.barh(index, frequency1, bar_height, label='Subphenotype 1', color='#82B0D2')
plt.barh(index + bar_height, frequency2, bar_height, label='Subphenotype 2', color='#FFBE7A')

# 添加标签和标题
# plt.xlabel('Totel Freq', fontproperties=font_prop)
# plt.ylabel('Event', fontproperties=font_prop)
# plt.title('事件的整体频率', fontproperties=font_prop)
plt.xlabel('Total frequency', fontsize=14, weight='bold')
# plt.ylabel('Event', fontsize=16)

# 添加刻度标签（处理事件名称换行显示）
# wrapped_labels = [textwrap.fill(label, 25) for label in labels]  
plt.yticks(index + bar_height/2, labels, fontsize=14, weight='bold')
plt.xticks(fontsize=14, weight='bold')

# 添加图例
# plt.legend(prop=font_prop)
plt.legend(fontsize=14)

# 保存图像
plt.savefig('事件统计/验证队列整体频率柱状图_0815.tiff', dpi=300)


# In[ ]:




