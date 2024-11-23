#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from tableone import TableOne
from tqdm.notebook import tqdm
# import shap


# ### visit labels

# In[3]:


df_data_logs = pd.read_csv("zheyi_data_logs_combined.csv")


# In[4]:


df_data_logs_2 = df_data_logs[['PERSON_ID', 'VISIT_OCCURRENCE_ID', 'EVENT', 'TYPE']]
df_data_logs_2 = df_data_logs_2.drop_duplicates()


# In[5]:


df_visit = df_data_logs_2.groupby(['PERSON_ID','VISIT_OCCURRENCE_ID']).agg({'EVENT':' '.join, 'TYPE':' '.join}).reset_index()
df_visit.columns = df_visit.columns.get_level_values(0)


# In[7]:


result = pd.DataFrame(columns=['PERSON_ID', 'VISIT_OCCURRENCE_ID', 'VISIT_NUMBER', 'FST_ABN', 'EVENT', 'TYPE'])

grouped = df_visit.groupby('PERSON_ID')

# 遍历每个分组
for person_id, group in tqdm(grouped):
    # 初始化VISIT_NUMBER为递增编号
    group['VISIT_NUMBER'] = range(1, len(group) + 1)
    # 查找TYPE里第一次出现'实验室检验'的索引位置
    first_a_index = group[group['TYPE'].str.contains('实验室检验')].index.min()
    # 添加FST_ABN列，对应索引位置之前的值为0，之后的值为1
    group['FST_ABN'] = 0
    if first_a_index is not None:
        group.loc[group.index == first_a_index, 'FST_ABN'] = 1
        group.loc[group.index > first_a_index, 'FST_ABN'] = 2
    # 将当前分组的结果添加到最终结果DataFrame中
    result = pd.concat([result, group])


# In[8]:


df_visit_label = result[['PERSON_ID', 'VISIT_OCCURRENCE_ID', 'VISIT_NUMBER', 'FST_ABN', 'EVENT']]
df_visit_label.head()


# ### data preparing

# In[9]:


df = pd.read_csv("zheyi_subtype_combined_data_R.csv", encoding='gb18030')
df = df.rename(columns = {'首次透析年龄':'年龄'})
df.head()


# In[10]:


# df['文化程度'] = df['文化程度'].fillna(4)
df['disease.awareness'] = df['disease.awareness'].fillna(3)


# In[11]:


# 缺失值处理
# df = df.fillna(df.median())


# In[12]:


# 使用独热编码表示结局事件
one_hot_encoded = pd.get_dummies(df['status'])
df = pd.concat([df, one_hot_encoded], axis=1)


# In[13]:


# 将聚类2作为正样本，聚类1作为负样本
df['CLUSTER'] = df['CLUSTER'].map({1: 0, 2: 1})


# ### data split

# In[14]:


# df_train, df_test = train_test_split(df, test_size = 0.3, random_state = 0, stratify = df['CLUSTER'].values)
df_train = df[df['PERSON_ID'].isin(train_test_pid['train_pid']), :]
df_test = df[df['PERSON_ID'].isin(train_test_pid['test_pid']), :]


# ### subphenotype prediction model

# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import brier_score_loss

def model_evaluation(y_true, y_pred, y_pred_prob):
    [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred)
    print('[[TN, FP], [FN, TP]]:', [[TN, FP], [FN, TP]])
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    sen = TP * 1.0 / (TP + FN) if (TP + FN) != 0 else np.nan
    spe = TN * 1.0 / (TN + FP) if (TN + FP) != 0 else np.nan
    ppv = TP * 1.0 / (TP + FP) if (TP + FP) != 0 else np.nan
    npv = TN * 1.0 / (TN + FN) if (TN + FN) != 0 else np.nan
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)  
    auprc = average_precision_score(y_true, y_pred_prob)
    # bs = brier_score_loss(y_true, y_pred_prob)
    res = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'ppv', 'npv', 'F1-score', 'AUC', 'AUPRC'])
    res.loc[0] = [acc, precision, recall, sen, spe, ppv, npv, f1, auc, auprc]
    
    return(res)


# ### feature selection

# In[19]:


# 剔除特征
drop_list = ['PERSON_ID', 'CLUSTER', '观察窗', '结局事件', '首次指标异常时间',
             '死亡', '维持血透', '肾移植', '腹透']


# In[20]:


X_train_all = df_train.drop(drop_list, axis=1)
y_train = df_train['CLUSTER']


# In[21]:


X_test_all = df_test.drop(drop_list, axis=1)
y_test = df_test['CLUSTER']


# #### basic info + initial info at first visit

# In[25]:


fst_visit_list = df_visit_label.loc[df_visit_label['VISIT_NUMBER']==1, 'VISIT_OCCURRENCE_ID']


# In[26]:


df_data_logs_fst_visit = df_data_logs[df_data_logs['VISIT_OCCURRENCE_ID'].isin(fst_visit_list)]


# In[27]:


df_data_logs_fst_visit = df_data_logs_fst_visit.drop(['TYPE','DATE'], axis=1)
df_data_logs_fst_visit = df_data_logs_fst_visit.drop_duplicates()


# In[28]:


df_fst_visit = df_data_logs_fst_visit.pivot(index='PERSON_ID', columns='EVENT', values='EVENT')

df_fst_visit = df_fst_visit.fillna(0)

df_fst_visit = df_fst_visit.applymap(lambda x: 1 if x != 0 else 0)

df_fst_visit = df_fst_visit.reset_index()


# In[29]:


# basic info
basic_col = ['性别', '年龄', '身高', '体重', 'BMI', '吸烟', '饮酒', '对疾病认识', '药物依赖.药瘾.吸毒']


# In[30]:


X_train_fst = df_train[['PERSON_ID'] + basic_col].merge(df_fst_visit, how = 'left')
X_train_fst = X_train_fst.fillna(0)


# In[31]:


X_test_fst = df_test[['PERSON_ID'] + basic_col].merge(df_fst_visit, how = 'left')
X_test_fst = X_test_fst.fillna(0)


# #### start feature selection



import lightgbm as lgb
# from lightgbm import LGBMClassifier

feature_importance = pd.DataFrame()
rep_times = 100
for i in range(rep_times):
    lgb_best = lgb.LGBMClassifier(colsample_bytree=0.7, learning_rate=0.1,
                     max_bin=15, n_estimators=30, num_leaves=10,
                     reg_alpha=0.2, reg_lambda=0.1, subsample=0.7).fit(X_train_fst, y_train)

    booster = lgb_best.booster_
    importance = booster.feature_importance(importance_type='gain')
    feature_name = booster.feature_name()
    if i == 0:
        feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} ).sort_values(by=['importance'],
                                                                                                              ascending=False)
    else:
        tmp = pd.DataFrame({'feature_name':feature_name,'importance':importance} ).sort_values(by=['importance'],
                                                                                                              ascending=False)
        feature_importance = feature_importance.merge(tmp, how='inner', on='feature_name')
        feature_importance['importance'] = feature_importance['importance_x'] + feature_importance['importance_y']
        feature_importance = feature_importance.drop(['importance_x', 'importance_y'], axis=1)
    # feature_importance.to_csv('feature_importance.csv',index=False)
feature_importance['importance'] = feature_importance['importance']/rep_times


# In[34]:


feature_importance[feature_importance['importance'] > 0]['feature_name'].values


# In[35]:


drop_list_lgb = feature_importance[feature_importance['importance'] < 28]['feature_name'].values

col_list = list(set(X_train_fst.columns) - set(drop_list_lgb))
col_index = []
idx = -1

for col in X_train_fst.columns:
    idx = idx + 1
    if col in drop_list_lgb:
        col_list.remove(col)
    else:
        col_index.append(idx)


# In[36]:


X_train_fst = X_train_fst[col_list]
X_test_fst = X_test_fst[col_list]


# #### end feature selection

# ### Model compare

# In[41]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# In[42]:


from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def plot_combined_curves(y_true_train, y_proba_train, y_true_test, y_proba_test, save_path):
    # Compute ROC curve data for train set
    fpr_train, tpr_train, thresholds_roc_train = roc_curve(y_true_train, y_proba_train)
    auc_train = roc_auc_score(y_true_train, y_proba_train)

    # Compute PR curve data for train set
    precision_train, recall_train, thresholds_pr_train = precision_recall_curve(y_true_train, y_proba_train)
    avg_precision_train = average_precision_score(y_true_train, y_proba_train)

    # Compute ROC curve data for test set
    fpr_test, tpr_test, thresholds_roc_test = roc_curve(y_true_test, y_proba_test)
    auc_test = roc_auc_score(y_true_test, y_proba_test)

    # Compute PR curve data for test set
    precision_test, recall_test, thresholds_pr_test = precision_recall_curve(y_true_test, y_proba_test)
    avg_precision_test = average_precision_score(y_true_test, y_proba_test)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot ROC curve
    ax1.plot(fpr_train, tpr_train, label='Train ROC curve (AUC = {:.3f})'.format(auc_train), color='orange')
    ax1.plot(fpr_test, tpr_test, label='Test ROC curve (AUC = {:.3f})'.format(auc_test), color='red')
    ax1.plot([0, 1], [0, 1], 'k--')  # diagonal line
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    # ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.annotate('A', xy=(0, 1.02), xycoords='axes fraction', fontsize=12, fontweight='bold', color='black')
    
    # Plot PR curve
    ax2.plot(recall_train, precision_train, label='Train PR curve (Avg Precision = {:.3f})'.format(avg_precision_train), color='orange')
    ax2.plot(recall_test, precision_test, label='Test PR curve (Avg Precision = {:.3f})'.format(avg_precision_test), color='red')
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    # ax2.set_title('Precision-Recall Curve', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.annotate('B', xy=(0, 1.02), xycoords='axes fraction', fontsize=12, fontweight='bold', color='black')
    
    # Add grid lines
    ax1.grid(True, which='both', linestyle='dotted', linewidth=0.5)
    ax2.grid(True, which='both', linestyle='dotted', linewidth=0.5)

    # Adjust x and y ticks to show major and minor ticks for gridlines
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(base=0.125))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(base=0.125))
    
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(base=0.125))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(base=0.125))

    # Set the specific tick labels for the axes
    ax1.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax1.set_xticklabels([0.00, 0.25, 0.50, 0.75, 1.00], fontweight='bold')
    ax1.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax1.set_yticklabels([0.00, 0.25, 0.50, 0.75, 1.00], fontweight='bold')
    
    ax2.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax2.set_xticklabels([0.00, 0.25, 0.50, 0.75, 1.00], fontweight='bold')
    ax2.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax2.set_yticklabels([0.00, 0.25, 0.50, 0.75, 1.00], fontweight='bold')
    
    # Adjust spacing between subplots
    fig.tight_layout()

    # Save the combined plot
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


# #### LightGBM

# In[43]:


X_train_fst_eng = X_train_fst.copy()

english_list = [
    'Gender', 'Age', 'BMI', 'Smoking', 'Alcohol consumption', 'Disease awareness',
    'Drug dependence. Drug addiction. Drug abuse', 'ACE inhibitors', 'ARB', 'Alpha-adrenergic blockers', 'Beta blockers', 'Upper respiratory tract infection', 'Propofol',
    'Hepatitis B virus infection', 'Hepatitis B', 'Lactic acidosis', 'Human serum albumin', 'Constipation', 'Other antihypertensive drugs', 'Coronary heart disease', 'Coagulation dysfunction',
    'Diuretics', 'Benign prostatic hyperplasia', 'Forearm transplantation', 'Arteriovenous fistula', 'Fever', 'Cisatracurium',
    'Peripheral vascular disease', 'Celecoxib', 'Polycystic kidney disease', 'Dopamine', 'Dobutamine', 'Polysulfated glycosaminoglycan ointment', 'Berberine',
    'Urokinase', 'Levothyroxine sodium', 'Heart failure', 'Abnormal myocardial injury indicators',
    'Abnormal cardiovascular and cerebrovascular indicators',
    'Abnormal heart failure indicators', 'Infection',
    'Abnormal infection markers', 'Chronic kidney disease-related cardiomyopathy', 'Atrial fibrillation', 'Anticoagulants', 'Antiarrhythmic drugs', 'Anti-anginal drugs', 'Anti-heart failure drugs', 'Cold medicine',
    'Antibiotics', 'Abnormal antibiotic sensitivity indicators', 'Antiviral drugs', 'Antiplatelet drugs', 'Antianemia drugs', 'Adrenergic drugs', 'Bronchial lesions',
    'Antitussive and anti-asthmatic drugs', 'Antidiarrheal drugs', 'Abnormal oxygenation blood routine', 'Amino acids', 'Electrolyte imbalances',
    'Laxatives', 'Low GPDA', 'Somatostatin', 'Metoclopramide', 'Hyperparathyroidism', 'Parathyroid transplantation',
    'Thyroidectomy', 'Hypothyroidism', 'Abnormal thyroid function', 'Thyroid nodules', 'Electrolyte abnormalities', 'Electrolyte supplements', 'Gout',
    'Sleep disorders', 'Nitrate ester drugs', 'Fish protein sulfate', 'Iodinated contrast agents', 'Lanthanum carbonate',
    # 'Sodium phosphate',
    'Neuropathy', 'Midodrine',
    'Diabetes', 'Diabetic nephropathy', 'Histamine receptor blockers', 'Vitamins', 'Calcium hydroxybenzenesulfonate', 'Polyvinylpyrrolidone iodine', 'Liver function abnormalities', 'Hepatic cysts', 'Gastrointestinal drugs', 'Pulmonary infection', 
    'Nephrectomy',
    'Kidney cancer', 'Abnormal tumor markers', 'Gastrointestinal diseases', 'Gallbladder stones', 'Pleural effusion', 'Dyslipidemia',
    'Cerebral infarction', 'Abnormal dehydrogenase markers', 'Moxifloxacin', 'Malnutrition', 'Nutritional supplements', 'Blood beta2 microglobulin abnormality', 'Thrombin', # 'Abnormal complete blood count',
    'Abnormal thrombosis markers', 'Abnormal blood gas analysis', 'High erythrocyte sedimentation rate', 'Abnormal blood glucose', 'Lipid-lowering drugs', 'Proton pump inhibitors', 'Anemia', 'Abnormal anemia markers',
    'Abnormal highly sensitive C-reactive protein', 'Abnormal reduced blood routine', 'Calcium channel blockers', 'Calcium supplements', 'Analgesics', 'Aspirin', 'Alzheimers disease',
    'Antidiabetic drugs', 'Nonsteroidal anti-inflammatory drugs', 'Calcitriol', 'Fracture', 'Metabolic bone disease', 'Multiple myeloma', 'Hyperuricemia',
    'Hyperlipidemia', 'Salmon calcitonin'
]

# X_train_fst_eng.columns = english_list
X_train_fst_eng.columns = np.array(english_list)[col_index]


# In[44]:


lgb_best = lgb.LGBMClassifier(colsample_bytree=0.7, learning_rate=0.1,
                     max_bin=15, n_estimators=30, num_leaves=10,
                     reg_alpha=0.2, reg_lambda=0.1, subsample=0.7).fit(X_train_fst, y_train)

y_pred_lgb_tr = lgb_best.predict(X_train_fst) 
y_proba_lgb_tr = lgb_best.predict_proba(X_train_fst)[:,1]
model_evaluation(y_train, y_pred_lgb_tr, y_proba_lgb_tr)


# In[45]:


y_pred_lgb = lgb_best.predict(X_test_fst) 
y_proba_lgb = lgb_best.predict_proba(X_test_fst)[:,1]
model_evaluation(y_test, y_pred_lgb, y_proba_lgb)


# In[46]:


# Save the combined ROC curve and PR curve plot
plot_combined_curves(y_train, y_proba_lgb_tr, y_test, y_proba_lgb, '4.1 lgb_plot_train_test.png')


# In[47]:


booster = lgb_best.booster_
importance = booster.feature_importance(importance_type='gain')
feature_name = booster.feature_name()
feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} ).sort_values(by=['importance'],
                                                                                                          ascending=False)


# ### SHAP（XGBoost）

# In[49]:


import shap


# In[50]:


shap.initjs()

# 创建Explainer对象
# explainer = shap.LinearExplainer(lr_fst, X_train_fst)
explainer = shap.TreeExplainer(lgb_best, X_train_fst)
# explainer = shap.Explainer(xgb_best)

# 计算SHAP值
shap_values = explainer.shap_values(X_train_fst)
shap_values.shape


# In[51]:


# 解释单个样本
plot_index = 0
shap.force_plot(explainer.expected_value, shap_values[plot_index, :], X_train_fst.iloc[plot_index, :])


# In[52]:


shap.summary_plot(shap_values, X_train_fst_eng, plot_type="bar", max_display = 50, plot_size = (10,8), show=False)
plt.savefig('./4.2. shap_importance.png')


# In[53]:


# 解释整个数据集
shap.summary_plot(shap_values, X_train_fst_eng, max_display = 50, plot_size = (10,8), show=False)
plt.savefig('./4.3. shap_summary.png')


# In[54]:


df_shap_values = pd.DataFrame(shap_values)
df_shap_values.columns = X_train_fst_eng.columns




# ### data for survival prediction model


df_train.to_csv('cox_train.csv', index=False, encoding='utf-8-sig')
df_test.to_csv('cox_test.csv', index=False, encoding='utf-8-sig')



