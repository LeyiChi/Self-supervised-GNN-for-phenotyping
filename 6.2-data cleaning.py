#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from tableone import TableOne


# ### event logs

# In[3]:


df_data_logs = pd.read_csv("tr_data_logs.csv")
df_data_logs = df_data_logs.drop(df_data_logs.columns[0], axis=1)
df_data_logs.head()





# In[7]:


df_data_logs.to_csv('zheyi_data_logs_combined.csv', index=False, encoding='utf-8-sig')


# ### baseline features

# In[8]:


df = pd.read_csv("zheyi_subtype_data.csv")


# In[10]:


df2 = df[['PERSON_ID','CLUSTER',
          'Sex','Age','height','weight',
          'smoking','drinking','education','disease.awareness','time','status']]


# ### diagnosis

# In[11]:


df_diag = df[['慢性肾脏病','血液透析','高血压','骨矿物质代谢异常','贫血','糖尿病','动静脉瘘','肾炎','甲状旁腺功能亢进','心力衰竭','胃肠道疾病','冠心病','糖尿病肾病','多囊肾','肺部感染','周围血管疾病','睡眠障碍','上呼吸道感染','乙型病毒性肝炎','高脂血症','骨髓瘤','感染','水电解质平衡失调','脑梗死','骨折','肝囊肿','肾功能检查的异常结果','便秘','胆囊结石','肾病综合征','前列腺增生','甲状腺结节','痛风','阿尔茨海默病','房颤','胸腔积液','发热','神经病变','甲状腺功能减退','高尿酸血症','慢性肾脏病性心肌病','肾癌','支气管病变','营养不良','肾移植状态']]


# In[13]:


df2 = pd.concat([df2, df_diag], axis=1)




df_drug = df[['左卡尼汀','钙离子通道阻滞剂','钙补充剂','维生素','骨化三醇','β受体阻滞剂',
              'ARB','抗血小板药物','质子泵抑制剂','复方α酮酸','氨基酸','尿激酶',
              '利多卡因','利尿剂','硝酸酯类药物','司维拉姆','人血白蛋白','多磺酸粘多糖霜','阿司匹林',
              '碳酸镧','多巴胺','ACEI','磷酸钠盐','α受体阻滞剂','米多君','丙泊酚','吗替麦考酚酯',
              '羟苯磺酸钙','生长抑素','鲑降钙素','多巴酚丁胺','硫酸鱼精蛋白','甲氧氯普胺','血凝酶',
              '腹膜透析液','高渗枸橼酸盐嘌呤','左甲状腺素钠','塞来昔布']]




df2 = pd.concat([df2, df_drug], axis=1)




df_other = df[['胸部_X','腹部_X','心电图_ECG','肺部_CT','床边摄片_X','腹部_CT','磁共振扫描_MR','手_X','甲状旁腺显像_ECT','头颅_CT','胸部_CT','X线计算机体层特殊三维成像_CT','胃镜_ES','头颅_MR','腰椎_X','膝关节_X','骨盆_X','URO','冠状动脉CTA_CT','腕关节_X','上肢动脉CTA_CT','股骨_X','肠镜_ES','颈部_CT','肺部HR_CT','足_X','上肢静脉CTA_CT',
               '甲状旁腺切除术','甲状旁腺移植术','甲状腺切除术','前臂移植术','肾切除术']]


# In[66]:


examination_list = ['胸部_X','腹部_X','心电图_ECG','肺部_CT','床边摄片_X','腹部_CT','磁共振扫描_MR','手_X','甲状旁腺显像_ECT','头颅_CT','胸部_CT','X线计算机体层特殊三维成像_CT','胃镜_ES','头颅_MR','腰椎_X','膝关节_X','骨盆_X','URO','冠状动脉CTA_CT','腕关节_X','上肢动脉CTA_CT','股骨_X','肠镜_ES','颈部_CT','肺部HR_CT','足_X','上肢静脉CTA_CT']
surgery_list = ['甲状旁腺切除术','甲状旁腺移植术','甲状腺切除术','前臂移植术','肾切除术']


# In[67]:


df2 = pd.concat([df2, df_other], axis=1)


# In[65]:


df2.to_csv('zheyi_subtype_combined_data.csv', index=False, encoding='utf-8-sig')

