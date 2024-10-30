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


# In[4]:


# labtests
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('二氧化碳分压偏低|氧分压偏高|氧分压偏低|二氧化碳分压偏高|碳酸氢根偏低|标准碳酸氢盐偏低|碳酸氢根偏高|实际碱剩余偏低|酸碱度偏低|实际碱剩余偏高|标准碱剩余偏低|氧饱和度偏低|氧饱和度偏高|酸碱度偏高|总氧偏低|还原血红蛋白偏高|一氧化碳合血红蛋白偏高',
                                                          '血气分析异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('红细胞偏低|血红蛋白偏低|血小板偏低|红细胞偏高|血小板偏高|白细胞偏高|白细胞偏低|淋巴细胞偏低|中性粒细胞偏高|嗜酸性粒细胞偏高|嗜碱性粒细胞偏高|单核细胞偏低|嗜酸性粒细胞偏低|中性粒细胞偏低|血细胞压积偏低|单核细胞偏高|血红蛋白偏高',
                                                          '血常规异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('白球蛋白偏低|总蛋白偏低|白蛋白偏低|总胆红素偏高|间接胆红素偏低|酸碱度偏高|谷草谷丙偏低|谷氨酰转酞酶偏高|间接胆红素偏高|谷草转氨酶偏低|谷草转氨酶偏高|谷草谷丙偏高|谷丙转氨酶偏低|直接胆红素偏高|胆碱酯酶偏低|碱性磷酸酶偏高|球蛋白偏低|球蛋白偏高|总胆汁酸偏高|谷丙转氨酶偏高|腺苷酸脱氨酶偏高|谷氨酰转酞酶偏低',
                                                          '肝功能异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('肌酐偏高|尿素偏高|尿酸偏高|尿酸偏低|血beta2微球蛋白偏高|胱抑素c偏高',
                                                          '肾功能异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('甘油三酯偏高|总胆固醇偏低|高密度脂蛋白c偏低|极低密度脂蛋白c偏高|总胆固醇偏高|极低密度脂蛋白c偏低|低密度脂蛋白c偏低|低密度脂蛋白c偏高',
                                                          '脂异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('无机磷偏高|钾偏低|无机磷偏低|总钙偏低|钠偏低|氯偏低|总钙偏高|钾偏高|镁偏高',
                                                          '电解质异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('血糖偏高|血糖偏低',
                                                          '血糖异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('甲状旁腺激素偏高|磷酸肌酸激酶偏低|总三碘甲状腺原氨酸偏低',
                                                          '甲状腺功能异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('超敏c反应蛋白偏高|c反应蛋白偏高',
                                                          '超敏c反应蛋白异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('d二聚体偏高',
                                                          '血栓指标异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('国际标准化偏高|纤维蛋白原偏高|纤维蛋白原偏低|凝血酶原时间偏高|活化部分凝血活酶时间偏高',
                                                          '凝血功能异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('b型脑尿钠肽偏高',
                                                          '心衰指标异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('糖抗原125偏高',
                                                          '肿瘤标志物异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('转铁蛋白偏低|铁偏低|总铁结合力偏低|铁蛋白偏高|还原血红蛋白偏低|氧合血红蛋白偏低',
                                                          '贫血指标异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('乙型肝炎病毒核心抗体偏高|乙型肝炎病毒e抗体偏低|乙型肝炎病毒表面抗原偏高|乙型肝炎病毒表面抗体偏高',
                                                          '乙肝病毒指标异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('同型半胱氨酸偏高',
                                                          '心脑血管指标异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('肌钙蛋白偏高|磷酸肌酸激酶偏高|肌酸激酶同工酶偏高',
                                                          '心肌损伤指标异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('抗生素敏感性偏低|抗生素敏感性偏高',
                                                          '抗生素敏感性指标异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('降钙素原偏高|细菌偏高',
                                                          '感染指标异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('乳酸偏高',
                                                          '乳酸异常')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('羟丁酸脱氢酶偏高|乳酸脱氢酶偏高',
                                                          '脱氢酶异常')


# In[5]:


# medication
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('可乐定',
                                                          '其他抗高血压')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('胰岛素|格列奈类高血糖药|阿卡波糖',
                                                          '降血糖药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('他汀类药物|非诺贝特',
                                                          '调血脂药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('抗凝药物|华法令钠',
                                                          '抗凝血药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('氯化钾|硫酸镁',
                                                          '电解质补充药物')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('头孢|万古霉素|舒巴坦钠|沙星|西林|培南|氨基糖苷类抗生素|三唑类衍生物抗生素|大环内酯类抗生素|西司他丁钠',
                                                          '抗生素')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('吲哚美辛',
                                                          '非甾体抗炎药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('多潘立酮|莫沙必利|酪酸梭菌活菌|硫糖铝|铝碳酸镁',
                                                          '肠胃药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('地西泮|丙戊酸钠|唑吡坦|苯二氮䓬类镇静药',
                                                          '精神类药物')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('重组人促红素|叶酸|甲钴胺|铁补充剂|生血宁',
                                                          '抗贫血药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('曲美他嗪',
                                                          '抗心绞痛药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('氨溴索|愈创甘油醚|强力枇杷露|异丙托溴铵|布地奈德|糖皮质激素',
                                                          '止咳平喘药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('肠内营养乳剂|脂肪乳',
                                                          '营养剂')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('乳果糖|聚乙二醇4000|开塞露',
                                                          '泻药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('蒙脱石',
                                                          '止泻药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('酚麻美敏',
                                                          '抗感冒药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('抗组胺H1受体拮抗剂|异丙嗪',
                                                          '组胺受体阻断药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('托拉塞米',
                                                          '抗心衰药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('胺碘酮',
                                                          '抗心律失常药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('曲马多|去痛片',
                                                          '镇痛药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('核苷酸类逆转录酶抑制剂',
                                                          '抗病毒药')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('碘帕醇',
                                                          '碘造影剂')
df_data_logs['EVENT'] = df_data_logs['EVENT'].str.replace('去甲肾上腺素|间羟胺',
                                                          '拟肾上腺素药')


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


# ### Labtests

# In[14]:


df2['血常规异常'] = df[['红细胞偏低','血红蛋白偏低','血小板偏低','红细胞偏高','血小板偏高','白细胞偏高','白细胞偏低','淋巴细胞偏低','中性粒细胞偏高','嗜酸性粒细胞偏高','嗜碱性粒细胞偏高','单核细胞偏低','嗜酸性粒细胞偏低','中性粒细胞偏低','血细胞压积偏低','单核细胞偏高','血红蛋白偏高']].sum(axis=1)
df2['血常规异常'] = [1 if x>0 else 0 for x in df2['血常规异常']]


# In[15]:


df2['肝功能异常'] = df[['白球蛋白偏低','总蛋白偏低','白蛋白偏低','总胆红素偏高','间接胆红素偏低','酸碱度偏高','谷草谷丙偏低','谷氨酰转酞酶偏高','间接胆红素偏高','谷草转氨酶偏低','谷草转氨酶偏高','谷草谷丙偏高','谷丙转氨酶偏低','直接胆红素偏高','胆碱酯酶偏低','碱性磷酸酶偏高','球蛋白偏低','球蛋白偏高','总胆汁酸偏高','谷丙转氨酶偏高','腺苷酸脱氨酶偏高','谷氨酰转酞酶偏低']].sum(axis=1)
df2['肝功能异常'] = [1 if x>0 else 0 for x in df2['肝功能异常']]


# In[16]:


df2['肾功能异常'] = df[['肌酐偏高','尿素偏高', '尿酸偏高','尿酸偏低', '血beta2微球蛋白偏高', '胱抑素c偏高']].sum(axis=1)
df2['肾功能异常'] = [1 if x>0 else 0 for x in df2['肾功能异常']]


# In[17]:


df2['脂异常'] = df[['甘油三酯偏高','总胆固醇偏低','高密度脂蛋白c偏低','极低密度脂蛋白c偏高','总胆固醇偏高','极低密度脂蛋白c偏低','低密度脂蛋白c偏低','低密度脂蛋白c偏高']].sum(axis=1)
df2['脂异常'] = [1 if x>0 else 0 for x in df2['脂异常']]


# In[18]:


df2['电解质异常'] = df[['无机磷偏高','钾偏低','无机磷偏低','总钙偏低','钠偏低','氯偏低','总钙偏高','钾偏高','镁偏高']].sum(axis=1)
df2['电解质异常'] = [1 if x>0 else 0 for x in df2['电解质异常']]


# In[19]:


df2['血糖异常'] = df[['血糖偏高','血糖偏低']].sum(axis=1)
df2['血糖异常'] = [1 if x>0 else 0 for x in df2['血糖异常']]


# In[20]:


df2['血气分析异常'] = df[['二氧化碳分压偏低','氧分压偏高','氧分压偏低','二氧化碳分压偏高','碳酸氢根偏低','标准碳酸氢盐偏低','碳酸氢根偏高','实际碱剩余偏低','酸碱度偏低','实际碱剩余偏高','标准碱剩余偏低','氧饱和度偏低','氧饱和度偏高','酸碱度偏高','总氧偏低','还原血红蛋白偏高','一氧化碳合血红蛋白偏高']].sum(axis=1)
df2['血气分析异常'] = [1 if x>0 else 0 for x in df2['血气分析异常']]


# In[21]:


df2['甲状腺功能异常'] = df[['甲状旁腺激素偏高', '磷酸肌酸激酶偏低', '总三碘甲状腺原氨酸偏低']].sum(axis=1)
df2['甲状腺功能异常'] = [1 if x>0 else 0 for x in df2['甲状腺功能异常']]


# In[22]:


df2['超敏c反应蛋白异常'] = df[['超敏c反应蛋白偏高', 'c反应蛋白偏高']].sum(axis=1)
df2['超敏c反应蛋白异常'] = [1 if x>0 else 0 for x in df2['超敏c反应蛋白异常']]


# In[23]:


df2['血栓指标异常'] = df[['d二聚体偏高']].sum(axis=1)
df2['血栓指标异常'] = [1 if x>0 else 0 for x in df2['血栓指标异常']]


# In[24]:


df2['凝血功能异常'] = df[['国际标准化偏高','纤维蛋白原偏高','纤维蛋白原偏低','凝血酶原时间偏高', '活化部分凝血活酶时间偏高']].sum(axis=1)
df2['凝血功能异常'] = [1 if x>0 else 0 for x in df2['凝血功能异常']]


# In[25]:


df2['心衰指标异常'] = df[['b型脑尿钠肽偏高']].sum(axis=1)
df2['心衰指标异常'] = [1 if x>0 else 0 for x in df2['心衰指标异常']]


# In[26]:


df2['肿瘤标志物异常'] = df[['糖抗原125偏高']].sum(axis=1)
df2['肿瘤标志物异常'] = [1 if x>0 else 0 for x in df2['肿瘤标志物异常']]


# In[27]:


df2['贫血指标异常'] = df[['转铁蛋白偏低','铁偏低','总铁结合力偏低','铁蛋白偏高','还原血红蛋白偏低','氧合血红蛋白偏低']].sum(axis=1)
df2['贫血指标异常'] = [1 if x>0 else 0 for x in df2['贫血指标异常']]


# In[28]:


df2['乙肝病毒指标异常'] = df[['乙型肝炎病毒核心抗体偏高','乙型肝炎病毒e抗体偏低','乙型肝炎病毒表面抗原偏高','乙型肝炎病毒表面抗体偏高']].sum(axis=1)
df2['乙肝病毒指标异常'] = [1 if x>0 else 0 for x in df2['乙肝病毒指标异常']]


# In[29]:


df2['心脑血管指标异常'] = df[['同型半胱氨酸偏高']].sum(axis=1)
df2['心脑血管指标异常'] = [1 if x>0 else 0 for x in df2['心脑血管指标异常']]


# In[30]:


df2['心肌损伤指标异常'] = df[['肌钙蛋白偏高', '磷酸肌酸激酶偏高', '肌酸激酶同工酶偏高']].sum(axis=1)
df2['心肌损伤指标异常'] = [1 if x>0 else 0 for x in df2['心肌损伤指标异常']]


# In[31]:


df2['抗生素敏感性指标异常'] = df[['抗生素敏感性偏低','抗生素敏感性偏高']].sum(axis=1)
df2['抗生素敏感性指标异常'] = [1 if x>0 else 0 for x in df2['抗生素敏感性指标异常']]


# In[32]:


df2['感染指标异常'] = df[['降钙素原偏高', '细菌偏高']].sum(axis=1)
df2['感染指标异常'] = [1 if x>0 else 0 for x in df2['感染指标异常']]


# In[33]:


df2['乳酸异常'] = df[['乳酸偏高']].sum(axis=1)
df2['乳酸异常'] = [1 if x>0 else 0 for x in df2['乳酸异常']]


# In[34]:


df2['脱氢酶异常'] = df[['羟丁酸脱氢酶偏高','乳酸脱氢酶偏高']].sum(axis=1)
df2['脱氢酶异常'] = [1 if x>0 else 0 for x in df2['脱氢酶异常']]


# In[35]:


df2['血沉偏高'] = df['血沉偏高']
df2['甘脯二肽氨基肽酶偏低'] = df['甘脯二肽氨基肽酶偏低']


# ### Medications

# In[38]:


df_drug = df[['左卡尼汀','钙离子通道阻滞剂','钙补充剂','维生素','骨化三醇','β受体阻滞剂',
              'ARB','抗血小板药物','质子泵抑制剂','复方α酮酸','氨基酸','尿激酶',
              '利多卡因','利尿剂','硝酸酯类药物','司维拉姆','人血白蛋白','多磺酸粘多糖霜','阿司匹林',
              '碳酸镧','多巴胺','ACEI','磷酸钠盐','α受体阻滞剂','米多君','丙泊酚','吗替麦考酚酯',
              '羟苯磺酸钙','生长抑素','鲑降钙素','多巴酚丁胺','硫酸鱼精蛋白','甲氧氯普胺','血凝酶',
              '腹膜透析液','高渗枸橼酸盐嘌呤','左甲状腺素钠','塞来昔布']]


# In[39]:


df_drug['其他抗高血压'] = df[['可乐定']].sum(axis=1)
df_drug['其他抗高血压'] = [1 if x>0 else 0 for x in df_drug['其他抗高血压']]


# In[40]:


df_drug['降血糖药'] = df[['胰岛素','格列奈类高血糖药','阿卡波糖']].sum(axis=1)
df_drug['降血糖药'] = [1 if x>0 else 0 for x in df_drug['降血糖药']]


# In[41]:


df_drug['调血脂药'] = df[['他汀类药物','非诺贝特']].sum(axis=1)
df_drug['调血脂药'] = [1 if x>0 else 0 for x in df_drug['调血脂药']]


# In[42]:


df_drug['抗凝血药'] = df[['抗凝药物','华法令钠']].sum(axis=1)
df_drug['抗凝血药'] = [1 if x>0 else 0 for x in df_drug['抗凝血药']]


# In[43]:


df_drug['电解质补充药物'] = df[['氯化钾','硫酸镁']].sum(axis=1)
df_drug['电解质补充药物'] = [1 if x>0 else 0 for x in df_drug['电解质补充药物']]


# In[44]:


df_drug['抗生素'] = df[['头孢','万古霉素','舒巴坦钠','沙星','西林','培南','氨基糖苷类抗生素','三唑类衍生物抗生素','大环内酯类抗生素','西司他丁钠']].sum(axis=1)
df_drug['抗生素'] = [1 if x>0 else 0 for x in df_drug['抗生素']]


# In[45]:


df_drug['非甾体抗炎药'] = df[['吲哚美辛']].sum(axis=1)
df_drug['非甾体抗炎药'] = [1 if x>0 else 0 for x in df_drug['非甾体抗炎药']]


# In[46]:


df_drug['肠胃药'] = df[['多潘立酮','莫沙必利','酪酸梭菌活菌','硫糖铝','铝碳酸镁']].sum(axis=1)
df_drug['肠胃药'] = [1 if x>0 else 0 for x in df_drug['肠胃药']]


# In[47]:


df_drug['精神类药物'] = df[['地西泮','丙戊酸钠','唑吡坦','苯二氮䓬类镇静药']].sum(axis=1)
df_drug['精神类药物'] = [1 if x>0 else 0 for x in df_drug['精神类药物']]


# In[48]:


df_drug['抗贫血药'] = df[['重组人促红素','叶酸','甲钴胺', '铁补充剂', '生血宁']].sum(axis=1)
df_drug['抗贫血药'] = [1 if x>0 else 0 for x in df_drug['抗贫血药']]


# In[49]:


df_drug['抗心绞痛药'] = df[['曲美他嗪']].sum(axis=1)
df_drug['抗心绞痛药'] = [1 if x>0 else 0 for x in df_drug['抗心绞痛药']]


# In[50]:


df_drug['止咳平喘药'] = df[['氨溴索', '愈创甘油醚', '强力枇杷露', '异丙托溴铵', '布地奈德','糖皮质激素']].sum(axis=1)
df_drug['止咳平喘药'] = [1 if x>0 else 0 for x in df_drug['止咳平喘药']]


# In[51]:


df_drug['营养剂'] = df[['肠内营养乳剂', '脂肪乳']].sum(axis=1)
df_drug['营养剂'] = [1 if x>0 else 0 for x in df_drug['营养剂']]


# In[52]:


df_drug['泻药'] = df[['乳果糖','聚乙二醇4000','开塞露']].sum(axis=1)
df_drug['泻药'] = [1 if x>0 else 0 for x in df_drug['泻药']]


# In[53]:


df_drug['止泻药'] = df[['蒙脱石']].sum(axis=1)
df_drug['止泻药'] = [1 if x>0 else 0 for x in df_drug['止泻药']]


# In[54]:


df_drug['抗感冒药'] = df[['酚麻美敏']].sum(axis=1)
df_drug['抗感冒药'] = [1 if x>0 else 0 for x in df_drug['抗感冒药']]


# In[55]:


df_drug['组胺受体阻断药'] = df[['抗组胺H1受体拮抗剂', '异丙嗪']].sum(axis=1)
df_drug['组胺受体阻断药'] = [1 if x>0 else 0 for x in df_drug['组胺受体阻断药']]


# In[56]:


df_drug['抗心衰药'] = df[['托拉塞米']].sum(axis=1)
df_drug['抗心衰药'] = [1 if x>0 else 0 for x in df_drug['抗心衰药']]


# In[57]:


df_drug['抗心律失常药'] = df[['胺碘酮']].sum(axis=1)
df_drug['抗心律失常药'] = [1 if x>0 else 0 for x in df_drug['抗心律失常药']]


# In[58]:


df_drug['镇痛药'] = df[['曲马多','去痛片']].sum(axis=1)
df_drug['镇痛药'] = [1 if x>0 else 0 for x in df_drug['镇痛药']]


# In[59]:


df_drug['抗病毒药'] = df[['核苷酸类逆转录酶抑制剂']].sum(axis=1)
df_drug['抗病毒药'] = [1 if x>0 else 0 for x in df_drug['抗病毒药']]


# In[60]:


df_drug['碘造影剂'] = df[['碘帕醇']].sum(axis=1)
df_drug['碘造影剂'] = [1 if x>0 else 0 for x in df_drug['碘造影剂']]


# In[61]:


df_drug['拟肾上腺素药'] = df[['去甲肾上腺素','间羟胺']].sum(axis=1)
df_drug['拟肾上腺素药'] = [1 if x>0 else 0 for x in df_drug['拟肾上腺素药']]


# In[63]:


df2 = pd.concat([df2, df_drug], axis=1)


# ### Others

# In[64]:


df['甲状旁腺移植术'] = df[['甲状旁腺移植术','移植术']].sum(axis=1)


# In[65]:


df_other = df[['胸部_X','腹部_X','心电图_ECG','肺部_CT','床边摄片_X','腹部_CT','磁共振扫描_MR','手_X','甲状旁腺显像_ECT','头颅_CT','胸部_CT','X线计算机体层特殊三维成像_CT','胃镜_ES','头颅_MR','腰椎_X','膝关节_X','骨盆_X','URO','冠状动脉CTA_CT','腕关节_X','上肢动脉CTA_CT','股骨_X','肠镜_ES','颈部_CT','肺部HR_CT','足_X','上肢静脉CTA_CT',
               '甲状旁腺切除术','甲状旁腺移植术','甲状腺切除术','前臂移植术','肾切除术']]


# In[66]:


examination_list = ['胸部_X','腹部_X','心电图_ECG','肺部_CT','床边摄片_X','腹部_CT','磁共振扫描_MR','手_X','甲状旁腺显像_ECT','头颅_CT','胸部_CT','X线计算机体层特殊三维成像_CT','胃镜_ES','头颅_MR','腰椎_X','膝关节_X','骨盆_X','URO','冠状动脉CTA_CT','腕关节_X','上肢动脉CTA_CT','股骨_X','肠镜_ES','颈部_CT','肺部HR_CT','足_X','上肢静脉CTA_CT']
surgery_list = ['甲状旁腺切除术','甲状旁腺移植术','甲状腺切除术','前臂移植术','肾切除术']


# In[67]:


df2 = pd.concat([df2, df_other], axis=1)


# In[65]:


df2.to_csv('zheyi_subtype_combined_data.csv', index=False, encoding='utf-8-sig')

