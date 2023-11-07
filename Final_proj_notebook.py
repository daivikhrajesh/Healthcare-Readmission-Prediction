#!/usr/bin/env python
# coding: utf-8

# <h2>GROUP - 6<h2>
#     <h3>Dataset : Diabetes 130-US hospitals for years 1999-2008 <h3>
# 

# <p>Daivikh Rajesh<br>
#     Jagadeesh M<br>
#     Venkata Reddy<p>

# In[1]:


from Library import *


# In[2]:


df = load_dataset('diabetic_data.csv')


# In[3]:


df.head()


# In[4]:


Preprocessing.describe(df)


# In[5]:


Preprocessing.missing(df)


# In[6]:


Preprocessing.drop(df)


# In[7]:


Analysis.patientinfo(df)


# In[8]:


Analysis.readmitted1(df)


# In[9]:


Analysis.convert_readmitted(df)


# In[10]:


Analysis.race_distribution(df)


# In[11]:


Analysis.gender_readmission(df)


# In[12]:


Analysis.age_distribution(df)


# In[13]:


Analysis.time_in_hospital(df)


# <b>The diag_1 is the Primary Diagnosis of the Patient, which means the 
# 
# *   patient is admitted to the hospital on this diagnosis.
# 
# 
# *   The diag_2 is the Secondary Diagnosis, According to CMS Documentation : Secondary diagnoses are conditions that coexist at the time of admission
# 
# 
# *   The diag_3 is the Additional Secondary Diagnosis.<b>

# In[14]:


Analysis.count_unique_diag(df)


# <b>From Columns 21 - 44, the following data is observed
#     *   “up” if the dosage was increased during the encounter
# 
# *   “down” if the dosage was decreased
# 
# 
# *   “steady” if the dosage did not change
# 
# *   “no” if the drug was not prescribed<b>
# 

# In[15]:


Analysis.other_cols(df)


# In[16]:


Analysis.plot_histograms(df)


# **In all of the features, Majority of the population is labeled as No. Means patients are not perscribed to take these medicines.**<br>
# <br>
# **So we can drop these columns with almost no information**

# In[17]:


Analysis.drop_columns(df)


# In[18]:


df.drop(columns=['medical_specialty'], inplace=True)


# In[19]:


df.head()


# In[20]:


conv = Conversion()


# In[21]:


df1 = conv.label_enc2(df)


# In[22]:


df1.head()


# <h2>Result :<h2>

# In[23]:


RFC.RTclassifier(df1)


# <b>This model can help hospitals and healthcare providers identify patients who are at high risk of readmission, allowing for early intervention and potentially improving patient outcomes.<b>
