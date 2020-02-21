#!/usr/bin/env python
# coding: utf-8

# In[282]:


from pandas import read_excel, merge
from numpy import arange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[ ]:


#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
import pylab as plt
import os


# In[ ]:





# In[113]:


# obtain location of current working directory
import os
os.getcwd()


# In[114]:


# read in the excel (xlsx) file
df = pd.read_excel('./Downloads/THD Store Attributes 1.xlsx')


# In[115]:


# Set the print option to display ALL rows and columns 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[116]:


df.head()


# In[117]:


df.info(verbose = True)


# In[118]:


# replacing the NAN values to the correct category label of No or Non Coastal
df.fillna({'Hispanic Store':'No',                                                   
'African American Store': 'No',                                                                                      
'Lake Store?': 'No',
'Mountain Store?':'No',                                                  
'College Store?': 'No',                                                   
'Military Store?': 'No',
'Coastal Store': 'No',
'Coastal Classification': 'Non Coastal'}, inplace = True)


# converting data to appropriate type of category
fac_cats = ['Hispanic Store',
'African American Store',
'Coastal Store',
'Coastal Classification',
'Lake Store?',
'Mountain Store?',
'College Store?',
'Military Store?',
'Upscale / Core / Value?', 
'Urbanicity']
for col in fac_cats:
    df[col] = df[col].astype('category')
df['Store'] = df['Store'].astype(object)
df.dtypes


# In[119]:


# checking the percentage of missing values in each variable
df.isnull().sum()/len(df)*100


# In[120]:


# relabeling the factor Y to Yes for specified column features
str_cols = ['Lake Store?', 'Mountain Store?', 'College Store?', 'Military Store?']
df[str_cols] = df[str_cols].replace('Y','Yes', regex = True)
df.head()


# In[121]:


# frequency of particular cont for Competitor L Store informs if the future imputation is providing reasonable replacement for missingness
df['Competitor L Store Count'].value_counts()/len(df['Competitor L Store Count'])*100


# In[122]:


# Missing data visualization module for Python.
import missingno as msno
# checking for pattern of missingness MCAR, MAR, NMAR to identify the procedure for handling the missing data 
missing = df.columns[df.isnull().any()].tolist()
msno.matrix(df[missing])


# In[145]:


# filling missing values with existing ones about 15% missing originally for one feature
df.interpolate(method = 'pad', inplace = True)
df.isnull().sum()


# In[146]:


df2 = pd.read_excel('./Downloads/THD Stores Attributes 2.xlsx')
df2.head()


# In[124]:


df2.info()


# In[332]:


# inner join of the two THD datasets
# the revenue information from df2 provides a continuous target, SALES, for imputating of missing data in df1
# Note: using SQ_FT as the target provided similar results

# joining/merging df1 and df2
df2['Store'] = df2['STORE']
df_merge = pd.merge(df, df2, on = 'Store')
df_merge = df_merge.drop(['STORE'], axis = 1)
df_merge.head()


# In[333]:


df_merge.to_csv(r'C:\Users\14047\Desktop\THD_Clean.csv')


# In[334]:


# calculate the variance of all the numeric variables
numeric = df_merge.select_dtypes(exclude =['category','object'])
numeric.var()

# this code gives the list of variables that have a variance greater than 1.5e+07 
var = numeric.var()
numeric = numeric.columns
variable = [ ]
for i in range(0,len(var)):
    if var[i]>=1.5e+07:   #setting the threshold 
       variable.append(numeric[i+1])


# In[335]:


# first feature reduction reduces numeric variables by 23%
df_merge[variable].shape
numeric


# In[336]:


# applying the high correlation filter on the remaining numeric features

# create correlation matrix
corr_matrix = df_merge[variable].corr().abs()

# select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k =1).astype(np.bool))

# find index of feature columns with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# the remaining numeric features after the variance and correlation filters
corr_red = df_merge[variable].drop(df_merge[variable][to_drop], axis =1).columns

# the variance and correlation filters resulted in a dimensionality reduction leaving 21% of the original numeric features
corr_var_red = df_merge[corr_red]
df_merge[variable].var()


# In[337]:


# 2nd approach to dimensioality reduction using the random forest algorithm
from sklearn.ensemble import RandomForestRegressor
data=df_merge.iloc[1:,1:].drop(['SALES'], axis=1)
model = RandomForestRegressor(random_state=1, max_depth=10)
data=pd.get_dummies(data)
model.fit(data,df_merge.iloc[1:,1:].SALES)


# In[338]:


# graph indicating the features that explain the majority of the variation in THD SALES by random forest method
#  all features (numeric and categorical) reduced by approximately 93%
features = df_merge.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-9:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[340]:


# third approach to dimensionality reduction: principal component analysis 
df_PCA = df_merge.select_dtypes(exclude =['category','object']).iloc[1:,:] # dropping the first column, store 9999 and the non-numeric featrues
X =df_PCA.drop(['SALES'], axis = 1)
Y = df_PCA.SALES # sales represents the dependent variable
pca_data = preprocessing.StandardScaler().fit(X).transform(X) # standardizing the numeric feature data
pca = PCA(n_components=4)
pca_result = pca.fit_transform(df_PCA.values)
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
labels = ['PC' + str(i) for i in range(1, len(per_var)+1)]
plt.bar(x = range(1,len(per_var)+1),height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


# In[158]:


plt.plot(range(4), pca.explained_variance_ratio_)
plt.plot(range(4), np.cumsum(pca.explained_variance_ratio_))
plt.title("Component-wise and Cumulative Explained Variance")
plt.ylabel('Percentage of variance explained')
plt.xlabel('Number of Principal Components')
plt.show()


# In[159]:


# fourth approach to dimensionality reduction

# although the model is of poor quality it is being used to compare the amount of change 
# in variation explained to compare PCA and SVD feature reduction methods
reg = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state = 4)
reg.fit(X_train, Y_train)

# variation explained by regressing using all features
reg.score(X_test, Y_test)


# In[162]:


pca1 = PCA(n_components = 25, whiten = True)
x = pca1.fit(X).transform(X)
pca1.explained_variance_

reg = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test= train_test_split(x, Y, test_size=0.2, random_state = 4)
reg.fit(X_train, Y_train)

# variation explained by regressing using 25 PCA components
reg.score(X_test, Y_test)


# In[163]:


svd = TruncatedSVD(n_components =25 )
x = svd.fit(X).transform(X)
reg = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test= train_test_split(x, Y, test_size=0.2, random_state = 4)
reg.fit(X_train, Y_train)

# variation explained by regressing using 25 SVD components
# PCA slightly better than SVD at lower dimensions
# dimensions lower than 25 yield worse overall explained variance
reg.score(X_test, Y_test)


# In[280]:


features
indices


# In[274]:


df_merge.columns = df_merge.columns.to_series().apply(lambda x: x.strip())
clean = df_merge[df_merge.columns]
clean.iloc[[],[ 83,  31,  32, 114,  96,  18,  71,  85, 113]].columns


# In[291]:


data = clean[['Store','Pop(Age 18+), Hisp/Lat', 'HHs, Hisp/Lat, % Chg', 'HHs, Asian',
       'HUs, Built 2000 -2009', 'Med HUs in Structure (exc mobile, other)',
       'Competitor W Store Count', 'Pop (>=5): Speak Indo-European',
       'Pop 16+: Armed Forces', 'HUs, Built 2010 - 2013', 'SALES','Avg HH Networth', 'Population']]
data


# In[292]:


cols = data.columns[1:]
cols


# In[293]:


data.set_index('Store',inplace=True)


# In[294]:


data.head()


# In[303]:



data=data.fillna(data.mean())
data.head()


# In[311]:


dataNorm = ((data-data.min())/(data.max()-data.min()))*20
    
dataNorm


# In[ ]:





# In[312]:


cluster = KMeans(n_clusters = 12)


# In[316]:


cols = dataNorm.columns[1:]
cols
dataNorm["cluster"]=cluster.fit_predict(dataNorm[dataNorm.columns[2:]])
dataNorm.tail()


# In[318]:


# principal component separation to create a 2-dimensional picture
pca = PCA(n_components = 2)
dataNorm['x'] = pca.fit_transform(dataNorm[cols])[:,0]
dataNorm['y'] = pca.fit_transform(dataNorm[cols])[:,1]
dataNorm = dataNorm.reset_index()


# In[319]:


dataNorm.tail()


# In[322]:


patient_clusters = dataNorm[["Store", "cluster", "x", "y"]]


# In[323]:


patient_clusters.tail()


# In[324]:


final = merge(df_response, patient_clusters)
final = merge(df_campaign, final);


# In[326]:


import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()


# In[ ]:




