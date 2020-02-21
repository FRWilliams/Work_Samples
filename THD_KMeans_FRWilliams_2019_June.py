# Import required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

import os
os.getcwd() 

# Read in data and inspect the first 5 records
data = pd.read_csv('./Desktop/THD_Clean.csv')
data.head()

data.info(verbose = True)

# Split of categorical and continuous features
quantitative  = data.select_dtypes(exclude = ['object'])
continuous_features = quantitative.columns 
qualitative = data.select_dtypes(include =['object'])
categorical_features = qualitative.columns

# Descriptive statistics 
quantitative.describe()

# Correlation heat map 
quantitative.corr().style.background_gradient(cmap = 'hot')

# Convert the categorical features to binary dummy variables
for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
data.head()

data.shape

# To give equal importance to all features, we need to scale the continuous features
data.fillna(data.mean(), inplace = True)
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
	
Sum_of_squared_distances


# Elbow plot showing tha any K between three and six is optimal
plt.style.use('fivethirtyeight')
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# Create column indices for the PCA table
cols = data.columns[1:]
cols

# Set the index to the store number
data.set_index('Store', inplace = True)
data.head()

# KMeans Clustering
cluster = KMeans(n_clusters = 6)

cols = data.columns[1:]
cols
data["cluster"] = cluster.fit_predict(data[data.columns[2:]])
data.tail()

# Principal component separation to create a 2-dimensional picture
data =data*10
pca = PCA(n_components = 2)
data['X'] = pca.fit_transform(data[cols])[:,0]
data['Y']= pca.fit_transform(data[cols])[:,1]
data = data.reset_index()
data.tail()

store_clusters = data[['Store', 'cluster','X','Y']]
store_clusters.tail()


trace0 = go.Scatter(x = store_clusters[stores_clusters.cluster == 0]["x"],
                    y = store_clusters[stores_clusters.cluster == 0]["y"],
                    name = "Cluster 1",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rbga(15,152, 0.5)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))

trace1 = go.Scatter(x = store_clusters[stores_clusters.cluster == 1]["x"],
                    y = store_clusters[stores_clusters.cluster == 1]["y"],
                    name = "Cluster 2",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rbga(180,18,180, 0.5)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))
trace2 = go.Scatter(x = store_clusters[stores_clusters.cluster == 2]["x"],
                    y = store_clusters[stores_clusters.cluster == 2]["y"],
                    name = "Cluster 3",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rbga(132,132,132 0.8)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))
trace3 = go.Scatter(x = store_clusters[stores_clusters.cluster == 3]["x"],
                    y = store_clusters[stores_clusters.cluster == 3]["y"],
                    name = "Cluster 4",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rbga(122,122,12, 0.8)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))
trace4 = go.Scatter(x = store_clusters[stores_clusters.cluster == 4]["x"],
                    y = store_clusters[stores_clusters.cluster == 4]["y"],
                    name = "Cluster 5",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rbga(230,20,30, 0.5)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))
trace5 = go.Scatter(x = store_clusters[stores_clusters.cluster == 5]["x"],
                    y = store_clusters[stores_clusters.cluster == 5]["y"],
                    name = "Cluster 6",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rbga(116, 0, 83, 0.8)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))