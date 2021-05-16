# Importing libraries

import pandas as pd
import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go

# Pulling invoice data
df_retail = pd.read_csv(r'C:\Users\Prashant\Documents\Machine Learing Workspace\Online-Retail-Clustering\.gitignore\OnlineRetail.csv')

# ## Null value treatment
df_retail.info()

print('Number of na in customer id =' ,df_retail['CustomerID'].isna().sum())
print('Number of null in customer id =' ,df_retail['CustomerID'].isnull().sum())

df_retail['CustomerID'].fillna(0,inplace=True)
df_retail['Description'].fillna('No description',inplace=True)
df_retail.info()

### Implementing K means clustering 
k=5
k_means = (df_retail.loc[:,['Quantity','UnitPrice','CustomerID']].sample(k, replace=False))
k_means2 = pd.DataFrame()                    
clusters = pd.DataFrame()

data=df_retail.loc[:,['Quantity','UnitPrice','CustomerID']]
data.head()

while not k_means2.equals(k_means):
    # distance matrix (euclidean distance)
    cluster_count = 0
    for idx, k_mean in k_means.iterrows():
        clusters[cluster_count] = (data[k_means.columns] - np.array(k_mean)).pow(2).sum(1).pow(0.5)
        cluster_count += 1

    # update cluster
    data['MDCluster'] = clusters.idxmin(axis=1)

    # store previous cluster
    k_means2 = k_means
    k_means = pd.DataFrame()
    k_means_frame = data.groupby('MDCluster').agg(np.mean)

    k_means[k_means_frame.columns] = k_means_frame[k_means_frame.columns]


# plotting
data_graph = [go.Scatter(
              x=data['Quantity'],
              y=data['UnitPrice'].where(data['MDCluster'] == c),
              mode='markers',
              name='Cluster: ' + str(c)
              ) for c in range(k)]

data_graph.append(
    go.Scatter(
        x=k_means['Quantity'],
        y=k_means['UnitPrice'],
        mode='markers',
        marker=dict(
            size=10,
            color='#000000',
        ),
        name='Centroids of Clusters'
    )
)

plt.plot(data_graph, filename='cluster.html')






