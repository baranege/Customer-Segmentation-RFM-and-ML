## Customer Segmentation with Machine Learning
#libraries and settings
import datetime as dt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

## Task 1: analysing and processing dataset
#1 read excel btw 2010-2011 and copy
df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df  = df_.copy()

#2 Review of descriptive statistics
df.info()
df.head()
df.describe()
df.shape

#3 check NA
df.isnull().any()
df.isnull().sum()

#4 Drop NAs
df.dropna(inplace=True)

#5 Number of total unique goods
df["StockCode"].nunique()

#6 Amount of each unique good in inventory
df["StockCode"].value_counts()

#df.groupby("StockCode")["Description"].value_counts()

#7 Sort by amount of order
df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(20)

#8 Exclusion of cancelled("C") transactions
df = df[~df["Invoice"].str.contains("C", na=False)]


#9 Revenue per transaction
df["TotalRevenue"] = df["Quantity"]*df["Price"]

# Task 2: Calculation of RFM metrics

# Recency: How recently a customer has made a purchase
# Frequency: How often a customer makes a purchase
# Monetary Value: How much money a customer spends on purchases
# Source: Investopedia

today_date = dt.datetime(2011, 12, 11)

# Creating RFM dataframe
rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days, #recency
                                     "Invoice": lambda num: num.nunique(), #frequency
                                     "TotalRevenue": lambda TotalRevenue: TotalRevenue.sum()}) #monetary

## Naming RFM metrics
rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()
rfm = rfm[rfm["monetary"] > 0]

# Task 3: Calculating RFM scores

#recency score
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels = [5,4,3,2,1])

#frequency score
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels = [1,2,3,4,5])
###rank(method="first") -> first: ranks assigned in order they appear in the array

#monetary score
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels = [1,2,3,4,5])

#creating RFM scores
rfm['RFM_SCORE'] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str))

# Task 4: Identification of segments with RFM scores
seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm.reset_index(inplace=True)

# # Observing segments and scores
# rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
# rfm[["Customer ID" ,"segment"]].groupby("segment").agg(["mean", "count"])
# rfm[["Customer ID" ,"segment"]].groupby("segment").agg(["count"])


# Task 5 Modelling with K-Means

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Creating new dataframe for K-Means
rfm.columns
rfm_kmeans = rfm.set_index("Customer ID") # Moving Customer ID to index column
rfm_kmeans = rfm_kmeans[["recency", "frequency", "monetary"]]
rfm_kmeans.head()

# Scaling RFM dataframe
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(rfm_kmeans) # fitting rfm_kmeans then transform the values (original values are not scaled)
df[0:5]

# Determining Optimum Cluster Numbers
# kmeans = KMeans(n_clusters=3)
# k_fit = kmeans.fit(df)
# k_fit
#
# k_fit.n_clusters # The number of clusters to form as well as
# # the number of centroids to generate.
#
#
#
# k_fit.cluster_centers_ #Coordinates of cluster centers. If the algorithm stops before fully converging,
# # these will not be consistent with labels_.
#
#
# k_fit.labels_ #Labels of each point
# k_fit.inertia_ #Sum of squared distances of samples to their closest cluster center.
# df[0:5]


# Determining the number of clusters
k_means = KMeans(n_clusters=10).fit(df) # There are 10 segments for customers
clusters = k_means.labels_
type(df) # requires to chang into dataframe
df = pd.DataFrame(df)
df.columns = ["recency", "frequency", "monetary"]

df.shape

# Visualization of Clusters
plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=clusters,
            s=50,
            cmap="viridis")
plt.show()

# Determining Centres
centres = k_means.cluster_centers_

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=clusters,
            s=50,
            cmap="viridis")

plt.scatter(centres[:, 0],
            centres[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

# Selecting Optimum Number of Clusters with SSD

# kmeans = KMeans()
# ssd = [] # sum of squared distances
# K = range(2,10)
#
# for k in K:
#     kmeans = KMeans(n_clusters=k).fit(df)
#     ssd.append(kmeans.inertia_)
#
# ssd
#
# plt.plot(K, ssd, "bx-")
# plt.xlabel("'sum_of_squared_distances")
# plt.title("elbow method for optimal k")
# plt.show()
# https://towardsdatascience.com/clustering-with-k-means-1e07a8bfb7ca

# Automtic method to choose the optimal number of clusters K:(Elbow Method)
#
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=10) # looking for optimum number of clusters btw 2 and 40
elbow.fit(df)
elbow.show()

# Optimal number of clusters
elbow.elbow_value_

# Creating The Final Clusters with Elbow method
kmeans_elbow = KMeans(n_clusters=elbow.elbow_value_).fit(df)

clusters_elbow = kmeans_elbow.labels_

# Arranging the RFM dataframe
pd.DataFrame({"Customer_ID": rfm.index, "Clusters": clusters, "Clusters_Elbow": clusters_elbow})

rfm["cluster_no"], rfm["cluster_no_elbow"] = [clusters,  clusters_elbow]

rfm.head()
rfm.shape

# Correcting the cluster no
rfm["cluster_no"], rfm["cluster_no_elbow"] = rfm["cluster_no"] + 1, rfm["cluster_no_elbow"] +1

# checking the number of customers in each cluster
rfm.groupby("cluster_no").agg({"cluster_no": "count"})

# Checking the clusters by RFM metrics
rfm.groupby(["cluster_no", "cluster_no_elbow"]).agg(np.mean)

rfm[rfm["cluster_no"] == 4]

# Observing and comparing segments(based on RFM scores) with clusters done by K-Means

rfm.groupby(["cluster_no", "cluster_no_elbow", "segment"])["segment"].count()

agg_rfm = rfm.groupby(["cluster_no", "cluster_no_elbow", "segment", "RFM_SCORE"])["segment"].count()

rfm.groupby("segment")["cluster_no_elbow"].value_counts()

rfm.groupby("segment")["cluster_no"].value_counts()



#rfm.groupby("cluster_no")["frequency"].mean().sort_values(ascending=False)















########################################################################################


#Task 6 Interpretation of results and policy recommendations

# Grouping segments by means of RFM scores
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#Best Customers
rfm[rfm["segment"].str.contains("champions")].head()

#High-spending New Customers
rfm[rfm["segment"].str.contains("potential_loyalists")].head()

#Low-Spending Active Loyal Customers
rfm[rfm["segment"].str.contains("need_attention")].head()

##########################################################################################
## Additional Observations
rfm.groupby("segment")["frequency"].mean().sort_values(ascending=False)

agg_rfm = rfm[["recency", "frequency",
               "monetary", "RFM_SCORE",
               "segment", "cluster_no"]].groupby(["segment", "cluster_no"])["recency", "frequency", "monetary"].mean()

agg_rfm = agg_rfm.reset_index()
agg_rfm.sort_values(by="recency", ascending=False)

agg_rfm.sort_values(by=["recency", "frequency", "monetary"],
                    ascending=[False, False, True])

agg_rfm.sort_values(by=["recency", "frequency", "monetary"],
                    ascending = [True, False, False])
########################################################################################## Useless
agg_rfm = rfm[["recency", "frequency", "monetary", "segment",
               "cluster_no", "cluster_no_elbow"]].groupby(["segment", "cluster_no", "cluster_no_elbow"])["recency", "frequency", "monetary"].mean()

agg_rfm.reset_index(inplace=True)
agg_rfm.head()
metrics = ["recency", "frequency", "monetary"]
cluster_nos = ["cluster_no", "cluster_no_elbow"]
agg_rfm.sort_values(by = metrics, ascending=False)


