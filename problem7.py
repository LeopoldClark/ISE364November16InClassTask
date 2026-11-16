# written answers are writtne in comments for their respective parts

from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target

# partA
# (a) How many observations and features are in the dataset:
print("Part A-a:")
print("Observations in dataset: ", X.shape[0])
print("Number of features in dataset: ", X.shape[1])
#=======================================================================#
#=======================================================================#
# (b) Display names of the four features:
print("\nPart A-b:")

features = data.feature_names
print("Features in dataset: ", features)
#=======================================================================#
#=======================================================================#
# (c) Calc mean and std of each feature:
print("\nPart A-c:")
import pandas as pd

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
print("Feature Mean:")
print(X.mean())
print("\nFeature Standard Deviation")
print(X.std())
#=======================================================================#
#=======================================================================#
# (d) Standardize Features 
print("Part A-d: ")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print("Standardized Feature Means: ")
print(X_scaled.mean(axis=0))
print("\nStandardized Feature Standard Deviations: ")
print(X_scaled.std(axis=0))
#=======================================================================#
#=======================================================================#
# (e) Why is standardization important before applying K-means and PCA?
# Standardization is important before applying K-means and PCA becuase 
# features that have a very large variation between their samples will 
# drown out the smaller ones which could skew the analsyis of the data
# K-means uses distance so that larger the distance relative to scale
# the more those will dominate the break down of the data
# PCA uses direction, so features with larger variance can dorown out 
# principal components.
#=======================================================================#
#=======================================================================#
#=======================================================================#
#=======================================================================#
# PART B:
from sklearn.cluster import KMeans
# (a) Crfeate a table containing K and WCSS
print("Part B-a:")

kValues = []
wcssValues = []
k = 1
while k <= 8:

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia = kmeans.inertia_
    kValues.append(k)
    wcssValues.append(inertia)
    k += 1

comparison = pd.DataFrame({"K": kValues, "WCSS": wcssValues})
print(comparison)
#=======================================================================#
#=======================================================================#
# (b) Plot WCSS vs K
print("Part B-b:")
import matplotlib.pyplot as plt

plt.plot(kValues, wcssValues, marker="o")
plt.xlabel("Number of Clusters(K)")
plt.ylabel("WCSS")
plt.xticks(kValues)
plt.show()
#=======================================================================#
#=======================================================================#
# (c) Based on the elbow method, what value of K appears reasonable:
print("Part B-c:")
# K = 3 is the more resonable value according to the elbow method
print("K = 3 is the best value according to the elbow method")
#=======================================================================#
#=======================================================================#
# (d) Why should the elbow method be viewed as a heuristic rather than
# an exact rule
# The elbow method should be viewed as a heuristic because it identifies
# only where the curve begins to saturate. Some datasets don't have a clear
# elbow or there are several. 
#=======================================================================#
#=======================================================================#
#=======================================================================#
#=======================================================================#
# PART C:
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
# (a) report the number of observations assigned to each cluster
print("Part C-a:")
prediction = kmeans.predict(X_scaled)
K1 = 0
K2 = 0
K3 = 0
k = 0
while k < len(prediction):

    if (prediction[k] == 0):
        K1 += 1
    elif (prediction[k] == 1):
        K2 += 1
    elif (prediction[k] == 2):
        K3 += 1

    k += 1

print("Observations in cluster 1: ", K1)
print("Observations in cluster 2: ", K2)
print("Observations in cluster 3: ", K3)
#=======================================================================#
#=======================================================================#
# (b) report the cluster centroids in the standardized feature space
print("Part C-b:")
centroids = kmeans.cluster_centers_
print(centroids)
#=======================================================================#
#=======================================================================#
# (c) compare cluster assignment to true iris class labels
print("Part C-c: ")
comparison = pd.crosstab(
    prediction,
    y,
    rownames=["Cluster"],
    colnames=["True Iris Class"]
)

print(comparison)
#=======================================================================#
#=======================================================================#
# (d) report the cluster centroids in the standardized feature space
# The cluster labels 0, 1, and 2 don't necessarily correspond to the ture 
# class labels 0, 1, 2. K-means groups the samples with thei features 
# without using true labels. The numbers 0, 1, and 2 identify clusters. 
# The crosstab displays the way the cluster is correspondant to the actual 
# species.
#=======================================================================#
#=======================================================================#
#=======================================================================#
#=======================================================================#
# Part D:
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
# (a) report the explained variance ratio for each principla component
print("Part D-a")
varRaitios = pca.explained_variance_ratio_
k = 0
while k < len(varRaitios):
    print(k + 1,": ", varRaitios[k])
    k += 1
#=======================================================================#
#=======================================================================#
# (b) calc cumulative explained variance
print("Part D-b")
cumVar = []
totVar = 0

k = 0
while k < len(varRaitios):
    totVar += varRaitios[k]
    cumVar.append(totVar)

    print("PC1 through PC", k+1, ": ", totVar)
    k += 1

#=======================================================================#
#=======================================================================#
# (c) how many principal components are required to explain at least .9 
# of tot variance:
print("Part D-c")
comp9 = 1
while cumVar[comp9 - 1] < .9:
    comp9 += 1

print("Components required for at least 90% variasnce:", comp9)
#=======================================================================#
#=======================================================================#
# (d) how many principal components are required to explain at least .95 
# of tot var
print("Part D-d")
comp95 = 1
while cumVar[comp95 - 1] < .95:
    comp95 += 1
print("Components required for at least 95% variasnce:", comp95)
#=======================================================================#
#=======================================================================#
# (e) Why the number of principal components can't exceed numbner of 
# original features:
# The number of principal components can't exceed the number of original 
# features becuase each principal component is a linear combination of the 
# oringal features. SO the directions are essentailly orthogonal. This data 
# set of 4 features can only have at most 4 principal components.
#=======================================================================#
#=======================================================================#
#=======================================================================#
#=======================================================================#
# PART E:
X_pca_2d = X_pca[:, :2]
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_pca = kmeans_pca.fit_predict(X_pca_2d)
#=======================================================================#
#=======================================================================#
# (a) compare the clustering results from K-means on all standardized 
# features with K means on first two principal components:
print("Part E-a: ")
compOg = pd.crosstab(
    prediction,
    y,
    rownames=["Original Cluster"],
    colnames=["True Iris Class"]
)
print(compOg, "\n")

compPCA = pd.crosstab(
    clusters_pca,
    y, 
    rownames=["PCA Cluster"],
    colnames=["True Iris Class"]
)
print(compPCA, "\n")

compClusters = pd.crosstab(
    prediction,
    clusters_pca, 
    rownames=["Original Cluster"],
    colnames=["PCA Cluster"]
)
print(compClusters, "\n")
#=======================================================================#
#=======================================================================#
# (b) does pca necessarily imporve kmeans clustering?
# PCA does not necessarily improve kmeans clustering becuase it retains
# the directions of teh greatest variance which does not necessarily 
# make for the best cluster separation.m
#=======================================================================#
#=======================================================================#
# (c) What circumstances might applying PCA before clustering be helpful:
# The case where there are mahy correlated features and a lot of 
# strucutre retained by a small number of principle components.
#=======================================================================#
#=======================================================================#
# (d) What circumstances could PCA reduce clustering performance:
# PCA can reduce clustering performance when there are dumped components 
# that have information or metadata concerning how clusters are seperated. 
# When there are few components retained, that can cause groups to overlap.

print("Program End")