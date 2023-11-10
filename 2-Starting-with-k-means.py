"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="R15.arff"
#name="square1.arff"
#name="banana.arff"
#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()
k=15
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_
#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances

dists = euclidean_distances(centroids)
print(centroids)
print(dists)

# Calculer la distance entre chaque point et son centre
distancesClusterPoints = euclidean_distances(datanp,centroids)

for i in range(k):
    clusterPoints = distancesClusterPoints[labels == i ,i]
    min_distance = clusterPoints.min(axis=0)
    max_distance = clusterPoints.max(axis=0)
    average_distance = clusterPoints.mean(axis=0)

    print(f"Score de distance minimale pour le cluster: {i}", min_distance)
    print(f"Score de distance maximale pour le cluster:{i}", max_distance)
    print(f"Score de distance moyenne pour le cluster:{i}", average_distance)
    print("\n")

#Calcul de la distance entre les différents centroides
distanceClusterCentroid = euclidean_distances(centroids)
print(distanceClusterCentroid)

## Application itérative de la méthode k-means
inerties = []
for k in range(1,11):
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    iteration = model.n_iter_
    inertie = model.inertia_
    inerties.append(inertie)

plt.plot(range(1,11),inerties)
plt.title("Evolution de l'inertie en fonction du nombre de cluster")
plt.show()
##La meilleure solution de clustering 
# est celle avec l'inertie la plus grande (ici k=3) 
# nous avons déjà calculer les scores de regroupements

from sklearn.metrics import calinski_harabasz_score
score   = []
scoreMB = []
for k in range(2,21):
  tps1 = time.time()
  model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
  model_pred = model.fit_predict(datanp)
  score.append(calinski_harabasz_score(datanp,model_pred))
  tps2 = time.time()

  tps1MB = time.time()
  model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
  model_pred = model.fit_predict(datanp)
  scoreMB.append(calinski_harabasz_score(datanp,model_pred)) 
  tps2MB = time.time()

  plt.plot(score,'r')
  plt.plot(scoreMB,'b')
  plt.title(f"Evolution du score de CH entre kmeans et minibatch pour {k} clusters")
  labels = model.labels_
  # informations sur le clustering obtenu
  iteration = model.n_iter_
  #inertie = model.inertia_
  centroids = model.cluster_centers_
  #plt.figure(figsize=(6, 6))
  ##plt.scatter(f0, f1, c=labels, s=8)
  ##plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
  ##plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
  ##print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms","index Calinski-Harabasz :" , CHScore)
  plt.show()
chosenBadSamples = ["zelnik1.arff","banana.arff"]
ChosenGoodSamples = ["spherical_4_3.arff","triangle1.arff"]

## la méthode K-means pour le clustering ne fonctionne pas pour tous les cas,
## si figure un peu complexe => KO


##Différence entre batch et kmeans : rapidité, précision du score 