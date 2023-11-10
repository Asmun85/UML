import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestCentroid


###################################################################
# Exemple : Agglomerative Clustering


path = './artificial/'
name="2d-4c-no4.arff"

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



### FIXER la distance
# 
tps1 = time.time()
seuil_dist=10
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


###
# FIXER le nombre de clusters
###
k=4
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


#######################################################################

temps = []
scores = []
for d in range(2,30):
  print("----------------------------------------------------------------")
  tps1 = time.time()
  model = cluster.AgglomerativeClustering(linkage='single', n_clusters=d)
  model = model.fit(datanp)
  tps2 = time.time()
  temps.append(round((tps2 - tps1)*1000,2))
  labels = model.labels_
  silhouetteScore = metrics.silhouette_score(datanp,labels)
  scores.append(silhouetteScore)
  

  #plt.scatter(f0, f1, c=labels, s=8)
  #plt.title("Clustering agglomératif (average, distance_treshold= "+str(d)+") "+str(name))
  #plt.show()
  print("nb clusters =",model.n_clusters_,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms",",Silhouette Score : " ,{silhouetteScore})

plt.title("Evolution of silhouette score")
plt.xlabel("number of cluster")
plt.ylabel("Silhouette Score")
plt.plot(range(2,30),scores,'g-o')

plt.show()

linkage = ['ward','average','complete','single']
tempsWard=[]
tempsAverage =[]
tempsComplete = []
tempsSingle = []

scoreDBWard = []
scoreDBAvg = []
scoreDBComp = []
scoreDBSingle = []

##Add the score for each method
##For each method of linkage, depending on the metric used computional time
for k in range(2,50):
  #Calcul temps pour linkage ward
  tps1 = time.time()
  model = cluster.AgglomerativeClustering( linkage='ward', n_clusters=k)
  model = model.fit(datanp)
  tps2 = time.time()
  tempsWard.append(round((tps2 - tps1)*1000,2))
  scoreDBWard.append(metrics.silhouette_score(datanp,model.labels_))
  
  #Calcul temps pour linkage average
  tps1 = time.time()
  model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
  model = model.fit(datanp)
  tps2 = time.time()
  tempsAverage.append(round((tps2 - tps1)*1000,2))
  scoreDBAvg.append(metrics.silhouette_score(datanp,model.labels_))


  #Calcul du temps linkage Complete
  tps1 = time.time()
  model = cluster.AgglomerativeClustering(linkage='complete', n_clusters=k)
  model = model.fit(datanp)
  tps2 = time.time()
  tempsComplete.append(round((tps2 - tps1)*1000,2))
  scoreDBComp.append(metrics.silhouette_score(datanp,model.labels_))


  #Calcul temps linkage single
  tps1 = time.time()
  model = cluster.AgglomerativeClustering(linkage='single', n_clusters=k)
  model = model.fit(datanp)
  tps2 = time.time()
  tempsSingle.append(round((tps2 - tps1)*1000,2))
  scoreDBSingle.append(metrics.silhouette_score(datanp,model.labels_))

# fig , (ax1,ax2) = plt.subplots(1,2)

plt.title("Evolution of computing time depending on number of clusters")
plt.plot(range(2,50),tempsWard,'r',label='Ward')
plt.plot(range(2,50),tempsAverage,'b',label='Average')
plt.plot(range(2,50),tempsComplete,'g',label='Complete')
plt.plot(range(2,50),tempsSingle,'k',label='Single')
plt.legend()

# ax2.set_title("Evolution of the Score depending on number of clusters")
# ax2.plot(range(2,50),scoreDBWard,'r',label='Ward')
# ax2.plot(range(2,50),scoreDBAvg,'b',label='Average')
# ax2.plot(range(2,50),scoreDBComp,'g',label='Complete')
# ax2.plot(range(2,50),scoreDBSingle,'k',label='Single')
# ax2.legend()
plt.show()
