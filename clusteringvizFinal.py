# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:40:12 2021

@author: natha
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 14:12:51 2021

@author: IsaacN
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np


# Data Preparation
df = pd.read_csv('artist_danceability_valence1.CSV')

#KMeans
kmeans = KMeans(n_clusters=4, random_state=0)#cluster=4, 
#random_state: Determines random number generation for centroid initialization. 
#Use an int to make the randomness deterministic
#predict which data belongs to which cluster
df['cluster'] = kmeans.fit_predict(df[['Danceability', 'Valence']])


#get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]

#add centroids to df
df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2], 3:cen_x[3]})
df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_y[3]})

# define map colors
colors = ['#DF2020', '#81DF20', '#2095DF', 'y']
df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3]})

#plotting
fig, ax = plt.subplots(1, figsize=(8,8))
# plot data
plt.scatter(df.Danceability, df.Valence, c=df.c, alpha=0.6, s=10)
# plot center
plt.scatter(cen_x, cen_y, marker="^", c=colors, s=70)

#draw enclosure
for i in df.cluster.unique():
    points = df[df.cluster == i][['Danceability', 'Valence']].values
    #get convex hull
    hull = ConvexHull(points)
    # get x and y coordinates
    # repeat last point to close the polygon
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    # plot shape
    plt.fill(x_hull, y_hull, alpha=0.3, c=colors[i])

plt.xlim(0,1)
plt.ylim(0,1)

#legend elements
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
               markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]

#show legend elements
plt.legend(handles=legend_elements, loc='upper right')
    

#titles and labels
plt.title('Artists features \n', loc='left', fontsize=22)
plt.xlabel('Danceability')
plt.ylabel('Valence')

# #export dataframe to excel to see
# df.to_excel("output.xlsx")

#to see the sample of each cluster
cluster1 = df[kmeans.labels_==0]
cluster2 = df[kmeans.labels_==1]
cluster3 = df[kmeans.labels_==2]
cluster4 = df[kmeans.labels_==3]

#export the sample of each cluster to excel file
cluster1.to_excel("cluster1.xlsx")
cluster2.to_excel("cluster2.xlsx")
cluster3.to_excel("cluster3.xlsx")
cluster4.to_excel("cluster4.xlsx")