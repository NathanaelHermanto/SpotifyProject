# -*- coding: utf-8 -*-
"""
Elbow Method for K-Means


@author: IsaacN
"""

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd


df = pd.read_csv('artist_danceability_valence1.CSV')
print(df)

model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,10), timings= True)
visualizer.fit(df[['Danceability', 'Valence']])        # Fit data to visualizer
visualizer.show()        # Finalize and render figure
