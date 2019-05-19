from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = pd.read_csv('data.csv')
data = df[df.columns[1]]
df[df.columns[2]] = pd.factorize(df[df.columns[2]])[0]


tf_idf_vectorizer = TfidfVectorizer(lowercase=False, max_features=2000)
tf_idf = tf_idf_vectorizer.fit_transform(data)
tf_idf_norm  = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()

#kmeans here
from KMeans import Kmeans

sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)
test_e = Kmeans(20, 1, 600)
fitted = test_e.fit_kmeans(Y_sklearn)
predicted_values = test_e.predict(Y_sklearn)

#
prefined_labels = df[df.columns[2]].values

from evaluation import purity_score, purity_score2
purity = purity_score2(predicted_values, prefined_labels)
print(purity)

# plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted_values, s=50, cmap='viridis')
#
# centers = fitted.centroids
# plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6)
#
# plt.show()