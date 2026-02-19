import numpy as np, pandas as pd

dataset_predictions = pd.read_pickle('/media/lleger/LaCie/mit/disease_geometry/dataset_predictions.pkl')
dataset_predictions.keys()
from umap import UMAP
from sklearn.decomposition import PCA

X = np.concatenate(dataset_predictions['hidden_states'])
pca = PCA(n_components=15,)
umap = UMAP(n_neighbors=10, min_dist=0.95, spread=1)
pca = PCA(n_components=10,)
umap = UMAP(n_neighbors=25, min_dist=1.75, spread=2,)# densmap=True)

spread=2
min_dist=1.5
n_neighbors=15
n_pca_components = 10
print(spread, min_dist, n_neighbors, n_pca_components)
pca = PCA(n_components=n_pca_components,)
umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=spread, init="random")#, densmap=False)
X_transf = umap.fit_transform(pca.fit_transform(X))
np.save("X_transf.npy", X_transf)