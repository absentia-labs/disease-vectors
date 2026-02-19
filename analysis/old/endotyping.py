import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from ripser import ripser
import sys
sys.path.append('../../')
from polygene.analysis.attributions_old import AttributionAnalysis
from tqdm import tqdm
import umap
class EndotypeAnalysis():
    def __init__(self, model=None, tokenizer=None, null_path=None, alpha=0.05):

        self.model = model
        self.tokenizer=tokenizer
        self.endotypes=None
        self.null_distribution = pd.read_pickle(null_path) if null_path is not None else None

    def get_endotypes(self,cells, points, labels, significance=0.05, percentile=95, method="significance"):

        endotypes = {}
        progress_bar = tqdm(np.unique(labels))
        for label in progress_bar:
            progress_bar.set_description(f"Endotyping {label}")
            label_mask = labels == label
            cells_label = cells[label_mask][:2000]
            
            # Winsorize outliers
            disease_points = points[label_mask][:2000]
            distance_matrix = squareform(pdist(disease_points, metric="euclidean"))
            distance_matrix = np.minimum(distance_matrix, np.percentile(distance_matrix[distance_matrix > 0], percentile))

            # Persistent homology
            results, persistent_entropy, colless_imbalance = self.zeroth_persistent_homology(distance_matrix, d=points.shape[1])
            # results is shape (n, 3), birth, death, size@death

            if method == "significance":
                p_comb, significant_mask = self.fisher_significance(distance_matrix, results[:, 1], results[:,2], alpha=significance)
            else:
                deaths, sizes = results[:, 1], results[:, 2]
                p_comb, significant_mask = ((deaths > np.percentile(deaths, q=90)) & (sizes > np.percentile(sizes, q=99)) for _ in range(2))

            n_endotypes = significant_mask.sum()
            progress_bar.set_description(f"Endotyping {label}: Found {n_endotypes}, Entropy = {persistent_entropy}")
            if n_endotypes == 1: epsilon, size = distance_matrix.max(), distance_matrix.shape[0]
            else: epsilon, size = results[:,1][significant_mask].min(), results[:,2][significant_mask].min()

            adjacency = {i: set(np.where(distance_matrix[i] < epsilon)[0]) - {i} for i in range(len(disease_points))}
            visited, components = set(), []
            for v in adjacency:
                if v not in visited:
                    stack, comp = [v], []
                    while stack:
                        u = stack.pop()
                        if u not in visited:
                            visited.add(u)
                            comp.append(u)
                            stack.extend(adjacency[u] - visited)
                    components.append(comp)
            components = [c for c in components if len(c) >= size]

            gene_tags = []
            for endotype in components:
                self.tokenizer.bypass_inference = True
                attr_analyzer = AttributionAnalysis(self.model, self.tokenizer, data=cells_label[endotype], biotype_json= "../data_utils/vocab/gene_biotypes.json",
                                                   ensembl_json="../data_utils/vocab/ensembl_to_gene.json" )
                attributed_genes = attr_analyzer.gradients(only_protein_encoding=True, disable_tqdm=True).sum(axis=0).sort_values(ascending=False).index.tolist()[:2]
                gene_tags.append( "/".join(attributed_genes) )

            # Best for higher dimensional homology groups and classic plots
            ripser_result = ripser(disease_points, maxdim=1)["dgms"]
            
            # umap for plots 

            projector = umap.UMAP(spread=1, min_dist=1, n_neighbors=25, random_state=3)
            disease_points_2d = projector.fit_transform(disease_points)
            endotypes[label] = {"n_endotypes": n_endotypes, "entropy": persistent_entropy, "colless": colless_imbalance, "h0": results[:, 1],
                                 "h0_significance": significant_mask, "components": components, "gene_tags": gene_tags, "ripser": ripser_result, "umap": disease_points_2d.tolist(),
                                 "points": disease_points}
        self.endotypes = endotypes

    def zeroth_persistent_homology(self, distance_matrix, d=1):

        number_of_points = distance_matrix.shape[0]
        
        parent = np.arange(number_of_points)
        component_size = np.ones(number_of_points, dtype=int)
        results = np.zeros((number_of_points, 3))
        results[:,0], results[:,1], results[:,2] = 0.0, np.inf, 1.0
        merge_size_differences = []
        def find_root(vertex):
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex
        edges = [(distance_matrix[i, j], i, j) for i in range(number_of_points) for j in range(i+1, number_of_points)]
        edges.sort()
        for distance_value, i, j in edges:
            root_i, root_j = find_root(i), find_root(j)
            if root_i != root_j:
                merge_size_differences.append(abs(component_size[root_i] - component_size[root_j]))
                if component_size[root_i] < component_size[root_j]:
                    root_i, root_j = root_j, root_i
                results[root_j, 1] = distance_value
                results[root_j, 2] = component_size[root_j]
                parent[root_j] = root_i
                component_size[root_i] += component_size[root_j]
        root = find_root(0)
        results[root, 2] = component_size[root]
        
        # persistent entropy
        finite = results[:,1] < np.inf
        probabilities = (results[finite,1]**np.sqrt(d)) / (results[finite,1]**np.sqrt(d)).sum()
        probabilities = np.clip(probabilities, 1e-12, 1)
        persistent_entropy = float((-np.sum(probabilities * np.log(probabilities))) / np.log(len(results[finite,1])))

        #colless imbalance
        colless_imbalance = float(sum(merge_size_differences) / ((number_of_points - 1) * (number_of_points - 2))) if number_of_points > 2 else 0.0
        return results, persistent_entropy, colless_imbalance
    
    def fisher_significance(self, distance_matrix, deaths_obs, sizes_obs, alpha=0.05):
        deaths_unit = np.clip(deaths_obs / distance_matrix.max(), 0.0, 1.0)
        null_deaths = self.null_distribution["deaths_unit"][:int(1e6)]
        null_sizes = self.null_distribution["sizes"][:int(1e6)] * (distance_matrix.shape[0] / 1000)

        sorted_deaths = np.sort(null_deaths)
        sorted_sizes  = np.sort(null_sizes)

        idx_deaths = np.searchsorted(sorted_deaths, deaths_unit, side='left')
        idx_sizes  = np.searchsorted(sorted_sizes, sizes_obs, side='left')

        p_death = (1 + (len(null_deaths) - idx_deaths)) / (1 + len(null_deaths))
        p_size  = (1 + (len(null_sizes) - idx_sizes)) / (1 + len(null_sizes))

        x_stat = -2 * (np.log(p_death) + np.log(p_size))
        p_comb = 1 - chi2.cdf(x_stat, 4)
        return p_comb, p_comb < alpha
    
import os
import pandas as pd, numpy as np
from polygene.model.model import load_trained_model

EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/'
SAVE = False
model, tokenizer = load_trained_model('../../runs/gesam_polygene_run_4/', checkpoint_n=-1)

df_list = []
for file in [EMBEDDINGS_DIR + f for f in os.listdir(EMBEDDINGS_DIR) if 'embeddings_part' in f]:
    embeddings = pd.read_pickle(file)
    df_list.append(pd.DataFrame.from_dict({"embeddings": embeddings[0].tolist(), 'disease':embeddings[2][:,tokenizer.phenotypic_types.index('disease')],
                                           'disease_pred': embeddings[1][:,tokenizer.phenotypic_types.index('disease')]}))
embedding_df = pd.concat(df_list)

import scanpy as sc
anndata_list = []
for file in [EMBEDDINGS_DIR + f for f in os.listdir(EMBEDDINGS_DIR) if 'cells_part' in f]:
    anndata = sc.read_h5ad(file)
    anndata.obs.index.name = None
    anndata_list.append(anndata)
cells = sc.concat(anndata_list)

e = EndotypeAnalysis(model, tokenizer, null_path=EMBEDDINGS_DIR + "endotyping_null_distribution.pkl")
e.get_endotypes(cells, points=np.array(embedding_df['embeddings'].tolist()), labels=embedding_df['disease'].values, significance=1e-7, method="thresh")

pd.to_pickle(e.endotypes, EMBEDDINGS_DIR + "endotype_results.pkl")