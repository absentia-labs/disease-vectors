import numpy as np, pandas as pd
from ripser import ripser
import sys
import umap
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
sys.path.append('../../../')
from polygene.analysis.attributions.attributions import AttributionAnalysis
from persim import bottleneck
from sklearn.metrics import adjusted_mutual_info_score
from joblib import Parallel, delayed

class EndotypeAnalysis():
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.endotypes = None

    def get_endotypes(self, points, cells, confounders, stability_threshold=0.9, number_of_bootstraps=100, size_of_bootstraps=1000, genes=True, qsize=99, qlength=90):

        distance_matrix = squareform(pdist(points, metric="euclidean"))
        outlier_mask = np.median(distance_matrix, axis=1) < np.percentile(np.median(distance_matrix, axis=1), 95)
        
        filtered_points = points[outlier_mask]
        filtered_distance_matrix = distance_matrix[np.ix_(outlier_mask, outlier_mask)]

        results, components, persistent_entropy, colless_imbalance = self.zeroth_persistent_homology(filtered_distance_matrix)

        deaths, sizes = results[:, 1], results[:, 2]
        endotype_mask = (deaths > np.percentile(deaths, q=qlength)) & (sizes > np.percentile(sizes, q=qsize))

        number_of_endotypes = endotype_mask.sum()
        print("Number of endotypes", number_of_endotypes)
        if number_of_endotypes == 1:
            filtration_distance, filtration_size = filtered_distance_matrix.max(), len(filtered_points)
        elif number_of_endotypes > 1:
            filtration_distance = results[:, 1][endotype_mask].min()
            filtration_size = results[:, 2][endotype_mask].min()

        adjacency = {i: set(np.where(filtered_distance_matrix[i] < filtration_distance)[0]) - {i} for i in range(len(filtered_points))}
        visited, components_final = set(), []
        for v in adjacency:
            if v not in visited:
                stack, component_indices = [v], []
                while stack:
                    u = stack.pop()
                    if u not in visited:
                        visited.add(u)
                        component_indices.append(u)
                        stack.extend(adjacency[u] - visited)
                components_final.append(component_indices)
        components_final = [c for c in components_final if len(c) >= filtration_size]
        print("Endotypes Corrected", len(components_final))
        print("Component sizes", [len(c) for c in components_final])

        stability_scores = self.cluster_stability(filtered_points, filtered_distance_matrix, components_final,
                                                   number_of_bootstraps=number_of_bootstraps, size_of_boostrap=size_of_bootstraps, qsize=qsize, qlength=qlength)
        print("Endotype stability scores", stability_scores)

        stability_mask = stability_scores >= stability_threshold
        components_final = [set(c) for i, c in enumerate(components_final) if stability_mask[i]]
        print("Final Components and sizes", len(components_final), [len(c) for c in components_final])
        stability_scores = stability_scores[stability_mask]

        
        corrected_endotype_mask = np.array([
            any(cf.issubset(comp) for cf in components_final)
            for comp in components
        ])
        
        print("corrected_endotype_mask", corrected_endotype_mask.sum())
        
        top_genes_per_component = []
        if genes: 
            for endotype in tqdm(components_final, desc="Endotype top genes"):
                self.tokenizer.bypass_inference = True
                analyzer = AttributionAnalysis(
                    self.model,
                    self.tokenizer,
                    data=cells[outlier_mask][list(endotype)[:100]],
                    biotype_json="../../data_utils/vocab/gene_biotypes.json",
                    ensembl_json="../../data_utils/vocab/ensembl_to_gene.json",
                )
                attributed_genes = analyzer.gradients(only_protein_encoding=True, disable_pbar=True).sum(axis=0).sort_values(ascending=False).index.tolist()[:2]
                top_genes_per_component.append(attributed_genes)

        ripser_result = ripser(filtered_points, maxdim=1)["dgms"]

        projector = umap.UMAP(spread=2, min_dist=1.5, n_neighbors=50, random_state=3, n_jobs=1)
        points_2d = projector.fit_transform(filtered_points)

        endotype_labels = -np.ones(len(filtered_points), dtype=int)
        for i, component in enumerate(components_final): endotype_labels[list(component)] = i

        confounder_results = {}
        for column_name in confounders.columns:
            column_values = confounders[outlier_mask].reset_index(drop=True)[column_name].values
            confounder_labels = column_values[endotype_labels != -1]
            ami, p_value = self.mutual_information_with_permutation_test(
                endotype_labels[endotype_labels != -1],
                confounder_labels
            )
            confounder_results[column_name] = {"ami": ami, "p_value": p_value}

        self.endotypes = {
            "outlier_mask": outlier_mask,
            "endotype_mask": endotype_mask,
            "results": results,
            "corrected_endotype_mask": corrected_endotype_mask,
            "distance_matrix": distance_matrix,
            "components": components_final,
            "stability_scores": stability_scores,
            "stability_threshold": stability_threshold,
            "filtration_distance": filtration_distance,
            "filtration_size": filtration_size,
            "persistent_entropy": persistent_entropy,
            "colless_imbalance": colless_imbalance,
            "bootstrap_entropy": self.mean_bootstrap_entropy,
            "bootstrap_colless": self.mean_bootstrap_colless,

            "top_genes_per_component": top_genes_per_component,
            "ripser_result": ripser_result,
            "umap_projection": points_2d,
            "points": points,
            "confounder_results": confounder_results,
        }
        return self.endotypes

    def zeroth_persistent_homology(self, distance_matrix):

        number_of_points = distance_matrix.shape[0]
        parent = np.arange(number_of_points)
        component_size = np.ones(number_of_points, dtype=int)
        components = [{i} for i in range(number_of_points)]
        merge_size_differences = []

        results = np.zeros((number_of_points, 3))
        results[:, 0], results[:, 1], results[:, 2] = 0.0, np.inf, 1.0

        def find_root(vertex):
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex

        edges = [(distance_matrix[i, j], i, j) for i in range(number_of_points) for j in range(i + 1, number_of_points)]
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
                components[root_i] |= components[root_j]

        root = find_root(0)
        results[root, 2] = component_size[root]

        def persistent_entropy_(bars):
            bars = bars[np.isfinite(bars[:, 1])]
            lengths = bars[:,1] - bars[:,0]
            lengths = lengths[lengths > 0]
            p = lengths / lengths.sum()
            return (-np.sum(p*np.log(p))) / np.log(len(lengths))
        persistent_entropy = persistent_entropy_(results)

        colless_imbalance = float(sum(merge_size_differences) / ((number_of_points - 1) * (number_of_points - 2))) if number_of_points > 2 else 0.0
        return results, components, persistent_entropy, colless_imbalance

    def mutual_information_with_permutation_test(self, endotype_labels_valid, confounder_labels_valid, number_of_permutations=1000):
        observed_ami = adjusted_mutual_info_score(endotype_labels_valid, confounder_labels_valid)
        permutation_distribution = [
            adjusted_mutual_info_score(endotype_labels_valid, np.random.permutation(confounder_labels_valid))
            for _ in range(number_of_permutations)
        ]
        p_value = (np.sum(np.array(permutation_distribution) >= observed_ami) + 1) / (number_of_permutations + 1)
        return observed_ami, p_value

    def cluster_stability(self, filtered_points, filtered_distance_matrix, components_final, number_of_bootstraps=100, size_of_boostrap=1000,
                          qsize=99, qlength=90):
        number_of_points = len(filtered_points)
        original_components = [set(component) for component in components_final]
        number_of_components = len(original_components)
        stability_scores = np.zeros(number_of_components, dtype=float)
        entropy_values = []
        colless_values = []

        sampling_fraction = size_of_boostrap / number_of_points
        for _ in tqdm(range(number_of_bootstraps), desc="Bootstrap stability"):
            bootstrap_indices = np.random.choice(number_of_points, size=size_of_boostrap, replace=False)
            submatrix = filtered_distance_matrix[np.ix_(bootstrap_indices, bootstrap_indices)]

            sub_results, _, subentropy, subcolless = self.zeroth_persistent_homology(submatrix)
            entropy_values.append(subentropy)
            colless_values.append(subcolless)
            deaths_sub = sub_results[:, 1]; sizes_sub = sub_results[:, 2]
            sub_endotype_mask = (deaths_sub > np.percentile(deaths_sub, q=qlength)) & (sizes_sub > np.percentile(sizes_sub, q=qsize))
            number_of_endotypes_sub = sub_endotype_mask.sum()

            if number_of_endotypes_sub == 1:
                filtration_distance_sub, filtration_size_sub = submatrix.max(), submatrix.shape[0]
            else:
                filtration_distance_sub = sub_results[:, 1][sub_endotype_mask].min()
                filtration_size_sub = sub_results[:, 2][sub_endotype_mask].min()

            adjacency_sub = {i: set(np.where(submatrix[i] < filtration_distance_sub)[0]) - {i} for i in range(submatrix.shape[0])}
            visited_sub, bootstrap_components_local = set(), []
            for v in adjacency_sub:
                if v not in visited_sub:
                    stack = [v]
                    component_indices_local = []
                    while stack:
                        u = stack.pop()
                        if u not in visited_sub:
                            visited_sub.add(u)
                            component_indices_local.append(u)
                            stack.extend(adjacency_sub[u] - visited_sub)
                    if len(component_indices_local) >= filtration_size_sub:
                        bootstrap_components_local.append(set(component_indices_local))

            bootstrap_components = [set(bootstrap_indices[list(component_local)]) for component_local in bootstrap_components_local]
            
            for k, original_component in enumerate(original_components):
                best_score = 0.0
                original_size = len(original_component)

                for bootstrap_component in bootstrap_components:
                    intersection_size = len(original_component & bootstrap_component)

                    recall = intersection_size / original_size
                    precision = intersection_size / len(bootstrap_component) if len(bootstrap_component) > 0 else 0.0

                    if precision + recall > 0: score = 2 * precision * recall / (precision + recall)
                    else: score = 0.0
                    if score > best_score: best_score = score
    
                stability_scores[k] += best_score


        if number_of_bootstraps > 0:
            stability_scores /= float(number_of_bootstraps)
        
        self.mean_bootstrap_entropy = float(np.mean(entropy_values))
        self.mean_bootstrap_colless = float(np.mean(colless_values))
        return stability_scores
    
if __name__ == "__main__":
    import os
    import scanpy as sc
    SAVED_DATA = "/media/lleger/LaCie/mit/disease_geometry/endotypes/"
    print(os.listdir(SAVED_DATA))
    from polygene.model.model import load_trained_model
    from sklearn.decomposition import PCA
    model, tokenizer = load_trained_model("../../../saved_models/gesam_polygene_run_4/", checkpoint_n=-1)

    subsample = 10000
    size_of_boostraps = 1000
    number_of_bootstraps = 100
    stability_threshold = 0.1 # depends on sampling frac etc. 
    qlength=85
    qsize=99
    def process_file(file):
        disease = file.split('_')[0]
        print('Disease:', disease, '\n\n')
        embedding_pickle = pd.read_pickle(SAVED_DATA + file)
        cell_data = sc.read_h5ad(SAVED_DATA + disease + "_cells.h5ad")
        predicted_confounders =  pd.DataFrame(embedding_pickle[1],columns = tokenizer.phenotypic_types)
        confounders = pd.DataFrame(embedding_pickle[2],columns = tokenizer.phenotypic_types)
        accuracies = (predicted_confounders.values == confounders.values).mean(axis=0)
        manifold = embedding_pickle[0]

        linear_projector = PCA(manifold.shape[1])
        cell_expressions_reduced = linear_projector.fit_transform(cell_data[:subsample].X.toarray())

        endotype_analyzer = EndotypeAnalysis(model, tokenizer)
        res1 = endotype_analyzer.get_endotypes(manifold[:subsample],
                                            cell_data[:subsample],
                                            confounders[:subsample],
                                            stability_threshold=stability_threshold,
                                            number_of_bootstraps=number_of_bootstraps,
                                            size_of_bootstraps=size_of_boostraps,
                                            qlength=qlength,
                                            qsize=qsize)

        res2 = endotype_analyzer.get_endotypes(cell_expressions_reduced,
                                            cell_data[:subsample],
                                            confounders[:subsample],
                                            stability_threshold=stability_threshold,
                                            number_of_bootstraps=number_of_bootstraps,
                                            size_of_bootstraps=size_of_boostraps,
                                            genes=False,
                                            qlength=qlength,
                                            qsize=qsize)

        return (disease, res1, res2)


    results_list = Parallel(n_jobs=30)(
        delayed(process_file)(file) for file in os.listdir(SAVED_DATA) if "embedding" in file
    )

    endotype_results = {}
    for r in results_list:
        if r is None: continue
        disease, res1, res2 = r
        endotype_results[disease] = res1
        endotype_results[disease + "_raw"] = res2
    pd.to_pickle(endotype_results, "/media/lleger/LaCie/mit/disease_geometry/endotyping_results.pkl")