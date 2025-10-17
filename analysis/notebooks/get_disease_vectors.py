# Compute centroid based disease vectors and for all the populations dont compute every combination just take a subset. 

import argparse
from sklearn.decomposition import PCA
import scanpy as sc, pandas as pd, numpy as np
import itertools
import sys
from tqdm import tqdm
sys.path.append('../../../')
from polygene.model.model import load_trained_model
from polygene.data_utils.tokenization import normalise_str
m, tok = load_trained_model("../../../runs/gesam_polygene_run_4/")
phenotypic_types = tok.phenotypic_types

DISEASES_DICT = {'respiratory': ['COVID-19','influenza','lung adenocarcinoma'],
                 'neurological': ['Alzheimer disease','Parkinson disease','glioblastoma'],
                 'cardiometabolic': ['myocardial infarction','dilated cardiomyopathy','arrhythmogenic right ventricular cardiomyopathy']}
EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/vector_data/'

diseases = sum(list(DISEASES_DICT.values()), [])
def compute_disease_vectors(cell_count=1000, raw_vectors = False, force_min_length=100):
    print(raw_vectors)
    results = {}
    pbar = tqdm(diseases,)
    for disease in pbar:
        pbar.set_description(f'Computing Vectors {disease}')
        if disease == 'normal': continue
        cells = sc.read_h5ad(EMBEDDINGS_DIR + disease + "_cells.h5ad")
        cells.obs_names_make_unique()
        embeddings = pd.read_pickle(EMBEDDINGS_DIR + disease + "_embeddings.pkl")
        df = pd.DataFrame(
            {'embedding': embeddings[0].tolist(),
            'y_pred': embeddings[1].tolist(),
            'obs_name': cells.obs_names.tolist()}
            | {phenotypic_types[idx]: embeddings[2][:, idx] for idx in range(len(phenotypic_types))}
        )
        if raw_vectors: df['cell'] =  cells.X.toarray().tolist()

        df[['disease','cell_type']] = df[['disease','cell_type']].applymap(lambda x: x[1:-1])
        df = df.sample(min(cell_count, len(df)), random_state=3)
        disease = normalise_str(disease)[1:-1]
       
        for cell_type in df['cell_type'].unique():
            cell_group = df[df['cell_type'] == cell_type]
            if "normal" not in cell_group['disease'].tolist() or disease not in cell_group['disease'].tolist(): continue
            normal_cells, disease_cells = cell_group[cell_group['disease'] == 'normal'], cell_group[cell_group['disease'] == disease]
            min_length = min(len(normal_cells), len(disease_cells)) if force_min_length is None else min(min(len(normal_cells), len(disease_cells)), force_min_length)
            
            normal_embeddings, disease_embeddings = np.array(normal_cells['embedding'].tolist())[:min_length], np.array(disease_cells['embedding'].tolist())[:min_length]
            
            results[f"{disease} {cell_type}"] = {
                "centroid": disease_embeddings.mean(0) - normal_embeddings.mean(0),
                "vectors": (disease_embeddings - normal_embeddings).tolist(),
                "anndata_obs_pairs": list(zip(normal_cells['obs_name'].iloc[:min_length].tolist(), disease_cells['obs_name'].iloc[:min_length].tolist())),
                "counts": len(disease_embeddings),
            }
            if raw_vectors:
                normal_raw_cells, disease_raw_cells = np.array(normal_cells['cell'].tolist())[:min_length], np.array(disease_cells['cell'].tolist())[:min_length]
                results[f"{disease} {cell_type}"].update({
                "raw_centroid": disease_raw_cells.mean(0) - normal_raw_cells.mean(0),
                "raw_vectors": (disease_raw_cells - normal_raw_cells).tolist()
                })
    return results
if __name__ == "__main__":
    results = compute_disease_vectors(raw_vectors=True)
    pd.to_pickle(results, EMBEDDINGS_DIR + f"disease_vector_results.pkl")
    from sklearn.metrics.pairwise import cosine_similarity
    dv = pd.DataFrame(results)
    func = lambda a, b, v: cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0] if 'centroid' in v else cosine_similarity(np.array(a), np.array(b)).mean()
    similarity_matrices = {v: pd.Series({(a, b): func(dv.loc[v, a], dv.loc[v, b], v) for a in dv.columns for b in dv.columns}).unstack() for v in ['centroid', 'raw_centroid', 'vectors', 'raw_vectors']}
    pd.to_pickle(similarity_matrices, EMBEDDINGS_DIR + "disease_vector_similarity_matrices_.pkl")
