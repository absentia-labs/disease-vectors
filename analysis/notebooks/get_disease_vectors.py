# Compute centroid based disease vectors and for all the populations dont compute every combination just take a subset. 

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

cell_count = 1000
cell_count_vectors = 100
results = {}
for disease in tqdm(diseases, desc="Disease Vectors"):
    if disease == 'normal': continue
    cells = sc.read_h5ad(EMBEDDINGS_DIR + disease + "_cells.h5ad")
    embeddings = pd.read_pickle(EMBEDDINGS_DIR + disease + "_embeddings.pkl")
    df = pd.DataFrame(
        {'embedding': embeddings[0].tolist(),
        'y_pred': embeddings[1].tolist(),
        'cell': cells.X.toarray().tolist(),
        'obs_name': cells.obs_names.tolist()}
        | {phenotypic_types[idx]: embeddings[2][:, idx] for idx in range(len(phenotypic_types))}
    )
    df[['disease','cell_type']] = df[['disease','cell_type']].applymap(lambda x: x[1:-1])
    df = df.sample(cell_count, random_state=3)
    disease = normalise_str(disease)[1:-1]
    for cell_type in df['cell_type'].unique():
        cell_group = df[df['cell_type'] == cell_type]
        if "normal" not in cell_group['disease'].tolist() or disease not in cell_group['disease'].tolist(): continue
        normal_cells = cell_group[cell_group['disease'] == 'normal'][:cell_count_vectors]
        disease_cells = cell_group[cell_group['disease'] == disease][:cell_count_vectors]
        disease_embeddings = np.array(disease_cells['embedding'].tolist())
        normal_embeddings = np.array(normal_cells['embedding'].tolist())
        disease_raw_cells = np.array(disease_cells['cell'].tolist())
        normal_raw_cells = np.array(normal_cells['cell'].tolist())
        min_length = min(len(disease_embeddings), len(normal_embeddings))
        disease_embeddings = disease_embeddings[:min_length]
        normal_embeddings = normal_embeddings[:min_length]
        disease_raw_cells = disease_raw_cells[:min_length]
        normal_raw_cells = normal_raw_cells[:min_length]
        disease_obs_names = disease_cells['obs_name'].iloc[:min_length].tolist()
        normal_obs_names = normal_cells['obs_name'].iloc[:min_length].tolist()
        results[f"{disease} {cell_type}"] = {
            "centroid": disease_embeddings.mean(0) - normal_embeddings.mean(0),
            "raw_centroid": disease_raw_cells.mean(0) - normal_raw_cells.mean(0),
            "vectors": (disease_embeddings - normal_embeddings).tolist(),
            "raw_vectors": (disease_raw_cells - normal_raw_cells).tolist(),
            "anndata_obs_pairs": list(zip(disease_obs_names, normal_obs_names)),
            "counts": len(disease_embeddings),
        }
pd.to_pickle(results, EMBEDDINGS_DIR + "disease_vector_results.pkl")
from sklearn.metrics.pairwise import cosine_similarity
dv = pd.DataFrame(results)
func = lambda a, b, v: cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0] if 'centroid' in v else cosine_similarity(np.array(a), np.array(b)).mean()
similarity_matrices = {v: pd.Series({(a, b): func(dv.loc[v, a], dv.loc[v, b], v) for a in dv.columns for b in dv.columns}).unstack() for v in ['centroid', 'raw_centroid', 'vectors', 'raw_vectors']}
pd.to_pickle(similarity_matrices, EMBEDDINGS_DIR + "disease_vector_similarity_matrices.pkl")
