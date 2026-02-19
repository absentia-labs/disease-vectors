import os, sys
sys.path.append("../../../")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, scanpy as sc
import json
from tqdm import tqdm
from anndata import AnnData
from polygene.data_utils.tokenization import normalise_str

DATASET_PATH = '/media/rohola/ssd_storage/primary/'
SAVE_PATH = '/media/lleger/LaCie/mit/disease_geometry/cxg_lung_dataset/'

# hand pick cell types for comparison
cell_type_selection = {
            #"small cell lung carcinoma": ['epithelial cell', 'T cell'],
            #"lung large cell carcinoma": ['native cell', 'CD8-positive, alpha-beta T cell', 'CD4-positive, alpha-beta T cell'],
            #"lung adenocarcinoma": ['CD4-positive, alpha-beta T cell', 'CD8-positive, alpha-beta T cell', 'alveolar macrophage', 'malignant cell',],
            #"squamous cell lung carcinoma": ['malignant cell', 'CD4-positive, alpha-beta T cell', 'CD8-positive, alpha-beta T cell', 'alveolar macrophage'], 
            'normal': ['alveolar macrophage', 'epithelial cell of alveolus of lung', 'type II pneumocyte', 'CD4-positive, alpha-beta T cell', 'CD8-positive, alpha-beta T cell']
            }

os.makedirs(SAVE_PATH, exist_ok=True)
TEST_CHUNK_IDS = (2503, 3570)
dataset_files = [DATASET_PATH + f'cxg_chunk{i}.h5ad' for i in range(*TEST_CHUNK_IDS)]
number_of_cells = int(5e3)

disease_tissue = json.load(open("/media/lleger/LaCie/mit/disease_geometry/disease_tissues.json"))
top_diseases = json.load(open('/media/lleger/LaCie/mit/disease_geometry/diseases.json'))['diseases']

age_map = json.load(open('../../data_utils/vocab/age_map.json'))
is_adult = lambda row: "yes" if age_map.get(normalise_str(row['development_stage']), 40) // 10 >= 4 else "no"

for disease in cell_type_selection:    
    tissue = "lung"
    cell_types = cell_type_selection[disease]
    required_cells = {(disease, cell_type, tissue, 'yes'): number_of_cells for cell_type in cell_types}
    
    total_cells = 0
    anndata_slices = []
    progressbar = tqdm(np.random.permutation(dataset_files), desc=disease)
    included_phenotypes = ['disease', 'cell_type', 'tissue_general', 'adult']

    for file_path in progressbar:
        progressbar.set_description(f"{disease}: {total_cells} cells")
        loaded_chunk = sc.read_h5ad(file_path)
        loaded_chunk.obs['adult'] = loaded_chunk.obs.apply(is_adult, axis=1)

        for phenotype_combination in list(required_cells.keys()):
            mask = np.logical_and.reduce([ loaded_chunk.obs[phenotype] == value 
                for phenotype, value in zip(included_phenotypes, phenotype_combination)])

            if not mask.any(): continue
            filtered_chunk = loaded_chunk[mask][:required_cells[phenotype_combination]].copy()
            anndata_slice = AnnData(X=filtered_chunk.X.copy(), obs=filtered_chunk.obs.copy(), var=filtered_chunk.var.copy())
            anndata_slices.append(anndata_slice)
            total_cells += anndata_slice.n_obs
            required_cells[phenotype_combination] -= anndata_slice.n_obs

            if required_cells[phenotype_combination] <= 0: del required_cells[phenotype_combination]
            del filtered_chunk
        del loaded_chunk
        if not required_cells: break

    shard = sc.concat(anndata_slices)
    shard.obs.index.name = None
    shard.write(f"{SAVE_PATH + disease}_shard.h5ad")
    del shard