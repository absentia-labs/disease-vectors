import os, sys
sys.path.append("../../")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, scanpy as sc
import json
from tqdm import tqdm
from anndata import AnnData
from polygene.data_utils.tokenization import normalise_str
from collections import defaultdict

DATASET_PATH = '/media/rohola/ssd_storage/primary/'
SAVE_PATH = '/media/lleger/LaCie/mit/disease_geometry/dataset/'
os.makedirs(SAVE_PATH, exist_ok=True)
test_shard_idx = (2503, 3503)
dataset_files = [DATASET_PATH + f'cxg_chunk{i}.h5ad' for i in range(*test_shard_idx)]

top_diseases = json.load(open('/media/lleger/LaCie/mit/disease_geometry/diseases.json'))['diseases']
cell_types_per_disease = json.load(open("/media/lleger/LaCie/mit/disease_geometry/disease_cell_type.json"))

age_map = json.load(open('../data_utils/vocab/age_map.json'))
is_adult = lambda development_stage: age_map.get(normalise_str(development_stage)[1:-1], 40) >= 30


for idx in np.arange(0, len(top_diseases), 5):
    required_cells = {(disease, cell_type, tissue, True): int(2e3) if disease !="normal" else int(1e3) for disease in top_diseases[idx:idx+5]
                    for cell_type, tissue in cell_types_per_disease[disease]}

    total_cells = 0
    anndata_slices = defaultdict(list)
    progressbar = tqdm(dataset_files, desc="cells")
    included_phenotypes = ['disease', 'cell_type', 'tissue_general', 'adult']

    for file_path in progressbar:
        progressbar.set_description(f"{total_cells} cells")
        cxg_shard = sc.read_h5ad(file_path)
        cxg_shard.obs['adult'] = cxg_shard.obs['development_stage'].apply(is_adult)
        
        for phenotype_combination in list(required_cells.keys()):
            mask = np.logical_and.reduce([cxg_shard.obs[phenotype] == value for phenotype, value in zip(included_phenotypes, phenotype_combination)])
            if not mask.any():
                continue

            filtered_chunk = cxg_shard[mask][:required_cells[phenotype_combination]].copy()
            anndata_slice = AnnData(X=filtered_chunk.X.copy(), obs=filtered_chunk.obs.copy(), var=filtered_chunk.var.copy())
            
            anndata_slices[phenotype_combination[0]].append(anndata_slice)
            total_cells += anndata_slice.n_obs
            required_cells[phenotype_combination] -= anndata_slice.n_obs
            
            if required_cells[phenotype_combination] <= 0: del required_cells[phenotype_combination]
            del filtered_chunk
        del cxg_shard
        if not required_cells: break

    for disease in anndata_slices: #some diseases are 
        if not anndata_slices[disease]: continue
        shard = sc.concat(anndata_slices[disease])
        shard.obs.index.name = None
        shard.write(f"{SAVE_PATH + disease}_shard.h5ad")
        del shard