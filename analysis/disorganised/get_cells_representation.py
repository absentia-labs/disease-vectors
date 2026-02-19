import sys
sys.path.append("../../")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, scanpy as sc
import json
from tqdm import tqdm
from anndata import AnnData, concat
from polygene.data_utils.tokenization import normalise_str

OUTPUT_DIR = '/media/lleger/LaCie/mit/disease_geometry/dataset/'
DATASET = '/media/rohola/ssd_storage/primary/'
TEST_CHUNKS = (2503, 3574)
CELLS_PER_TYPE = 100
SHARD_SIZE = 10000

#age_map = json.load(open('../data_utils/vocab/age_map.json'))
#disease_tissue = json.load(open("/media/lleger/LaCie/mit/disease_geometry/disease_tissues.json"))
#cell_types_per_disease = json.load(open("/media/lleger/LaCie/mit/disease_geometry/cell_types_per_disease.json"))
frequent_diseases = json.load(open('/media/lleger/LaCie/mit/disease_geometry/diseases.json'))['diseases']

#def get_adult(row):
#    value = age_map.get(normalise_str(row['development_stage']), 40)
#    if pd.isna(value): print('Error', row['development_stage'])
#    return "yes" if value // 10 >= 4 else "no"

def save_shards(anndata_slices, output_prefix):
    if not anndata_slices:
        return 0
    combined = concat(anndata_slices)
    total_cells = combined.n_obs
    number_of_shards = (total_cells + SHARD_SIZE - 1) // SHARD_SIZE
    for shard_index in range(number_of_shards):
        start_index = shard_index * SHARD_SIZE
        end_index = min((shard_index + 1) * SHARD_SIZE, total_cells)
        shard = combined[start_index:end_index].copy()
        shard.obs.index.name = None
        shard.write(f"{output_prefix}_shard{shard_index}.h5ad")
        del shard
    del combined
    return total_cells

def collect_cells(required_cells, description):
    included_phenotypes = ['disease',]
    anndata_slices = []
    total_cells = 0
    
    chunk_files = [DATASET + f'cxg_chunk{i}.h5ad' for i in range(TEST_CHUNKS[0], TEST_CHUNKS[1] + 1)]
    progressbar = tqdm(chunk_files, desc=description)
    
    for file_path in progressbar:
        progressbar.set_description(f"{description}: {total_cells} cells")
        loaded_chunk = sc.read_h5ad(file_path)
        #loaded_chunk.obs['adult'] = loaded_chunk.obs.apply(get_adult, axis=1)
        
        for phenotype_combination in list(required_cells.keys()):
            mask = np.logical_and.reduce([
                loaded_chunk.obs[phenotype] == value 
                for phenotype, value in zip(included_phenotypes, phenotype_combination)
            ])
            if not mask.any():
                continue
            filtered_chunk = loaded_chunk[mask][:required_cells[phenotype_combination]].copy()
            anndata_slice = AnnData(X=filtered_chunk.X.copy(), obs=filtered_chunk.obs.copy(), var=filtered_chunk.var.copy())
            anndata_slices.append(anndata_slice)
            total_cells += anndata_slice.n_obs
            required_cells[phenotype_combination] -= anndata_slice.n_obs
            if required_cells[phenotype_combination] <= 0:
                del required_cells[phenotype_combination]
            del filtered_chunk
        del loaded_chunk
        
        if not required_cells:
            break
    
    return anndata_slices



required_cells = {(disease,): CELLS_PER_TYPE for disease in frequent_diseases}
anndata_slices = collect_cells(required_cells, f"Collecting representation test set")
saved_cells = save_shards(anndata_slices, OUTPUT_DIR + "representation_test")
print(f"Saved {saved_cells} cells")
del anndata_slices
