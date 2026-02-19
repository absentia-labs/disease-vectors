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
CELLS_PER_TYPE = 5000
SHARD_SIZE = 10000

age_map = json.load(open('../data_utils/vocab/age_map.json'))
disease_tissue = json.load(open("/media/lleger/LaCie/mit/disease_geometry/disease_tissues.json"))
cell_types_per_disease = json.load(open("/media/lleger/LaCie/mit/disease_geometry/cell_types_per_disease.json"))
frequent_diseases = json.load(open('/media/lleger/LaCie/mit/disease_geometry/diseases.json'))['diseases']

def get_adult(row):
    value = age_map.get(normalise_str(row['development_stage']), 40)
    if pd.isna(value): print('Error', row['development_stage'])
    return "yes" if value // 10 >= 4 else "no"

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
    included_phenotypes = ['disease', 'cell_type', 'tissue_general', 'adult']
    anndata_slices = []
    total_cells = 0
    
    chunk_files = [DATASET + f'cxg_chunk{i}.h5ad' for i in range(TEST_CHUNKS[0], TEST_CHUNKS[1] + 1)]
    progressbar = tqdm(chunk_files, desc=description)
    
    for file_path in progressbar:
        progressbar.set_description(f"{description}: {total_cells} cells")
        loaded_chunk = sc.read_h5ad(file_path)
        loaded_chunk.obs['adult'] = loaded_chunk.obs.apply(get_adult, axis=1)
        
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

for disease in frequent_diseases:
    break
    if disease == "normal":
        continue
    
    tissue = disease_tissue[disease]
    cell_types = cell_types_per_disease[disease]
    required_cells = {(disease, cell_type, tissue, 'yes'): CELLS_PER_TYPE for cell_type in cell_types}
    
    anndata_slices = collect_cells(required_cells, f"Collecting {disease}")
    saved_cells = save_shards(anndata_slices, OUTPUT_DIR + disease.replace(' ', '_'))
    print(f"{disease}: Saved {saved_cells} cells")
    del anndata_slices

def collect_cells_streaming(required_cells, output_prefix, description):
    included_phenotypes = ['disease', 'cell_type', 'tissue_general', 'adult']
    buffer = []
    buffer_size = 0
    shard_index = 0
    total_cells = 0
    
    chunk_files = [DATASET + f'cxg_chunk{i}.h5ad' for i in range(TEST_CHUNKS[0], TEST_CHUNKS[1] + 1)]
    progressbar = tqdm(chunk_files, desc=description)
    
    for file_path in progressbar:
        progressbar.set_description(f"{description}: {total_cells} cells, shard {shard_index}")
        loaded_chunk = sc.read_h5ad(file_path)
        loaded_chunk.obs['adult'] = loaded_chunk.obs.apply(get_adult, axis=1)
        
        for phenotype_combination in list(required_cells.keys()):
            mask = np.logical_and.reduce([
                loaded_chunk.obs[phenotype] == value 
                for phenotype, value in zip(included_phenotypes, phenotype_combination)
            ])
            if not mask.any():
                continue
            filtered_chunk = loaded_chunk[mask][:required_cells[phenotype_combination]].copy()
            anndata_slice = AnnData(X=filtered_chunk.X.copy(), obs=filtered_chunk.obs.copy(), var=filtered_chunk.var.copy())
            buffer.append(anndata_slice)
            buffer_size += anndata_slice.n_obs
            total_cells += anndata_slice.n_obs
            required_cells[phenotype_combination] -= anndata_slice.n_obs
            if required_cells[phenotype_combination] <= 0:
                del required_cells[phenotype_combination]
            del filtered_chunk
            
            while buffer_size >= SHARD_SIZE:
                combined = concat(buffer)
                shard = combined[:SHARD_SIZE].copy()
                shard.obs.index.name = None
                shard.write(f"{output_prefix}_shard{shard_index}.h5ad")
                shard_index += 1
                remainder = combined[SHARD_SIZE:].copy() if combined.n_obs > SHARD_SIZE else None
                del combined, shard, buffer
                buffer = [remainder] if remainder is not None else []
                buffer_size = remainder.n_obs if remainder is not None else 0
        
        del loaded_chunk
        if not required_cells:
            break
    
    if buffer_size > 0:
        combined = concat(buffer)
        combined.obs.index.name = None
        combined.write(f"{output_prefix}_shard{shard_index}.h5ad")
        del combined, buffer
    
    return total_cells

#for disease in frequent_diseases:
#    if disease == "normal":
#        continue
#    tissue = disease_tissue[disease]
#    cell_types = cell_types_per_disease[disease]
#    required_cells = {(disease, cell_type, tissue, 'yes'): CELLS_PER_TYPE for cell_type in cell_types}
#    saved_cells = collect_cells_streaming(required_cells, OUTPUT_DIR + disease.replace(' ', '_'), f"Collecting {disease}")
#    print(f"{disease}: Saved {saved_cells} cells")

normal_specifications = cell_types_per_disease['normal']
unique_normal_combinations = {(spec[0], spec[1]) for spec in normal_specifications}
required_cells = {('normal', cell_type, tissue, 'yes'): CELLS_PER_TYPE for cell_type, tissue in unique_normal_combinations}
saved_cells = collect_cells_streaming(required_cells, OUTPUT_DIR + "normal", "Collecting normal")
print(f"Normal: Saved {saved_cells} cells")

