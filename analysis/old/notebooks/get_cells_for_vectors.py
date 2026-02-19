EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/vector_data/'

DISEASES_DICT = {
                'respiratory': [ 'COVID-19', 'influenza', 'lung adenocarcinoma'],
                'neurological': ['Alzheimer disease', 'Parkinson disease', 'glioblastoma'],
                'cardiac': ['myocardial infarction', 'dilated cardiomyopathy', 'arrhythmogenic right ventricular cardiomyopathy']
                }

CELL_TYPE_DICT = {
                'COVID-19': ['CD8-positive, alpha-beta T cell', 'CD4-positive, alpha-beta T cell', 'B cell', 'classical monocyte'],
                  'influenza': ['CD8-positive, alpha-beta T cell', 'CD4-positive, alpha-beta T cell', 'classical monocyte'],
                  'lung adenocarcinoma': ['CD8-positive, alpha-beta T cell', 'CD4-positive, alpha-beta T cell', 'B cell'],
                  'Alzheimer disease': ['neuron', 'microglial cell'],
                  'Parkinson disease':['oligodendrocyte', 'neuron', 'microglial cell', 'astrocyte', 'endothelial cell'],
                  'glioblastoma':['malignant cell', 'microglial cell', 'endothelial cell', 'oligodendrocyte', 'monocyte'],
                  'myocardial infarction':['cardiac muscle myoblast', 'fibroblast of cardiac tissue', 'cardiac endothelial cell'],
                  'dilated cardiomyopathy':['cardiac muscle cell', 'fibroblast of cardiac tissue', 'endothelial cell'],
                  'arrhythmogenic right ventricular cardiomyopathy' :['fibroblast of cardiac tissue', 'cardiac muscle cell', 'endothelial cell']
                  }

TISSUE_DICT = {
                'COVID-19': ['blood'],
                  'influenza': ['blood'],
                  'lung adenocarcinoma': ['lung'],
                  'Alzheimer disease': ['brain'],
                  'Parkinson disease':['brain'],
                  'glioblastoma':['brain'],
                  'myocardial infarction':['heart'],
                  'dilated cardiomyopathy':['heart'],
                  'arrhythmogenic right ventricular cardiomyopathy' :['heart']
                  }
import sys
sys.path.append("../../../")
import warnings; warnings.filterwarnings("ignore")

import json
from polygene.model.model import load_trained_model
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, scanpy as sc
import torch
from tqdm import tqdm
from anndata import AnnData, concat
from polygene.eval.metrics import prepare_cell
from polygene.data_utils.tokenization import normalise_str

model, tokenizer = load_trained_model("../../../runs/gesam_polygene_run_4/", checkpoint_n=-1)
tokenizer.bypass_inference=True
diseases = json.load(open(EMBEDDINGS_DIR + '../diseases.json'))['diseases']

N_CHUNKS = 1000
DATASET = '/media/rohola/ssd_storage/primary/'
TEST_CHUNK_ID = 2502

age_map = pd.read_csv('../../data_utils/age_relabeling.csv')
age_map = {v[0]: v[1] for v in age_map[['label', 'age']].values.tolist()}

def get_adult(x, disease):
    val = age_map.get(normalise_str(x['development_stage']))
    if pd.isna(val): print('Error', x['development_stage'])
    return "yes" if val // 10 >= 4 else "no"

for disease in sum(list(DISEASES_DICT.values()), []):
    required_cells = {comb: 2000 for comb in [(d, c, t, 'yes') for d in ['normal', disease] for c in CELL_TYPE_DICT[disease] for t in TISSUE_DICT[disease]]}
    included_phenotypes = ['disease', 'cell_type', 'tissue_general', 'adult']
    anndata_slices, total_cells = [], 0

    progressbar = tqdm([DATASET + f'cxg_chunk{i}.h5ad'  for i in range(TEST_CHUNK_ID, TEST_CHUNK_ID + N_CHUNKS)], desc=f"Collecting {disease} cells")
    for file_path in progressbar:
        progressbar.set_description(f"Collecting {total_cells} {disease} cells")
        loaded_chunk = sc.read_h5ad(file_path)
        loaded_chunk.obs['adult'] = loaded_chunk.obs.apply(lambda x: get_adult(x, disease), axis=1)

        for phenotype_combination in list(required_cells.keys()):
            mask = np.logical_and.reduce([(loaded_chunk.obs[p] == v) 
                                          for p, v in zip(included_phenotypes, phenotype_combination)])
            if not mask.any():
                continue
            filtered_chunk = loaded_chunk[mask][:required_cells[phenotype_combination]].copy()
            slice = AnnData(X=filtered_chunk.X.copy(), obs=filtered_chunk.obs.copy(), var=filtered_chunk.var.copy())
            anndata_slices.append(slice)
            total_cells += slice.n_obs
            required_cells[phenotype_combination] -= slice.n_obs
            if required_cells[phenotype_combination] <= 0:
                del required_cells[phenotype_combination]
            del filtered_chunk
        del loaded_chunk
        if not required_cells:
            break

    result = concat(anndata_slices) if anndata_slices else AnnData()
    print(f"{disease}: Loaded {result.n_obs} cells total.")
    result.obs.index.name = None
    result_path = EMBEDDINGS_DIR + f'{disease}_cells.h5ad'
    result.write(result_path)
    del result
    embeddings, predictions, labels = ([] for _ in range(3))
    cells = sc.read_h5ad(result_path)

    for idx in tqdm(range(cells.n_obs), desc=f"Embeddings {disease}"):
        cell = cells[idx, :]
        cell_dict = prepare_cell(cell, tokenizer)
        cell_dict['input_ids'][np.arange(1, 1 + len(tokenizer.phenotypic_types))] = 2
        with torch.no_grad():
            output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items() if key != 'str_labels'})
        encoder_output = output.hidden_states
        embeddings.append(encoder_output[:, 1 + tokenizer.phenotypic_types.index('disease')])
        labels.append(cell_dict['str_labels'][1:1 + len(tokenizer.phenotypic_types)])
        predictions.append([tokenizer.flattened_tokens[output.logits.argmax(dim=-1).squeeze()[1 + idx]] 
                            for idx in range(len(tokenizer.phenotypic_types))])

    embeddings = ( torch.cat(embeddings).detach().cpu().numpy(), np.array(predictions), np.array(labels) )
    pd.to_pickle(embeddings, EMBEDDINGS_DIR + f"{disease}_embeddings.pkl")
    del cells, embeddings, predictions, labels