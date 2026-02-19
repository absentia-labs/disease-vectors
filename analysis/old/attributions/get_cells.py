EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_geometry/attributions/'

diseases = [#['small cell lung carcinoma', 'lung large cell carcinoma',
            'glioblastoma', 
            #'Alzheimer disease', 'Parkinson disease',
            #'myocardial infarction', 'dilated cardiomyopathy',  
            'non-small cell lung carcinoma'
            ]

CELL_TYPE_DICT = {
                'small cell lung carcinoma': ["epithelial cell of lower respiratory tract", "epithelial cell"],
                'lung large cell carcinoma': ['native cell'],

                'glioblastoma': ['malignant cell', 'oligodendrocyte precursor cell'],
                'Parkinson disease':['neuron'],
                'Alzheimer disease': ['neuron'], 

                'myocardial infarction':['cardiac endothelial cell'],
                'dilated cardiomyopathy':['endothelial cell'],

                'non-small cell lung carcinoma': ['epithelial cell of lower respiratory tract','malignant cell'],
                }

TISSUE_DICT = {
                'small cell lung carcinoma': ["lung"],
                'lung large cell carcinoma': ['lung'],
                'non-small cell lung carcinoma': ['lung'],
                'glioblastoma': ['brain'],

                'Alzheimer disease': ['brain'], 
                'Parkinson disease':['brain',],

                'myocardial infarction':['heart'],
                'dilated cardiomyopathy':['heart'],
            }
import sys
sys.path.append("../../../")
import warnings; warnings.filterwarnings("ignore")
from polygene.model.model import load_trained_model
import pandas as pd, numpy as np, scanpy as sc
import torch
from tqdm import tqdm
from anndata import AnnData, concat
from polygene.eval.metrics import prepare_cell
from polygene.data_utils.tokenization import normalise_str

model, tokenizer = load_trained_model("../../../saved_models/gesam_polygene_run_4/", checkpoint_n=-1)
tokenizer.bypass_inference=True

DATASET = '/media/rohola/ssd_storage/primary/'
TEST_CHUNKS = (2503, 3574)
import json
age_map = json.load(open(('../../data_utils/vocab/age_map.json')))

def get_adult(x, disease):
    val = age_map.get(normalise_str(x['development_stage']), 40)
    if pd.isna(val): print('Error', x['development_stage'])
    return "yes" if val // 10 >= 4 else "no"

for disease in diseases:
    required_cells = {comb: {"normal":int(5e4), disease: int(1e4)}[comb[0]] for comb in [(d, c, t, 'yes') for d in ['normal', disease] for c in CELL_TYPE_DICT[disease] for t in TISSUE_DICT[disease]] if not (comb[0] == "normal" and comb[1] == "malignant cell")}
    print(required_cells)
    included_phenotypes = ['disease', 'cell_type', 'tissue_general', 'adult']
    anndata_slices, total_cells = [], 0

    progressbar = tqdm([DATASET + f'cxg_chunk{i}.h5ad'  for i in range(TEST_CHUNKS[0], TEST_CHUNKS[1] + 1)], desc=f"Collecting {disease} cells")
    for file_path in progressbar:
        progressbar.set_description(f"Collecting {total_cells} {disease} cells")
        loaded_chunk = sc.read_h5ad(file_path)
        loaded_chunk.obs['adult'] = loaded_chunk.obs.apply(lambda x: get_adult(x, disease), axis=1)

        for phenotype_combination in list(required_cells.keys()):
            mask = np.logical_and.reduce([(loaded_chunk.obs[p] == v) for p, v in zip(included_phenotypes, phenotype_combination)])
            if not mask.any(): continue
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
        embeddings.append( encoder_output[:, 1 + tokenizer.phenotypic_types.index('disease')].detach().cpu().numpy() )
        labels.append(cell_dict['str_labels'][1:1 + len(tokenizer.phenotypic_types)])
        predictions.append([tokenizer.flattened_tokens[output.logits.argmax(dim=-1).squeeze()[1 + idx]] 
                            for idx in range(len(tokenizer.phenotypic_types))])

    embeddings = ( np.concatenate(embeddings), np.array(predictions), np.array(labels) )
    pd.to_pickle(embeddings, EMBEDDINGS_DIR + f"{disease}_embeddings.pkl")
    del cells, embeddings, predictions, labels