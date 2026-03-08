
import sys
sys.path.append("../../../")
import warnings; warnings.filterwarnings("ignore")

import json
import torch
import pandas as pd, numpy as np, scanpy as sc
from tqdm import tqdm
from anndata import AnnData, concat
from polygene.eval.metrics import prepare_cell
from polygene.model.model import load_trained_model
from polygene.data_utils.tokenization import normalise_str
EMBEDDINGS_DIR =  "/media/lleger/LaCie/mit/disease_geometry/endotypes/"

model, tokenizer = load_trained_model("../../../saved_models/gesam_polygene_run_4/", checkpoint_n=-1)
tokenizer.bypass_inference=True
model.eval()
diseases = json.load(open(EMBEDDINGS_DIR + '../diseases.json'))['diseases']

DATASET = '/media/rohola/ssd_storage/primary/'
TEST_CHUNKS = (2503, 3574)
N_CELLS = int(1e4)
cell_count = {d: N_CELLS for d in diseases}

for disease in diseases:
    anndata_slices, total_cells = [], 0

    progressbar = tqdm([DATASET + f'cxg_chunk{i}.h5ad'  for i in range(TEST_CHUNKS[0], TEST_CHUNKS[1] + 1)], desc=f"Collecting {disease} cells")
    for file_path in progressbar:
        progressbar.set_description(f"Collecting {total_cells} {disease} cells")
        loaded_chunk = sc.read_h5ad(file_path)
        mask = loaded_chunk.obs["disease"] == disease
        if not mask.any(): continue

        filtered_chunk = loaded_chunk[mask][:cell_count[disease]].copy()
        slice = AnnData(X=filtered_chunk.X.copy(), obs=filtered_chunk.obs.copy(), var=filtered_chunk.var.copy())
        anndata_slices.append(slice)
        
        total_cells += slice.n_obs; cell_count[disease] -= slice.n_obs
        
        del filtered_chunk, loaded_chunk
        if cell_count[disease] <= 0: 
            del cell_count[disease]
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
        predictions.append([tokenizer.flattened_tokens[output.logits.argmax(dim=-1).squeeze()[1 + idx]] for idx in range(len(tokenizer.phenotypic_types))])

    embeddings = ( torch.cat(embeddings).detach().cpu().numpy(), np.array(predictions), np.array(labels) )
    pd.to_pickle(embeddings, EMBEDDINGS_DIR + f"{disease}_embeddings.pkl")
    del cells, embeddings, predictions, labels