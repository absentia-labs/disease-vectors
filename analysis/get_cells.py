import scanpy as sc
from tqdm import tqdm
import os, json, argparse
import pandas as pd, numpy as np
import sys
sys.path.append('../../')
from polygene.model.model import load_trained_model
from itertools import product
import torch
from polygene.eval.metrics import prepare_cell
import gc

TEST_CHUNK_ID = 2502
DATASET = "/media/rohola/ssd_storage/primary/"
SAVE_DIR = "/media/lleger/LaCie/mit/disease_vector/"
age_map = json.load(open('../data_utils/vocab/age_map.json'))
diseases = json.load(open(SAVE_DIR + "diseases.json"))['diseases']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Cells")

    parser.add_argument("--n_cells", type=int, default=500, help="number of cells per phenotype")
    parser.add_argument("--n_chunks", type=int, default=500, help="number of test chunks to explore")
    parser.add_argument("--embeddings", action="store_true", help="save embeddings")
    parser.add_argument("--use_50_diseases", action="store_true", help="put all 50 diseases in disease phenotype")
    parser.add_argument("--print_metadata_info", action="store_true", help="put all 50 diseases in disease phenotype")
    
    phenotypes = ["sex", "tissue", "cell_type", "disease", "development_stage"]
    parser.add_argument("--sex",)
    parser.add_argument("--tissue",)
    parser.add_argument("--disease",)
    parser.add_argument("--cell_type",)
    parser.add_argument("--development_stage",)

    args = vars(parser.parse_args())
    if args['use_50_diseases']: args['disease'] = str(diseases)
    included_phenotypes = [p for p in phenotypes if args[p] is not None]
    required_cells = {combo:args['n_cells'] for combo in product(*[eval(args[phene]) for phene in included_phenotypes])}

    if args['embeddings']:
        model, tok = load_trained_model('../../runs/gesam_polygene_run_4/', checkpoint_n=-1)
        tok.bypass_inference=True

anndata_slices, total_cells, chunk_index = [], 0, 0
for idx, file_path in enumerate(tqdm([DATASET + f'cxg_chunk{i}.h5ad' for i in range(TEST_CHUNK_ID, TEST_CHUNK_ID+args["n_chunks"])], desc="Searching dataset")):
    loaded_chunk = sc.read_h5ad(file_path); filtered_chunk = None
    for phenotype_combination in list(required_cells.keys()):
        mask = np.logical_and.reduce([(loaded_chunk.obs[phene] == value) for phene,value in zip(included_phenotypes, phenotype_combination)])
        if not mask.any():
            continue
        filtered_chunk = loaded_chunk[mask][:required_cells[phenotype_combination]]
        slice = sc.AnnData(X=filtered_chunk.X, obs=filtered_chunk.obs.copy(), var=filtered_chunk.var.copy())
        anndata_slices.append(slice)
        total_cells += slice.n_obs
        required_cells[phenotype_combination] -= slice.n_obs
        if required_cells[phenotype_combination] <= 0:
            del required_cells[phenotype_combination]
    
    if total_cells >= 10000 or not required_cells:
        cells = sc.concat(anndata_slices); cells.obs.index.name = None
        print("Saving\n:", cells.obs.value_counts(included_phenotypes))
        cells.write(SAVE_DIR + f"cells_part{chunk_index}.h5ad")
        if args['embeddings']:
            embeddings, predictions, labels = ([] for _ in range(3))
            for cell in tqdm(cells, "Embeddings"):
                cell_dict = prepare_cell(cell, tok)
                cell_dict['input_ids'][np.arange(1, 1+len(tok.phenotypic_types))] = 2
                with torch.no_grad():
                    output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items() if key != 'str_labels'})
                encoder_output = output.hidden_states
                embeddings.append(encoder_output[:, 1+tok.phenotypic_types.index('disease')])
                labels.append(cell_dict['str_labels'][1:1+len(tok.phenotypic_types)])
                predictions.append([tok.flattened_tokens[output.logits.argmax(dim=-1).squeeze()[1+idx]] for idx in range(len(tok.phenotypic_types))])
            embeddings = (torch.cat(embeddings).detach().cpu().numpy(), np.array(predictions), np.array(labels))
            pd.to_pickle(embeddings, SAVE_DIR + f"embeddings_part{chunk_index}.pkl")
            del embeddings, predictions, labels, cells
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        anndata_slices, total_cells, chunk_index = [], 0, chunk_index + 1
    if not required_cells: break
    del loaded_chunk, filtered_chunk
    gc.collect()