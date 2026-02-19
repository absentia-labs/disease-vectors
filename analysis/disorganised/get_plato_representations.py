# representation alignment compute RA for all models versus number of cells with scatter of model size number of parameters

plato_experiment_path = "/media/lleger/LaCie/mit/disease_geometry/plato/"
import os
model_paths = sorted([path for path in os.listdir(plato_experiment_path) if len(os.listdir(plato_experiment_path + path)) > 10 and "TIED" not in path])
print(model_paths)

import scanpy as sc
import pandas as pd, numpy as np

import sys
sys.path.append('../../')
from polygene.model.model import load_trained_model

checkpoints = [checkpoint for checkpoint in os.listdir(plato_experiment_path + model_paths[0]) if "checkpoint" in checkpoint]
n_checkpoints = len(checkpoints)
print("n_checkpoints:", n_checkpoints)

tokenizer = pd.read_pickle(plato_experiment_path + model_paths[0] + "/tokenizer.pkl")
batch_size = tokenizer.config.per_device_train_batch_size
print("batchsize:", batch_size)
saved_steps = [int(checkpoint.split('-')[1]) for checkpoint in checkpoints]
print("saved training steps:", saved_steps)
print("number of cells seen", np.array(saved_steps)*batch_size)

test_data_path = '/media/lleger/LaCie/mit/disease_geometry/dataset/representation_test_shard0.h5ad'
test_cells = sc.read_h5ad(test_data_path)
subsample_indices = np.random.choice(list(range(len(test_cells))), size=len(test_cells)//2)
test_cells = test_cells[subsample_indices]
print(test_cells.shape,)
models = [load_trained_model(plato_experiment_path + path + "/")[0] for path in model_paths]
reference_model = "polygene_seed_3_layers_6_dim_384"

def count_model_parameters(model, exclude = "embedding"):
    return sum(parameter.numel() for name, parameter in model.named_parameters() if exclude not in name)# / 1_000_000:.2f}M")
number_of_parameters = [count_model_parameters(model, exclude="blank") for model in models]
#display(pd.DataFrame(list(zip(model_paths, number_of_parameters)), columns=["name", "complexity"]))

import torch
checkpoints_to_exclude = [10, 12]
from polygene.eval.metrics import prepare_cell
from tqdm import tqdm
representations = {}
pbar = tqdm(desc="extracting embeddings", total=len(model_paths)*len(test_cells)*(len(checkpoints) - 2))
for idx, model_path in enumerate(model_paths):
    for checkpoint_number in range(len(checkpoints)):
        if checkpoint_number in checkpoints_to_exclude: continue
        model, tokenizer = load_trained_model(plato_experiment_path + model_path + "/", checkpoint_n=checkpoint_number)
        #model.eval()
        for cell in test_cells:
            cell_dict, labels = prepare_cell(cell, tokenizer)
            # mask phenotypes
            phenotype_token_mask = tokenizer.get_phenotypic_tokens_mask(cell_dict['token_type_ids'])
            cell_dict['input_ids'][phenotype_token_mask] = tokenizer.token_to_id_map[tokenizer.mask_token]
            
            # forward pass
            output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items()})
            hidden_state = torch.concatenate(output.hidden_states['hidden_states'])[-1, 1 + tokenizer.phenotypic_types.index('disease')]
            representations.setdefault(model_paths[idx], []).append(hidden_state.detach().cpu().numpy())
            pbar.update(1)

pd.to_pickle(representations, filepath_or_buffer="/media/lleger/LaCie/mit/disease_geometry/plato.pkl")