import pandas as pd, numpy as np, matplotlib.pyplot as plt

import os, sys, copy, json
sys.path.append('../../')
from polygene.model.model import load_trained_model

model_directory = '/media/lleger/LaCie/mit/runs/polygene_tied_linear_reconstruct/'
frequent_diseases = json.load(open('/media/lleger/LaCie/mit/disease_geometry/diseases.json'))['diseases']

DATA_DIR ="/media/rohola/ssd_storage/lung_cancer_atlas/"
test_dataset = [DATA_DIR + file for file in os.listdir(DATA_DIR)]
save_dir = model_directory + "predictions/"
os.makedirs(save_dir, exist_ok = True)

model, tokenizer = load_trained_model(model_directory)
model.eval()
tokenizer.bypass_inference=True

import scanpy as sc
import torch
from tqdm import tqdm
from polygene.eval.metrics import prepare_cell

disease_token_idx = tokenizer.phenotypic_types.index("disease")
#gene_mask_step = 0.25
#mask_range = np.arange(0, 1 + gene_mask_step, gene_mask_step)
#print('gene mask range', mask_range)
for test_shard_path in test_dataset:
    if not os.path.exists(test_shard_path): continue
    if os.path.exists(save_dir + test_shard_path.split('.')[0].split('/')[-1] + ".pkl"): continue
    inference_results = {"data_path": test_shard_path}
    test_shard = sc.read_h5ad(test_shard_path)
    print(test_shard_path)
    for i, cell in enumerate(tqdm(test_shard)):
        # tokenize
        cell_dict, labels = prepare_cell(cell, tokenizer)
        inference_results.setdefault('labels', []).append(labels)
        
        # mask phenotypes
        phenotype_token_mask = tokenizer.get_phenotypic_tokens_mask(cell_dict['token_type_ids'])
        label_ids = cell_dict['input_ids'][phenotype_token_mask].clone()
        cell_dict['input_ids'][phenotype_token_mask] = tokenizer.token_to_id_map[tokenizer.mask_token]
        
        # forward pass
        output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items()})
        phenotype_prediction = [[tokenizer.flattened_tokens[idx] for idx in torch.topk(token_logits, k=5).indices] for token_logits in output.logits.squeeze()[phenotype_token_mask]]
        top_prediction = [token_prediction[0] for token_prediction in phenotype_prediction]
        inference_results.setdefault('phenotype_prediction', []).append(phenotype_prediction)

        # mask gene expressions
        #gene_token_mask = tokenizer.get_gene_tokens_mask(cell_dict["token_type_ids"])
        #for gene_mask_percentage in mask_range:
        #    cell_dict_gene_masking = cell_dict.copy()
        #    #if int(gene_mask_percentage) == 1: cell_dict_gene_masking['input_ids'][phenotype_token_mask] = label_ids
        #    gene_prob_matrix = torch.zeros(cell_dict["input_ids"].shape).masked_fill(gene_token_mask, value=gene_mask_percentage)
        #    gene_masked_indices = torch.bernoulli(gene_prob_matrix, generator=None).bool()
        #    cell_dict_gene_masking["input_ids"][gene_masked_indices] = tokenizer.token_to_id_map[tokenizer.mask_token]
        #    output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items()})
        #    gene_prediction = [tokenizer.flattened_tokens[idx] for idx in output.logits.squeeze().argmax(dim=1)]
        #    inference_results.setdefault(f'gene_prediction_{gene_mask_percentage}', []).append(gene_prediction)

        # save latent embeddings, hidden states, intermediate activations
        # (L + 1, S, D)
        hidden_state = torch.concatenate(output.hidden_states['hidden_states'])[-1, 1 + disease_token_idx]
        hidden_state = model.prediction_head[0](hidden_state)
        inference_results.setdefault(f'hidden_states', []).append(hidden_state.detach().cpu().numpy())

    #split to not keep file extensions
    pd.to_pickle(inference_results, filepath_or_buffer=save_dir + test_shard_path.split('.')[0].split('/')[-1] + ".pkl")

# Need to run UMAP + plus visualizations, masking rates in same plot. 