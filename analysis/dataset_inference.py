import os, sys, json
sys.path.append('../../')
from polygene.model.model import load_trained_model
from polygene.eval.metrics import prepare_cell
import numpy as np, pandas as pd, scanpy as sc
from tqdm import tqdm
import torch

dataset_path = "/media/lleger/LaCie/mit/disease_geometry/dataset/"
neural_network_path ="/media/lleger/LaCie/mit/runs/polygene_unit_sphere/"
model, tokenizer = load_trained_model(neural_network_path, checkpoint_n=-1)


top_genes = list(json.load(open("../data_utils/vocab/gene_ranking_map.json")).keys())[:int(2e3)]
predictions = {}
mask_id = tokenizer.token_to_id_map[tokenizer.mask_token]
masking_range_step = 4
gene_expressions = []
for file in os.listdir(dataset_path):
    shard = sc.read_h5ad(dataset_path + file)
    #shard = shard[np.random.choice(np.arange(len(shard)), size=min(2000, len(shard)))]
    for cell in tqdm(shard, desc=file):
        cell_dict, labels = prepare_cell(cell, tokenizer)
        predictions.setdefault('phenotype_labels', []).append(labels[1:1+len(tokenizer.phenotypic_types)])
        #predictions.setdefault('genome_lengths', []).append(len(labels[1+len(tokenizer.phenotypic_types):]))
        #predictions.setdefault('genotype_labels', []).extend([int(bin_expression.split('_')[1]) for bin_expression in labels[1+len(tokenizer.phenotypic_types):]])

        label_ids = cell_dict['input_ids'][1:1+len(tokenizer.phenotypic_types)].clone()
        cell_dict['input_ids'][1:1+len(tokenizer.phenotypic_types)] = mask_id
        
        output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items()})
        phenotype_prediction = [[tokenizer.flattened_tokens[idx.item()]  for idx in torch.topk(token_logits, k=3).indices]
                                for token_logits in output.logits.squeeze()[1:1+len(tokenizer.phenotypic_types)]]
        predictions.setdefault('phenotype_top_3_predictions', []).append(phenotype_prediction)
        predictions.setdefault('hidden_states', []).append(output.hidden_states[:, 1+tokenizer.phenotypic_types.index('disease')].detach().cpu().numpy())

        #cell_dict['input_ids'][1:1+len(tokenizer.phenotypic_types)] = label_ids
        for mask_percentage in list(np.clip(np.arange(0, 1 + 1/masking_range_step, 1/masking_range_step), 0, 1)):
            break
            mask_indices = np.random.rand(len(cell_dict['input_ids']) - len(tokenizer.phenotypic_types) - 1) < masking_range_step
            cell_dict['input_ids'][1+len(tokenizer.phenotypic_types):][mask_indices] = mask_id
            output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items()})
            predictions.setdefault(f'genotype_predictions_{mask_percentage}', []).extend([int(tokenizer.flattened_tokens[idx.item()].split('_')[1]) for idx in output.logits.squeeze().argmax(
                                                                    dim=-1)[1+len(tokenizer.phenotypic_types):]])
    gene_expressions.extend(shard[:, top_genes].X.toarray().tolist())
    
from umap import UMAP
from sklearn.decomposition import PCA

pca = PCA(n_components=50,)
umap = UMAP(n_neighbors=100, min_dist=2.5,
            spread=3, repulsion_strength=0.5,
            metric='cosine',
            n_components=2)

print("starting umaps")
predictions['umap'] = umap.fit_transform(pca.fit_transform(np.array(gene_expressions)))
predictions['pgmap'] = umap.fit_transform(pca.fit_transform(np.concatenate(predictions['hidden_states'])))

pd.to_pickle(predictions, dataset_path + "../dataset_predictions.pkl")
