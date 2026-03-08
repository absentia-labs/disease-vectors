import torch
from tqdm import tqdm
import os
import scanpy as sc
import numpy as np, pandas as pd
import sys
sys.path.append('../../../../')
metadata_matrix = pd.read_csv("GSE148842-GPL18573_series_matrix.txt", sep='\t', comment='!', header=None, index_col=0).T
metadata_matrix.columns = ['age', 'gender', 'tissue', 'disease', 'drug', 'protocol', 'filename']
print(metadata_matrix.sample())

data_path = "/media/lleger/LaCie/mit/disease_vector/vector_data/glioblastoma_study/"
save_path = "/media/lleger/LaCie/mit/disease_geometry/vectors/glioblastoma_study/"


from polygene.eval.metrics import prepare_cell
from polygene.model.model import load_trained_model
from polygene.data_utils.tokenization import normalise_str
model, tokenizer = load_trained_model("../../../../runs/gesam_polygene_run_4/")
decoder = model.prediction_head

log_partition_function = lambda xi: torch.log( torch.sum( torch.exp( xi ) ) )

def fisher_rao_metric(z, decoder):
    jacobian_decoder = torch.func.jacfwd(decoder)(z)
    hessian_log_partition = torch.func.hessian(log_partition_function)(decoder(z))
    G = jacobian_decoder.T @ hessian_log_partition @ jacobian_decoder
    return G

embeddings, labels, predictions, drug, gradients, fisher_rao = ([] for _ in range(6))
for path in os.listdir(data_path):
    if path.endswith('pkl'): continue
    anndata = sc.read_h5ad(data_path + path)
    for cell in tqdm(anndata):
        cell_dict = prepare_cell(cell, tokenizer)
        cell_dict['input_ids'][np.arange(1, 1 + len(tokenizer.phenotypic_types))] = 2
        with torch.no_grad():
            output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items() if key != 'str_labels'})
        encoder_output = output.hidden_states
        embedding = encoder_output[0, 1 + tokenizer.phenotypic_types.index('disease')].detach().cpu().numpy()
        embeddings.append(embedding)
        labels.append(cell_dict['str_labels'][1:1 + len(tokenizer.phenotypic_types)])
        predictions.append([tokenizer.flattened_tokens[output.logits.argmax(dim=-1).squeeze()[1 + idx]] 
                            for idx in range(len(tokenizer.phenotypic_types))])
        
        z = torch.tensor(embedding, dtype=torch.float32, device=model.device).requires_grad_(True)
        probability_distribution = torch.nn.functional.softmax(decoder(z.unsqueeze(0)), dim=1).squeeze() 
        probability_of_y =  probability_distribution[tokenizer.token_to_id_map[normalise_str("glioblastoma")]]
        decoder.zero_grad()
        probability_of_y.backward(retain_graph=True)
        gradients.append( z.grad.detach().cpu().numpy() )
        z.grad.zero_()

        metric = fisher_rao_metric(z, decoder)
        fisher_rao.append(fisher_rao)

    drug.extend( metadata_matrix[metadata_matrix['filename'] == path.split('_')[0]]['drug'].tolist() * len(anndata))
    
df_g = pd.DataFrame({"embeddings": embeddings, "labels": labels, "predictions": predictions, "drug": drug, "gradients": gradients,
                     "metric": fisher_rao})
df_g.to_pickle(save_path + "embeddings.pkl")