import os, sys, time
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, scanpy as sc

import os, sys, json
sys.path.append('../../../')
from polygene.model.model import load_trained_model
neural_network_path = "/media/lleger/LaCie/mit/runs/polygene_unit_sphere/"

model, tokenizer = load_trained_model(neural_network_path, checkpoint_n=12)
tokenizer.bypass_inference = True

from scipy.spatial.distance import cdist
def manifold_projection(path, manifold):
    # function to project a smooth path onto a manifold or set of discrete observations

    # path is a (t, d) array

    # manifold is a (n,d) array
    distances = cdist(manifold, path) #(n, t) array
    
    closest_point_indices = distances.argmin(dim=0)
    return closest_point_indices

atlas_path =  "/media/rohola/ssd_storage/lung_cancer_atlas/"
luca_atlas = sc.concat([sc.read_h5ad(atlas_path + file) for file in os.listdir(atlas_path)])

from tqdm import tqdm
from polygene.eval.metrics import prepare_cell

if os.path.exists('polygene_embeddings.npy'):
    embeddings = np.load('polygene_embeddings.npy')
else: 
    hidden_states = []
    for i, cell in enumerate(tqdm(luca_atlas)):
        cell_dict, labels = prepare_cell(cell, tokenizer)
        cell_dict['input_ids'][1: 1+len(tokenizer.phenotypic_types)] = tokenizer.token_to_id_map[tokenizer.mask_token]
        output = model(**{key: val.to(model.device).unsqueeze(0) for key, val in cell_dict.items()})
        hidden_states.append( output.hidden_states[:, 1 + tokenizer.phenotypic_types.index('disease')].detach().cpu().numpy() )
    np.save("polygene_embeddings.npy", arr=np.concatenate(hidden_states))