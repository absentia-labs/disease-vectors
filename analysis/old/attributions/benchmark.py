import json,os
import sys
sys.path.append('../../../')
import scanpy as sc
import pandas as pd, numpy as np
from polygene.analysis.attributions.attributions import AttributionAnalysis
from polygene.model.model import load_trained_model
from tqdm import tqdm
from polygene.analysis.geodesics.geodesics import Geodesic
DIR =  "/media/lleger/LaCie/mit/disease_geometry/attributions/"
random_state=3
K = 100
n_cells = 1000

files = os.listdir(DIR)
model, tokenizer = load_trained_model("../../../saved_models/gesam_polygene_run_4/", checkpoint_n=-1)
tokenizer.bypass_inference=True
decoder = model.prediction_head
analyzer = AttributionAnalysis(model, tokenizer, biotype_json="../../data_utils/vocab/gene_biotypes.json", ensembl_json="../../data_utils/vocab/ensembl_to_gene.json")

results = {}
for file in files:
    if "cells" not in file: continue
    #if "non" not in file: continue
    print(file)
    disease = file.split('_cells')[0]
    results[disease] = {}

    cells = sc.read_h5ad(DIR + file)
    if len(cells.obs['disease'].value_counts()) == 1: continue

    manifold = pd.read_pickle(DIR + disease + "_embeddings.pkl")[0]
    cells.obs_names_make_unique()
    sampled = cells.obs.groupby("disease", group_keys=False).apply(lambda x: x.sample(n=min(n_cells, len(x)), random_state=random_state)).index
    analyzer.data = cells[sampled]

    gwas_eval, mutual_information_eval = analyzer.baselines(k=K, baseline_label="normal", target_label=disease) # OVERWRITE DISEASE ONTOLOGY ID HERE TOO. 

    # Disease Vector/ Information Geometry Attributions:
    normal_states = manifold[cells.obs_names.get_indexer(sampled)][analyzer.data.obs['disease'] == "normal"][:20]
    disease_states = manifold[cells.obs_names.get_indexer(sampled)][analyzer.data.obs['disease'] == disease][:20]
    disease_ontology_id = analyzer.data[analyzer.data.obs['disease'] == disease].obs['disease_ontology_term_id'].tolist()[0]
    disease_vector_attributions, geodesic_attributions = [], []
    for z0, z1 in zip(tqdm(normal_states), disease_states):
        geodesic = Geodesic(z0, z1, total_t=25, decoder=decoder, device=model.device)
        mask = geodesic.discretize(manifold=manifold, k=1)
        disease_vector_attributions.append( geodesic.path_integrated_gradients(
            cells, mask, model, tokenizer, phenotype_value=disease, disease_ontology_id=disease_ontology_id, k=K)['cumulative_integrated_gradients'] )
        geodesic.optimize(steps=250, lr=1e-3)
        mask = geodesic.discretize(manifold=manifold, k=1)
        geodesic_attributions.append( geodesic.path_integrated_gradients(
            cells, mask, model, tokenizer, phenotype_value=disease, disease_ontology_id=disease_ontology_id, k=K)['cumulative_integrated_gradients'],)

    disease_vector_attributions = pd.concat(disease_vector_attributions, axis=1).T.fillna(0).sum(axis=0)
    results[disease]['Disease Vector'] = analyzer.validate_attributions(k=K, phenotype_obs_value=disease,
                                                        overwrite_ontology_id=disease_ontology_id, baseline=disease_vector_attributions)

    geodesic_attributions =  pd.concat(geodesic_attributions, axis=1).T.fillna(0).sum(axis=0)
    results[disease]['Information Geometry'] = analyzer.validate_attributions(k=K, phenotype_obs_value=disease,
                                                        overwrite_ontology_id=disease_ontology_id, baseline=geodesic_attributions)
        
    results[disease]['DGE'] = gwas_eval
    results[disease]['Mutual Information'] = mutual_information_eval

    analyzer.data = analyzer.data[analyzer.data.obs['disease'] == disease]

    analyzer.gradients(force_phenotype_value=disease, disable_pbar=False)
    results[disease]['Gradients'] = analyzer.validate_attributions(k=K, phenotype_obs_value=disease, overwrite_ontology_id=disease_ontology_id)
    
    analyzer.integrated_gradients(force_phenotype_value=disease, disable_pbar=False)
    results[disease]['Integrated Gradients'] = analyzer.validate_attributions(k=K, phenotype_obs_value=disease, overwrite_ontology_id=disease_ontology_id)
    
    analyzer.deep_lift(force_phenotype_value=disease, disable_pbar=False)
    results[disease]['DeepLIFT'] = analyzer.validate_attributions(k=K, phenotype_obs_value=disease, overwrite_ontology_id=disease_ontology_id)
    pd.to_pickle(results, "results.pkl")