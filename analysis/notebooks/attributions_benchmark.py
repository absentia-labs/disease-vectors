import sys
from tqdm import tqdm
sys.path.append('../../../')
import scanpy as sc
from polygene.model.model import load_trained_model
import numpy as np, pandas as pd
from rapidfuzz import process, fuzz
from polygene.analysis.attributions import AttributionAnalysis
from polygene.data_utils.tokenization import normalise_str
from polygene.analysis.notebooks.get_disease_vectors import compute_disease_vectors
EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/vector_data/'
model_path = '/media/lleger/LaCie/mit/disease_vector/POLYGENE/' #
model_path = "../../../runs/gesam_polygene_run_4/" #'/media/lleger/LaCie/mit/disease_vector/POLYGENE/' #

m, tok = load_trained_model(model_path)
tok.bypass_inference=True
phenotypic_types = tok.phenotypic_types
DISEASES_DICT = {'respiratory': ['COVID-19','influenza','lung adenocarcinoma'],
                 'neurological': ['Alzheimer disease','Parkinson disease','glioblastoma'],
                 'cardiometabolic': ['myocardial infarction','dilated cardiomyopathy','arrhythmogenic right ventricular cardiomyopathy']}
disease_list = sum(list(DISEASES_DICT.values()), [])
results={}
dv_genes = None
pbar = tqdm(disease_list)
num_cells = 1000
num_opentarget_genes = 1000
METHOD = "Q3"
np.random.seed(3)
for disease in pbar:
    pbar.set_description(f"Attributions {disease}")
    cells = sc.read_h5ad(f"{EMBEDDINGS_DIR}{disease}_cells.h5ad")
    cells.obs_names_make_unique()
    embeddings = pd.read_pickle(f"{EMBEDDINGS_DIR}{disease}_embeddings.pkl")
    normal_cells = cells[cells.obs['disease'] == 'normal'][:num_cells]
    disease_cells = cells[cells.obs['disease'] == disease][:num_cells]

    analyzer = AttributionAnalysis(m, tok, data=sc.concat([normal_cells, disease_cells]),
                                   biotype_json="../../data_utils/vocab/gene_biotypes.json", ensembl_json="../../data_utils/vocab/ensembl_to_gene.json")

    opentarget_genes = analyzer.get_associated_genes(disease, disease_cells.obs['disease_ontology_term_id'].tolist()[0], top=num_opentarget_genes)
    if not opentarget_genes: continue

    #out_gwas, gwas_genes, out_mi, mi_genes = analyzer.baselines(phenotype_obs_key='disease', case_label=disease_cells.obs['disease'].unique()[0], control_label="normal", k=num_opentarget_genes, method_top_attr=METHOD, gwas_only=False,start_with_X=int(1e4))
    #out_gwas, gwas_genes = analyzer.baselines(phenotype_obs_key='disease', case_label=disease_cells.obs['disease'].unique()[0], control_label="normal", k=num_opentarget_genes, method_top_attr=METHOD, start_with_X=int(1e4), gwas_only=True)

    #dv_genes = analyzer.disease_vector_ig(index_pairs=list(zip(normal_cells.obs_names.tolist(), disease_cells.obs_names.tolist())), only_protein_encoding=True, disable_pbar=True, steps=2)
    #out_dv = analyzer.validate_attributions(k=num_opentarget_genes, method_top_attr=METHOD, phenotype_obs_value=disease_cells.obs['disease'].unique()[0], baseline=dv_genes.sum(axis=0))
    
    #analyzer.data = disease_cells
    grad_genes = analyzer.gradients(only_protein_encoding=True, disable_pbar=True, force_phenotype_value=normalise_str(disease))
    out_grad = analyzer.validate_attributions(k=num_opentarget_genes, method_top_attr=METHOD, phenotype_obs_value=disease_cells.obs['disease'].unique()[0], baseline=grad_genes.sum(axis=0))

    #ig_genes = analyzer.integrated_gradients(only_protein_encoding=True, steps=10, disable_pbar=True)
    #out_ig = analyzer.validate_attributions(k=num_opentarget_genes, method_top_attr=METHOD, phenotype_obs_value=disease_cells.obs['disease'].unique()[0], baseline=ig_genes.sum(axis=0))
#
    #dl_genes = analyzer.deep_lift(only_protein_encoding=True, disable_pbar=True)
    #out_dl = analyzer.validate_attributions(k=num_opentarget_genes, method_top_attr=METHOD, phenotype_obs_value=disease_cells.obs['disease'].unique()[0], baseline=dl_genes.sum(axis=0))
    
    # save open target genes, save cells, embeddings, 
    attr_analysis = {
                    "embeddings": np.concatenate([embeddings[0][cells.obs['disease'] == 'normal'][:num_cells], embeddings[0][cells.obs['disease'] != 'normal'][:num_cells]]),
                    "disease": np.concatenate([embeddings[2][:, tok.phenotypic_types.index('disease')][cells.obs['disease'] == 'normal'][:num_cells],
                                                embeddings[2][:, tok.phenotypic_types.index('disease')][cells.obs['disease'] != 'normal'][:num_cells]]),
                                                  "cells_obs_name": disease_cells.obs_names.tolist(),
                    #"gwas": gwas_genes, "mutual_info": mi_genes, 
                    #"dv_genes": dv_genes,
                      "grad_genes": grad_genes,# "ig_genes": ig_genes, "dl_genes": dl_genes,
                        "opentarget_genes": opentarget_genes
                    }
    pd.to_pickle(attr_analysis, EMBEDDINGS_DIR + disease + "_full_attributions_embeddings_fast.pkl")
    #results[disease] = {
    #                    "GWAS": out_gwas,
    #                    "Mutual Information": out_mi, 
    #                    "Disease Vector": out_dv,
    #                    "Gradients": out_grad,
    #                    "Integrated Gradients": out_ig,
    #                    "DeepLIFT": out_dl,
    #                    "OpenTargets":(set(list(opentarget_genes.keys())), pd.Series(opentarget_genes), 1, 1, 1, 0, (set(list(opentarget_genes.keys()))))
    #                    }

#pd.to_pickle(results, EMBEDDINGS_DIR + f'attribution_results.pkl')