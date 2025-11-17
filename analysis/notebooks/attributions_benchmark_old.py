import sys
from tqdm import tqdm
sys.path.append('../../../')
import scanpy as sc
from polygene.model.model import load_trained_model
import numpy as np, pandas as pd
from rapidfuzz import process, fuzz
from polygene.analysis.attributions_old import AttributionAnalysis
from polygene.analysis.notebooks.get_disease_vectors import compute_disease_vectors
EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/vector_data/'

K = 50
METHOD = "Q3"
cell_count = 8000
disease_vectors = pd.DataFrame(compute_disease_vectors(cell_count=cell_count, force_min_length=500, raw_vectors=False))

model_path = '/media/lleger/LaCie/mit/disease_vector/POLYGENE/' #"../../../runs/gesam_polygene_run_4/"
m, tok = load_trained_model(model_path)
tok.bypass_inference=True
phenotypic_types = tok.phenotypic_types
DISEASES_DICT = {'respiratory': ['COVID-19','influenza','lung adenocarcinoma'],
                 'neurological': ['Alzheimer disease','Parkinson disease','glioblastoma'],
                 'cardiometabolic': ['myocardial infarction','dilated cardiomyopathy','arrhythmogenic right ventricular cardiomyopathy']}
disease_list = sum(list(DISEASES_DICT.values()), [])
results={}
dv_genes = None
pbar = tqdm(disease_vectors.columns.tolist())
for cell_group in pbar:
    disease, cell_type = tuple(cell_group.split(' '))

    best_match_disease,_ , _ = process.extractOne(disease, disease_list, scorer=fuzz.ratio, processor=lambda x: x.lower().replace('-', ' ').replace('_', ' '))
    pbar.set_description(f"Disease Vectors {best_match_disease}")
    cells = sc.read_h5ad(f"{EMBEDDINGS_DIR}{best_match_disease}_cells.h5ad")
    cells.obs_names_make_unique()
    idx_pairs = np.array(disease_vectors.loc["anndata_obs_pairs", cell_group])
    normal_cells = cells[np.unique(idx_pairs[:, 0])]
    disease_cells = cells[np.unique(idx_pairs[:, 1])]
    #print(normal_cells.obs['disease'].unique(), '\n', disease_cells.obs['disease'].unique())

    analyzer = AttributionAnalysis(m, tok, data=sc.concat([normal_cells, disease_cells]), biotype_json="../../data_utils/vocab/gene_biotypes.json", ensembl_json="../../data_utils/vocab/ensembl_to_gene.json")
    opentarget_genes = analyzer.get_associated_genes(best_match_disease, disease_cells.obs['disease_ontology_term_id'].tolist()[0], top=1000)
    if not opentarget_genes: continue
    out_gwas, out_mi = analyzer.baselines(phenotype_obs_key='disease', case_label=disease_cells.obs['disease'].unique()[0], control_label="normal", k=K, method_top_attr=METHOD, start_with_X=int(1e4))
    dv_genes = analyzer.disease_vector_ig(index_pairs=idx_pairs, only_protein_encoding=True, disable_pbar=True, steps=2).sum(axis=0)
    out_dv = analyzer.validate_attributions(k=K, method_top_attr=METHOD, phenotype_obs_value=disease_cells.obs['disease'].unique()[0], baseline=dv_genes)
    
    analyzer.data = disease_cells
    grad_genes = analyzer.gradients(only_protein_encoding=True, disable_pbar=True).sum(axis=0)
    out_grad = analyzer.validate_attributions(k=K, method_top_attr=METHOD, phenotype_obs_value=disease_cells.obs['disease'].unique()[0], baseline=grad_genes)

    ig_genes = analyzer.integrated_gradients(only_protein_encoding=True, steps=10, disable_pbar=True).sum(axis=0)
    out_ig = analyzer.validate_attributions(k=K, method_top_attr=METHOD, phenotype_obs_value=disease_cells.obs['disease'].unique()[0], baseline=ig_genes)
    dl_genes = analyzer.deep_lift(only_protein_encoding=True, disable_pbar=True).sum(axis=0)
    out_dl = analyzer.validate_attributions(k=K, method_top_attr=METHOD, phenotype_obs_value=disease_cells.obs['disease'].unique()[0], baseline=dl_genes)
    results[cell_group] = {"GWAS": out_gwas, "Mutual Information": out_mi, 
                           "Disease Vector": out_dv, "Gradients": out_grad,
                            "Integrated Gradients": out_ig, "DeepLIFT": out_dl,
                            "OpenTargets":opentarget_genes}

pd.to_pickle(results, EMBEDDINGS_DIR + f'attribution_results_COUNT{cell_count}_K{K}_METHOD{METHOD}.pkl')