import sys
from tqdm import tqdm
sys.path.append('../../../')
import scanpy as sc
from polygene.model.model import load_trained_model
from polygene.data_utils.tokenization import normalise_str
import matplotlib.pyplot as plt, seaborn as sns, numpy as np, pandas as pd
from rapidfuzz import process, fuzz
from polygene.analysis.attributions import AttributionAnalysis
EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/vector_data/'
disease_vectors = pd.DataFrame(pd.read_pickle(EMBEDDINGS_DIR + "disease_vector_results.pkl"))
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
for cell_group in tqdm(disease_vectors.columns.tolist(), desc="Disease Vectors"):
    disease, cell_type = tuple(cell_group.split(' '))

    best_match,_ , _ = process.extractOne(disease, disease_list, scorer=fuzz.ratio, processor=lambda x: x.lower().replace('-', ' ').replace('_', ' '))
    print(disease, best_match, cell_type)
    cells = sc.read_h5ad(f"{EMBEDDINGS_DIR}{best_match}_cells.h5ad")
    cells.obs_names_make_unique()
    idx_pairs = np.array(disease_vectors.loc["anndata_obs_pairs", cell_group])
    normal_cells = cells[np.unique(idx_pairs[:, 1])]
    disease_cells = cells[np.unique(idx_pairs[:, 0])]
    normal_cells = normal_cells[normal_cells.obs['disease'] == "normal"]
    disease_cells = disease_cells[disease_cells.obs['disease'] == best_match]
    print(normal_cells.shape, disease_cells.shape)

    analyzer = AttributionAnalysis(m, tok, data=sc.concat([normal_cells, disease_cells]), biotype_json="../../data_utils/vocab/gene_biotypes.json", ensembl_json="../../data_utils/vocab/ensembl_to_gene.json")
    opentarget_genes = analyzer.get_associated_genes(disease_cells.obs['disease_ontology_term_id'].tolist()[0], top=1000)
    if not opentarget_genes: continue

    out_gwas, out_mi = analyzer.baselines(phenotype_obs_key='disease', case_label=disease_cells.obs['disease'].unique()[0], control_label="normal", k=50, method_top_attr="Q3", start_with_X=int(1e4))
    
    dv_genes = analyzer.disease_vector_ig(index_pairs=idx_pairs, only_protein_encoding=True).sum(axis=0)
    out_dv = analyzer.validate_attributions(k=50, method_top_attr="Q3", phenotype_obs_value=disease_cells.obs['disease'].unique()[0], baseline=dv_genes)
    
    analyzer.data = disease_cells
    analyzer.gradients(only_protein_encoding=True)
    out_grad = analyzer.validate_attributions(k=50, method_top_attr="Q3", phenotype_obs_value=disease_cells.obs['disease'].unique()[0])
    analyzer.integrated_gradients(only_protein_encoding=True, steps=10)
    out_ig = analyzer.validate_attributions(k=50, method_top_attr="Q3", phenotype_obs_value=disease_cells.obs['disease'].unique()[0])
    analyzer.deep_lift(only_protein_encoding=True)
    out_dl = analyzer.validate_attributions(k=50, method_top_attr="Q3", phenotype_obs_value=disease_cells.obs['disease'].unique()[0])
    results[cell_group] = {"GWAS": out_gwas, "Mutual Information": out_mi, "Disease Vector": out_dv, "Gradients": out_grad,
                            "Integrated Gradients": out_ig, "DeepLIFT": out_dl,
                            "OpenTargets":opentarget_genes}

pd.to_pickle(results, EMBEDDINGS_DIR + 'attribution_results.pkl')
