import os
import json
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

metadata_matrix = pd.read_csv("GSE148842-GPL18573_series_matrix.txt", sep='\t', comment='!', header=None, index_col=0).T
metadata_matrix.columns = ['age', 'gender', 'tissue', 'disease', 'drug', 'protocol', 'filename']
print(metadata_matrix.sample())

data_path = "/media/lleger/LaCie/dump/disease-vector/glioblastoma/GSE148842_RAW/"
processed_data_path = '/media/lleger/LaCie/mit/disease_vector/vector_data/glioblastoma_study/'
known_genes = set(list(json.load(open("../../data_utils/vocab/gene_ranking_map.json")).keys())) 

pbar = tqdm(os.listdir(data_path))
for file in pbar:
    drug = metadata_matrix[metadata_matrix['filename'] == file.split('_')[0]]['drug'].tolist()
    if not drug: continue
    pbar.set_description(f'Formatting study to anndata: {drug}')
    df = pd.read_csv(data_path + file, sep="\t", index_col=0)
    df = df.drop(columns=df.columns[0])
    df.index = pd.Series(df.index.tolist()).apply(lambda x: x.split('.')[0]).tolist()
    
    adata = sc.AnnData(df.T)
    adata.obs['drug'] = drug * len(adata)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    adata.write_h5ad(processed_data_path + f"{file.split('.')[0]}.h5ad")