import os, sys, time, json
from umap import UMAP
from tqdm import tqdm
sys.path.append('../../')
from sklearn.decomposition import PCA
import pandas as pd, numpy as np, scanpy as sc
from polygene.model.model import load_trained_model
from sklearn.metrics import normalized_mutual_info_score
from datetime import datetime as dt

seed = 3
pca = PCA(n_components=20,)
umap = UMAP(n_neighbors=100, min_dist=2, spread=2, n_components=2, low_memory=True)#, random_state=seed, n_jobs=1)

def compute_metrics(y_true, y_pred, k=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)[:, :k] if k is not None else np.array(y_pred)
    labels = np.unique(y_true)
    metrics = []
    for label in labels:
        true_mask = (y_true == label)
        pred_mask = (y_pred[:, 0] == label) if k is not None else (y_pred == label)
        TP = np.sum(true_mask & pred_mask)
        FN = np.sum(true_mask & ~pred_mask)
        FP = np.sum(~true_mask & pred_mask)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        if k is not None:
            pred_mask_at_k = np.any(y_pred == label, axis=1)
            recall_at_k = np.sum(pred_mask_at_k & true_mask)/np.sum(true_mask)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics.append([precision, recall, f1, recall_at_k]) if k is not None else metrics.append([precision, recall, f1])
    return pd.DataFrame(metrics, index=labels, columns=["precision", "recall", "f1", f"recall@{k}"]) if k is not None else pd.DataFrame(metrics, index=labels, columns=["precision", "recall", "f1"])


highly_variable_genes_rank = json.load(open('/home/lleger/Documents/polygene/data_utils/vocab/gene_ranking_map.json'))

if __name__ == "__main__":
    model_directory = '/media/lleger/LaCie/mit/runs/polygene_tied_linear_reconstruct/'
    DATA_DIR ="/media/rohola/ssd_storage/lung_cancer_atlas/"

    test_dataset = [DATA_DIR + file for file in os.listdir(DATA_DIR)]

    model, tokenizer = load_trained_model(model_directory, checkpoint_n=-1)
    save_dir = model_directory + "plotting_artifacts/"
    os.makedirs(save_dir, exist_ok=True)

    all_hidden_states = []
    all_labels = []
    all_phenotype_predictions = []
    all_gene_expression = []
    for shard_path in tqdm(test_dataset, "concatenating"):
        test_shard = sc.read_h5ad(shard_path)
        test_shard = test_shard[:, test_shard.var['is_highly_variable'].values.astype(bool)]
        inference_results = pd.read_pickle(model_directory + "predictions/" + shard_path.split('.')[0].split('/')[-1] + ".pkl")

        all_gene_expression.append(test_shard.X.toarray())
        all_hidden_states.append(np.array(inference_results['hidden_states']))
        all_labels.append([label[1: 1+len(tokenizer.phenotypic_types)] for label in inference_results['labels']])
        all_phenotype_predictions.append(inference_results['phenotype_prediction'])

    all_gene_expression = np.concatenate(all_gene_expression)
    all_hidden_states = np.concatenate(all_hidden_states)
    all_labels = [label for shard_labels in all_labels for label in shard_labels]
    all_phenotype_predictions = [pred for shard_preds in all_phenotype_predictions for pred in shard_preds]

    print(all_gene_expression.shape, all_hidden_states.shape)
    print(f"Running PCA on gene expression... {dt.now():%H:%M}")
    gene_expression_pca = pca.fit_transform(all_gene_expression)
    print(f"Running UMAP on gene expression... {dt.now():%H:%M}")
    gene_expression_umap = umap.fit_transform(gene_expression_pca)

    print(f"Running PCA on hidden states... {dt.now():%H:%M}")
    hidden_states_pca = pca.fit_transform(all_hidden_states)
    print(f"Running UMAP on hidden states... {dt.now():%H:%M}")
    attention_umap = umap.fit_transform(hidden_states_pca)

    plotting_artifacts = {
        "gene_expression_umap": gene_expression_umap,
        "attention_umap": attention_umap,
        "labels": all_labels
    }

    for pdx, phenotype in enumerate(tqdm(tokenizer.phenotypic_types, desc="phenotype predictions")):
        classification_metrics = compute_metrics(
            y_true=[labels[pdx] for labels in all_labels],
            y_pred=[prediction[pdx] for prediction in all_phenotype_predictions],
            k=3
        )
        plotting_artifacts["metrics_" + phenotype] = classification_metrics

    pd.to_pickle(plotting_artifacts, filepath_or_buffer=save_dir + "luca_plotting_artifacts.pkl")
    print(f"Saved plotting_artifacts")
