from typing import Optional

import numpy as np
import pandas as pd
import torch
import datasets
import transformers
from torch.utils.data import Dataset, DataLoader
from collections import deque, defaultdict
from polygene.data_utils.tokenization import GeneTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ShardedTrainer(transformers.Trainer):
    """
    Modified Trainer for our use case of distributed training on multiple GPUs with a sharded IterableDataset.
    Instead of loading all batches on GPU 0 and dispatching the batches to other GPUs, each GPU loads its own batches.
    """
    def __init__(self, *args, tokenizer: GeneTokenizer = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
        self.compute_metrics = self._compute_metrics
        self.preprocess_logits_for_metrics = self._preprocess_logits_argmax
        
        #self.buffer = defaultdict(lambda: deque(maxlen=50))
        #self.monitor_collapse = monitor_collapse
        #self.compound_loss = compound_loss

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Copied from Trainer.get_train_dataloader(), only last 2 lines modified.

        Assumes that `self.train_dataset` has already been filtered to the relevant shards for this process.
        In this case, we can directly wrap self.train_dataset in a DataLoader without considering distributed loading.
        """
        assert not self.args.dispatch_batches
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if transformers.is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        assert isinstance(train_dataset, torch.utils.data.IterableDataset), \
            "Assuming IterableDataset, normal Trainer should work for map-style datasets"

        return DataLoader(train_dataset, **dataloader_params)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        copied from train dataloader 
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        data_collator = self.data_collator
        if transformers.is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": 1, #self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": False,
            "drop_last": False,
        }

        return DataLoader(eval_dataset, **dataloader_params)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        step = self.state.global_step

        if self.control.should_log and step > self._globalstep_last_logged:
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs = {
                "loss": round(tr_loss_scalar / (step - self._globalstep_last_logged), 4),
                "learning_rate": self._get_learning_rate(),
            }
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = step
            self.store_flos()
            self.log(logs)
        
        logscale_checkpoints = (np.logspace(5, 7, num=10, base=10)/self.args.train_batch_size).astype(int).tolist()
        if step in logscale_checkpoints or (step > logscale_checkpoints[-1] and step % self.args.save_steps == 0):
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                self.log(metrics)
                self._save_checkpoint(model, trial, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _compute_metrics(self, p: transformers.EvalPrediction):
        """
        Computes MLM accuracy from EvalPrediction object.

        Args:
            - p (EvalPrediction): An object containing the predictions and labels.

        Returns:
            - dict: A dictionary containing the accuracy under the key 'accuracy'.
        """
        self.tokenizer.sync(self.model.config.updates_memory, num=self.model.config.vocab_size - len(self.model.config.updates_memory['token_value_str']))
        metrics = {}

        # Extract predictions and labels from the EvalPrediction object
        predictions = p.predictions # (B, S) argmax from preprocess_logits
        labels = p.label_ids # (B, S)  B = shard size/ eval set size, S = max sequence length, filled with token IDs
        #inputs = p.inputs

        # Ignoring -100 used for non-masked tokens (pad, cls, eos tokens)
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        mask = labels != -100
        overall_metrics = classification_metrics(predictions[mask], labels[mask]) # global masks flatten the array

        metrics.update({f"overall_{metric_name}": metric_val for metric_name, metric_val in overall_metrics.items()})

        # Compute metrics per phenotype type aka per token sequence column
        for i, phenotypic_type in enumerate(self.tokenizer.phenotypic_types):
            y_pred, y = predictions[:, i + 1], labels[:, i + 1]
            mask = y != -100
            if len(y[mask]):
                phenotype_metrics = classification_metrics(y_pred[mask], y[mask])
                metrics.update({f"{phenotypic_type}_{key}": value for key, value in phenotype_metrics.items()})

        # Metrics for genotype expression predictions (maybe only if theres gene masking)
        y_pred, y = predictions[:, self.tokenizer.gene_token_type_offset:], labels[:, self.tokenizer.gene_token_type_offset:]
        mask = y != -100
        if len(y[mask]):
            gene_metrics = classification_metrics(y_pred[mask], y[mask])
            metrics.update({f"Genotype_{key}": value for key, value in gene_metrics.items()})

        # Metrics to quantify feature collapse
        #NC1, NC2 = self.neural_collapse_metrics()
        #metrics["NC1"] = NC1; metrics["NC2"] = NC2
        return metrics
    
    def _preprocess_logits_argmax(self, logits, labels):
        """
        We currently only need the top predicted class instead of all the logits,
        so this preprocessing saves significant memory.
        """
        if isinstance(logits, tuple):
            # should not happen for `GeneBert` variants, but other models may have extra tensors like `past_key_values`
            logits = logits[0]

        return logits.argmax(dim=-1) 
    
def classification_metrics(flat_preds: np.ndarray, flat_labels: np.ndarray) -> dict[str, float]:
        """
        Args:
            flat_preds: Flat numpy array of predictions (argmax of logits)
            flat_labels: Flat numpy array of labels, with the same shape as `flat_preds`
            Note that it is assumed that the labels corresponding to -100 have already been filtered out.

        Returns:
            Dictionary of different metric values ("accuracy", "precision", "recall", "f1").

        Note: Setting `average='macro'` for macro-average (average over classes)
        Using `zero_division=0` to handle cases where there are no true or predicted samples for a class
        """

        r = recall_score(flat_labels, flat_preds, average='macro', zero_division=0)
        p = precision_score(flat_labels, flat_preds, average='macro', zero_division=0)
        return {
            "accuracy": accuracy_score(flat_labels, flat_preds),
            "f1":  2 / ((1 / r) + (1 / p)) if r > 0 and p > 0 else 0
        }

# Potential fix, compound loss function to mitigate feature collapse.
# 
#def compute_loss(
#    self,
#    model,
#    inputs,
#    return_outputs = False,
#    num_items_in_batch = None,
#):
#    outputs = model(**inputs)
#    loss = outputs.loss
#
#    if self.compound_loss and self.train_step > int(1e3) and self.train_step % self.update_every == 0:
#        all_embeddings = [torch.stack(list(dq)) for dq in self.buffer.values() if len(dq) > 0]
#        history_Z = torch.cat(all_embeddings, dim=0)  # (K, D)
#        current_Z = outputs.hidden_states[:, 1 + self.tokenizer.phenotypic_types.index("disease")]
#        Z = torch.cat([current_Z, history_Z])
#        v_disease = torch.mean(torch.clamp(1 -  torch.std(Z, dim=0), min=0.0).pow(2))
#        l_uniform = torch.log(torch.exp(-torch.cdist(current_Z, Z.detach(), p=2).pow(2)).mean())
#        loss = loss + 1e-2 * v_disease + 1e-3 * l_uniform
#
#    self.train_step += 1
#
#    if self.monitor_collapse:
#        x = inputs['input_ids'][:, 1 + self.tokenizer.phenotypic_types.index("disease")] # (B, S)[:, idx]
#        z = outputs.hidden_states[:, 1 + self.tokenizer.phenotypic_types.index("disease")] # (B, S, D)[:, idx]
#        for index, vector in zip(x.tolist(), z):
#            self.buffer[index].append(vector.detach().clone())
#    return (loss, outputs) if return_outputs else loss

#    def buffer_length(self):
#        size = 0
#        for key in self.buffer:
#            for batch in self.buffer[key]:
#                size += len(batch)
#        return size

#    def neural_collapse_metrics(self):
#        records = [{"disease": disease_id, "embedding": embedding.detach().cpu().numpy()} for disease_id, dq in self.buffer.items() for embedding in dq ]
#        df = pd.DataFrame(records)
#        K = len(df["disease"].unique())
#        stats_df = df.groupby("disease").apply( lambda g: pd.Series({
#                                                    "mean": np.array(g["embedding"].tolist()).mean(axis=0),
#                                                    "covar": np.cov(np.array(g["embedding"].tolist()).T, bias=True) })).reset_index()
#
#        covariance_K = np.array(stats_df["covar"].tolist()).mean(axis=0)
#        covariance_G = np.cov(np.array(stats_df["mean"].tolist()).T, bias=True)
#        NC1 = (1 / K) * np.trace(covariance_K @ np.linalg.pinv(covariance_G))
#
#        means = np.array(stats_df["mean"].tolist()) - np.array(stats_df["mean"].tolist()).mean(axis=0)
#        M = means / np.linalg.norm(means, axis=1, keepdims=True)
#        MMT = M @ M.T
#        NC2 = np.linalg.norm(MMT / np.linalg.norm(MMT, "fro") - (1 / np.sqrt(K - 1)) * (np.eye(K) - (1 / K) * np.ones((K, K))), "fro")
#
#        return np.log(float(NC1)), float(NC2)

