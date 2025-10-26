import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import transformers
from polygene.data_utils.tokenization import GeneTokenizer
from polygene.model.model import Polygene


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)

from typing import Callable

import torch
import torch.nn.functional as F
from anndata import AnnData
from transformers.modeling_outputs import SequenceClassifierOutput
from polygene.data_utils.tokenization import GeneTokenizer


def prepare_cell(cell: AnnData, tokenizer: GeneTokenizer, ) -> dict[str, torch.Tensor]:
    """
    Converts an h5ad cell to `input_ids`.

    Args:
        cell: AnnData object with n_obs = 1
        model_type: Expecting "mlm" or "cls"
        tokenizer: To encode cell into `input_ids` and `token_type_ids`
        label2id: Only required for model_type == "cls"
    """
    input_ids, token_type_ids, labels = tokenizer(cell)
    cell_data = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "str_labels": labels,
    }
    return cell_data


def test_batch(prepared_cells: dict[str, torch.Tensor], model: Polygene,
              data_collator: Callable) -> SequenceClassifierOutput:
    """
    Args:
        prepared_cell: Output of `prepare_cell`
        model: Model to perform inference with
        data_collator: Basically just needed to unsqueeze tensors into batch with one example
    """
    batched_cell = data_collator(prepared_cells)
    with torch.no_grad():
        output = model(**{key: val.to(model.device) for key, val in batched_cell.items()})

    return output


def phenotype_inference(cell_data: AnnData, phenotype_category: str, model: Polygene,
                        tokenizer: GeneTokenizer, data_collator: Callable, temperature: float = 1e-2, 
                        mask_other_phenotypes: bool = False, topk: int = 1) -> tuple:
    """
    Performing phenotype classification using an MLM model by masking the corresponding input token
    and returning the word with the highest predicted probability.

    Args:
        cell: AnnData object with n_obs = 1
        phenotype_category: the category to perform classification on; e.g. "tissue" or "cell_type"
        model: passed to `test_cell`
        tokenizer: passed to `prepare_cell`
        data_collator: passed to `test_cell`
        temperature: vary how confident the model is in it's predictions '<< 1' = deterministic
         '>> 1' = uniform sampling , can't imagine a use case but its cool
        mask_other_phenotypes: bool whether model can see other phenotypes to predict the phenotype_category
        topk: int return top K predictions.

    Returns:
        y: string of the label
        y_pred: string of the prediction
    """
    assert phenotype_category in tokenizer.phenotypic_types, f'Category must be one of {tokenizer.phenotypic_types}'
    phenotype_index = 1 + tokenizer.phenotypic_types.index(phenotype_category)  # offset by 1 because of 'special' token
    
    prepared_cells = []
    y = []
    for cell in cell_data:
        prepared_cell = prepare_cell(cell, tokenizer)
        prepared_cell["labels"] = prepared_cell['input_ids'].detach().clone()
        prepared_cell["input_ids"][phenotype_index] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        if mask_other_phenotypes: # Don't let our genotype see the other phenotypes to predict the masked one. 
            for pdx, phenotype in enumerate(tokenizer.phenotypic_types):
                if phenotype != phenotype_category: prepared_cell["attention_mask"][1 + pdx] = 0
        y.append(prepared_cell['str_labels'][phenotype_index])
        del prepared_cell['str_labels']
        prepared_cells.append(prepared_cell)
    
    output = test_batch(prepared_cells, model, data_collator) # (B, V)
    probabilities = F.softmax(output.logits[:, phenotype_index] / temperature, dim=-1)

    # Get the strings from flattened_tokens
    y_pred = [tokenizer.flattened_tokens[torch.distributions.Categorical(probability).sample().item()] for probability in probabilities]

    if topk > 1: # to compute Top@3 Accuracy and things like that for cell type and tasks with lots of classes, apparently widely accepted
        assert topk <= probabilities.shape[1], "topk less than vocab size"
        y_pred = [[tokenizer.flattened_tokens[idx] for idx in torch.topk(probability, topk).indices] for probability in probabilities]
    #if len(y) == 1: y, y_pred = y[0], y_pred[0]
    return y, y_pred

import pandas as pd
from tqdm import tqdm
from rapidfuzz.fuzz import partial_ratio
from polygene.data_utils.data_collators import collate_fn_wrapper
from anndata import read_h5ad

def accuracy_table(model, tokenizer, shard_path, phenotype_types = ['cell_type', 'sex', 'development_stage', 'tissue', 'disease'], batch_size=1, topk = 5, masking=False, n_cells=None):#:, once=True):
    shard  = read_h5ad(shard_path)[:n_cells]
    metric_names = ['Top@1', 'Top@1-Mask', f'Top@{topk}', f'Top@{topk}-Mask'] if masking else ['Top@1', f'Top@{topk}']
    hit_counts = {metric: {phenotype: 0 for phenotype in phenotype_types} for metric in metric_names}
    num_batches = (len(shard) + batch_size - 1) // batch_size
    collate = collate_fn_wrapper(tokenizer)

    match_threshold = .9
    with tqdm(total=len(phenotype_types) * num_batches * (2/(2-int(masking)))) as progress_bar:
        for phenotype in phenotype_types:
            for start_index in range(0, len(shard), batch_size):
                for mask in (True, False) if masking else (False,):
                    batch = shard[start_index:start_index + batch_size]
                    y, y_pred = phenotype_inference(
                        batch, phenotype, model, tokenizer, collate, temperature=1,
                        mask_other_phenotypes=False, topk=topk
                    )
                    suffix = '-Mask' if mask else ''
                    for true, prediction in zip(y, y_pred):
                        scores = [partial_ratio(true, candidate) / 100 for candidate in prediction]
                        hit_counts[f'Top@1{suffix}'][phenotype] += scores[0] >= match_threshold
                        hit_counts[f'Top@{topk}{suffix}'][phenotype] += any(score >= match_threshold for score in scores)
                    progress_bar.update()

    accuracy =  { metric: { phenotype: hit_counts[metric][phenotype] / len(shard) for phenotype in phenotype_types } for metric in metric_names }
    return pd.DataFrame(accuracy).T 