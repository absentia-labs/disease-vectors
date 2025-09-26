import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from .tokenization import GeneTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

# We expect SEQ_KEYS in every example, and their tensors should have the same length
SEQ_KEYS = {"input_ids", "token_type_ids", "attention_mask"}

def collate_fn_wrapper(tokenizer:GeneTokenizer):
    """
    Trainer expects a function that takes in just `examples`, so using a wrapper to pass in other arguments.
    """
    def collate_fn(examples: List[Dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Pads all examples in a batch to the same length, which is guaranteed to be a multiple of `pad_to_multiple_of`,
        if specified. Enforces strong assumptions about the examples; specifically, each example should take the form:
        {
            "input_ids": torch.Tensor of size (seq_len,)  # TODO: verify sizes
            "token_type_ids": torch.Tensor of size (seq_len,)
            "attention_mask": torch.Tensor of size (seq_len,)
            (and optionally) "labels": torch.Tensor of size (1,)
        }
        """
        all_keys = examples[0].keys()  # assuming all examples have the same keys, is verified later
        assert all(key in all_keys for key in SEQ_KEYS)  # TODO: create attention_mask if missing

        # `max_length` is the length of the longest sequence in the batch, or at most `self.tokenizer.config.max_length`
        max_length = max(len(example["input_ids"]) for example in examples)

        batch = {key: [] for key in all_keys}
        for example in examples:
            for key in all_keys:
                tensor = example[key]
                if len(tensor) < max_length:  # pad example to `max_length`
                    if key == "input_ids":  # add `pad_token`s
                        tensor = torch.cat([tensor, torch.full(
                            [max_length - len(tensor)],
                            fill_value=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),  # TODO: cache
                            dtype=torch.long,
                            device=tensor.device
                        )])
                    else:
                        tensor = torch.cat([tensor, torch.zeros(  # same code for token ids & attn
                            [max_length - len(tensor)],
                            dtype=torch.long,
                            device=tensor.device
                        )])
                batch[key].append(tensor)

        for key in all_keys:
            batch[key] = torch.stack(batch[key])

        return batch

    return collate_fn


def torch_mask_tokens(batch, tokenizer: GeneTokenizer, phenotype_mask_prob, genotype_mask_prob, rng=None):
    """
    Modified version of `DataCollatorForLanguageModeling.torch_mask_tokens`, signature is different though.
    Mutates `batch` in-place.

    Masking strategy for phenotype tokens:
        We mask phenotype tokens randomly with `phenotype_mask_prob`. To prepare the model for the inference scenario
        where some phenotypes are missing from the input (i.e. not even masked), we randomly "drop" masked tokens
        with `DROP_PHENOTYPE_PROB` (i.e. `phenotype_mask_prob * DROP_PHENOTYPE_PROB` of all phenotype tokens).
        To keep the indices of the input consistent, the dropping is done by replacing the input_id with the pad token.
    Masking strategy for gene tokens:
        We mask gene tokens randomly with `gene_mask_prob`.
        For each sequence in a batch, with probability`SWAP_GENE_IDS_PROB` we will randomly swap expressed genes
        with not expressed ones with `1 / num_bins / SWAP_GENE_IDS_PROB` probability.
        This is primarily designed for the two bin (not expressed/expressed) case,
        where the gene expression prediction would otherwise be trivial (always predict expressed bin).
        Note that the not expressed gene is selected uniformly from all genes not in the input,
        which is suboptimal because:
        1) Some genes are never expressed in any cell (~1k/58k genes in the current dataset, even more with truncation).
        2) On shorter sequence lengths, we could swap in an expressed gene that got truncated from the input.
        On `1 - SWAP_GENE_IDS_PROB` of sequences we perform no swapping,
        since that probably aligns closer with downstream tasks.
    """
    batch["labels"] = batch["input_ids"].clone()


    phenotypic_tokens_mask = tokenizer.get_phenotypic_tokens_mask(batch["token_type_ids"]) & (batch["input_ids"] != tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
    gene_tokens_mask = tokenizer.get_gene_tokens_mask(batch["token_type_ids"])

    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    # Gene MLM
    if genotype_mask_prob > 0:
        gene_prob_matrix = torch.zeros(batch["input_ids"].shape).masked_fill(gene_tokens_mask, value=genotype_mask_prob)
        gene_masked_indices = torch.bernoulli(gene_prob_matrix, generator=rng).bool()
        #batch["labels"][gene_masked_indices] = batch["input_ids"][gene_masked_indices]
        batch["input_ids"][gene_masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)


        # Replacing 1 / (num_bins + 1) with a random not expressed token, so that bin_0 is a label sometimes
        if tokenizer.config.sparse:
            rand_swap_prob = 1 / (2*(tokenizer.num_bins + 1))
            rand_swapped_indices = torch.bernoulli(gene_masked_indices.float() * rand_swap_prob, generator=rng).bool()
            for batch_idx in range(len(batch["input_ids"])):
                not_expr_genes = torch.ones(tokenizer.config.num_top_genes)
                not_expr_genes.scatter_(0, batch["token_type_ids"][batch_idx][gene_tokens_mask[batch_idx]] - tokenizer.gene_token_type_offset, torch.zeros_like(not_expr_genes))
                num_samples = rand_swapped_indices[batch_idx].long().sum().item()
                if num_samples > 0:  # very unlikely to have 0 samples
                    rand_not_expr_gene_indices = torch.multinomial(not_expr_genes, num_samples, replacement=False)
                    batch["labels"][batch_idx][rand_swapped_indices[batch_idx]] = tokenizer.convert_tokens_to_ids("bin_0")
                    batch["token_type_ids"][batch_idx][rand_swapped_indices[batch_idx]] = rand_not_expr_gene_indices + tokenizer.gene_token_type_offset

    # Phenotype MLM
    if phenotype_mask_prob > 0:
        phenotype_prob_matrix = torch.zeros(batch["labels"].shape).masked_fill(phenotypic_tokens_mask, value=phenotype_mask_prob)
        phenotype_masked_indices = torch.bernoulli(phenotype_prob_matrix, generator=rng).bool()
        #batch["labels"][phenotype_masked_indices] = batch["input_ids"][phenotype_masked_indices]
        batch["input_ids"][phenotype_masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # Commented because not consistent with masking, pad token is still attn matrix for these but not for others. 
        ## Replace DROP_PHENOTYPE_PROB of masked phenotypes with pad tokens. 
        #dropped_phenotype_masked_indices = torch.bernoulli(phenotype_masked_indices.float() * DROP_PHENOTYPE_PROB,
        #                                                   generator=rng).bool()
        #batch["labels"][dropped_phenotype_masked_indices] = -100
        #batch["input_ids"][dropped_phenotype_masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        #batch["token_type_ids"][dropped_phenotype_masked_indices] = 0


@dataclass
class DataCollatorForPhenotypicMLM(DataCollatorForLanguageModeling):
    tokenizer: GeneTokenizer
    genotype_mask_prob: float = 0. # initialised because of non-default errors
    phenotype_mask_prob: float = 0.

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Specifically adapted for IterableAnnDataset.
        Pads examples in a batch using `collate_fn` and additionally masks tokens (see `torch_mask_tokens`).
        """
        assert isinstance(examples[0], dict)
        batch = collate_fn_wrapper(self.tokenizer)(examples)

        torch_mask_tokens(batch, self.tokenizer, self.phenotype_mask_prob, self.genotype_mask_prob)

        batch['neural_updates'] = self.tokenizer.neural_updates.copy() # tough code
        self.tokenizer.neural_updates = {"token_values":[], "token_types": [],
                                        "token_value_str":[], "token_type_of_values":[]}
        batch['tokenizer'] = self.tokenizer
        return batch
