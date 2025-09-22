import argparse
import ast

import torch
import numpy as np
from typing import TypedDict, Tuple, Optional
from captum.attr import LayerIntegratedGradients, LayerDeepLift
from polygene.data_utils.tokenization import GeneTokenizer
from polygene.model.model import Polygene
import pandas as pd
import pickle
from braceexpand import braceexpand
import functools, operator
from anndata import read_h5ad, concat
from polygene.eval.metrics import prepare_cell#, test_batch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from captum.attr import KernelShap
from collections import defaultdict
from itertools import product
import os


class GeneAttribution:
    """
    Attribution Analysis for Polygene: 
    - Loading for cells of specific phenotype
    - 2D Visualization of Embeddings
    - Computing Gene Attributions

    """

    def __init__(
        self,
        model: Polygene,
        tokenizer: GeneTokenizer,
        layer: torch.nn.Module = None,
    ):
        """
        Class Initialization

        Args:
            - model (Polygene): a model for  the attribution analysis
            - layer (torch.nn.Module): a target embedding for attribution method (see more at https://captum.ai/tutorials/Bert_SQUAD_Interpret)
            - tokenizer (GeneTokenizer): a tokenizer associated with the model
        """
        self.model = model
        self.device = model.device
        self.layer = model.embeddings if layer is None else layer

        self.tokenizer = tokenizer

        self._target, self._baseline = None, None

        self.attributions, self.delta = None, None
        self.cells = None
        self.cell_attributions = None
        self.embeddings = None
        self.mask_phenotype_category = None
        self.pruning_results = None


    def get_phenotype_cells(self,  paths: list, n: int = 1, match_other_phenotypes=False, get_info=False, **phenotypes):
        """
        Method to get N cells with a specific phenotype

        n: int number of cells to get
        paths: list of braceexpandable paths to h5ad files for reading
        phenotypes: dict type of sex=['M'], tissue=['blood'] etc
        """
        processed_paths = [path for bracepath in paths for path in braceexpand(bracepath)]
        cells = []

        combination_types = list(phenotypes)[:]
        required = {phenotype_combination: n for phenotype_combination in product(*phenotypes.values())}
        for pdx, path in enumerate(processed_paths):
            if not len(required):
                break
            anndata = read_h5ad(path)
            anndata.obs.reset_index(drop=True, inplace=True)
            anndata.obs.index = anndata.obs.index.values.astype(str)
            if get_info:  print(anndata.obs[list(phenotypes.keys())].value_counts())
            # phenotypes is a dict where sex = M, tissue = blood etc
            mask = functools.reduce(operator.and_, (anndata.obs[phenotypic_type].isin(value) for phenotypic_type, value in phenotypes.items() if phenotypic_type in anndata.obs.columns))
            filtered_anndata = anndata[mask]

            if match_other_phenotypes and pdx == 0:
                index_of_sample = filtered_anndata.obs.value_counts().reset_index()[[phenotypic_type for phenotypic_type in self.tokenizer.phenotypic_types if phenotypic_type in filtered_anndata.obs.columns] + [0]].max().index[0]
                for col in [column for column in filtered_anndata.obs.iloc[index_of_sample:index_of_sample+1].columns if column not in phenotypes]:
                    phenotypes[col] =  [filtered_anndata.obs.iloc[index_of_sample][col]]
                mask = functools.reduce(operator.and_, (anndata.obs[phenotypic_type].isin(value) for phenotypic_type, value in phenotypes.items()))
                filtered_anndata = anndata[mask]
            
            for combination in list(required.keys()): # select cells from specific phenotype combo
                mask = functools.reduce(operator.and_, (filtered_anndata.obs[phenotypic_type] == value for phenotypic_type, value in zip(combination_types,  combination)))
                selected_cells = filtered_anndata[mask][:required[combination]].copy()
                cells.append(selected_cells)
                required[combination] -= len(selected_cells)
                if (required[combination] < 1): del required[combination]
            del anndata, filtered_anndata

        self.cells = concat(cells)
        print(f'Loaded {len(self.cells)} cells\n', self.cells.obs[list(phenotypes.keys())].value_counts())
        
        return self.cells
    

    def get_embeddings(self, phenotype, mask=False, mask_all=False):
        """
        Get model embeddings of a specific phenotypic type before the classification head

        phenotype: str, phenotype in list from tokenizer
        """
        phenotypic_type_index = 1 + self.tokenizer.phenotypic_types.index(phenotype)
        embeddings, predictions, labels = ([] for _ in range(3))

        for cell in tqdm(self.cells, "Embeddings"):
            cell_dict = prepare_cell(cell, self.tokenizer) # gets it in the right ModelInput format
            if mask:
                #cell_dict['input_ids'][np.arange(1, 1+len(self.tokenizer.phenotypic_types))] = 2 # id of mask token
                cell_dict['input_ids'][phenotypic_type_index] = 2 # id of mask token
            if mask_all:
                cell_dict['input_ids'][np.arange(1, 1+len(self.tokenizer.phenotypic_types))] = 2 # id of mask token
            with torch.no_grad():
                output = self.model(**{key: val.to(self.model.device).unsqueeze(0) for key, val in cell_dict.items() if key != 'str_labels'})

            encoder_output = output.hidden_states # tensor shape (B, S, D)
            embeddings.append(encoder_output[:, phenotypic_type_index, :])
            labels.append(cell_dict['str_labels'][phenotypic_type_index])
            predictions.append(self.tokenizer.flattened_tokens[output.logits.argmax(dim=-1).squeeze()[phenotypic_type_index]])
        
        self.embeddings = (torch.cat(embeddings).detach().cpu().numpy(), np.array(predictions), np.array(labels))
        return self.embeddings


    def plot_umap(self, reduction='umap', n_neighbors=30, min_dist=0.3, random_state=3, figure_size=(8,6),
                   marker_size=40, perplexity=30, save_path=None, alpha=0.6, palette="hls", lw=0, spread=1., label=None,
                   fontsize=12, fontweight='500', color_labels=None, points=None):
        """
        Plot embeddings using UMAP, t-SNE, or PCA with labeled points

        Parameters:
        - reduction: 'umap', 'tsne', or 'pca'
        - n_neighbors (UMAP): conserve local [5-15], global [30-100] structure
        - min_dist (UMAP): conserve local [0.001-0.1], global [0.3-0.5] structure
        - perplexity (t-SNE): conserve local [5-20], global [30-50] structure
        """
        embeddings, _, labels = self.embeddings

        if color_labels is not None: labels = color_labels

        if reduction == 'pca':
            reducer = PCA(n_components=2, random_state=random_state)
        elif reduction == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        else:
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, spread=spread)

        labels = pd.Categorical(labels)
        low_dim_embeddings = reducer.fit_transform(embeddings) if points is None else points
        df = pd.DataFrame({'UMAP1': low_dim_embeddings[:,0], 'UMAP2': low_dim_embeddings[:,1], 'Phenotype': labels})
        palette = sns.color_palette(palette, n_colors=len(labels.categories))
        
        fig, ax = plt.subplots(figsize=figure_size)
        sns.scatterplot(data=df, x='UMAP1', y='UMAP2', hue='Phenotype', palette=palette, s=marker_size, alpha=alpha, ax=ax, linewidth=lw)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([]); ax.set_yticks([])
        if label is None: label = reduction.upper()
        x_min, x_max = df['UMAP1'].min(), df['UMAP1'].max()
        y_min, y_max = df['UMAP2'].min(), df['UMAP2'].max()

        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.01 * (y_max - y_min)
        ax.text(x_min-x_margin, y_min, f'{label}1', ha='left', va='top', fontsize=fontsize, fontweight=fontweight)
        ax.text(x_min-1.1*x_margin, y_min+y_margin, f'{label}2', ha='left', va='bottom', rotation=90, fontsize=fontsize, fontweight=fontweight)
        ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
        for legend_handle in ax.legend_.legendHandles:
            legend_handle.set_alpha(1)
        if save_path is not None: plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


    def compute_all_attributions(self, mask_phenotype_category, baseline=None, n_steps=10, method='ig', mask_all=False, gene_intersection=False): # don't have implementation for better baseline than None
        self.method = method
        self._eval_method = self.initialize_attribution_method(gene_intersection)

        cell_attributions = []
        for cell in tqdm(self.cells, "Attributions"): 
            cell_dict = prepare_cell(cell, self.tokenizer)
            del cell_dict['str_labels']
            self._target = cell_dict
            if mask_all: self._target['input_ids'][np.arange(1, 1+len(self.tokenizer.phenotypic_types))] = 2 
            self._baseline = self._construct_blank_instance() if baseline is None else baseline
            attributions, delta = self.eval_attribution_method(masked_phenotype_category=mask_phenotype_category, n_steps=n_steps)
            cell_attributions.append((cell_dict["token_type_ids"].detach().cpu().numpy().squeeze(), attributions.detach().cpu().numpy(), delta))

        self.cell_attributions = np.array(cell_attributions, dtype=object)
        self.mask_phenotype_category = mask_phenotype_category
        return self.cell_attributions


    def summary(self, n_genes = 10):
        assert self.cell_attributions is not None, 'get the cell attributions'
        df = self.cells.obs
        vc = df[self.mask_phenotype_category].value_counts()
        print("Value counts\n",vc)

        gene_names = list(self.tokenizer.gene_type_id_map.keys())
        phenotype_global_attributions = []
        for phenotype_value in vc.index:
            mask = df[self.mask_phenotype_category] == phenotype_value
            phenotype_attributions = self.cell_attributions[mask]
            phenotype_attributions_df = pd.DataFrame( [dict(zip(gene_identifier_array, attributions))
                                                    for gene_identifier_array, attributions, _ in phenotype_attributions]
                                                    ).fillna(0)
            phenotype_attributions_df.columns = pd.Series(phenotype_attributions_df.columns.tolist()).apply(lambda gid: gene_names[gid - self.tokenizer.gene_token_type_offset]
                                                                                                                if gid >= self.tokenizer.gene_token_type_offset else gid).tolist()
            consistency = phenotype_attributions_df.apply(
                    lambda col: col[col.ne(0)].map(np.sign)
                                    .value_counts(normalize=True)
                                    .max() * 100
                )
            top_genes = phenotype_attributions_df.mean(axis=0) \
                                                    .sort_values(ascending=False) \
                                                    .iloc[:n_genes]
            convergence_error = np.mean([abs(err) for *_, err in phenotype_attributions])
            global_attribution = pd.DataFrame({
                f"{phenotype_value}_genes": ['Convergence Error'] + top_genes.index.tolist(),
                f"{phenotype_value}attributions": [convergence_error] + top_genes.values.tolist(),
                f"{phenotype_value}_consistency": [np.nan] + consistency[top_genes.index].tolist()
            })
            phenotype_global_attributions.append(global_attribution)

        return pd.concat(phenotype_global_attributions, axis=1)

    def initialize_attribution_method(self, gene_intersection=False) -> LayerIntegratedGradients | LayerDeepLift:
        """
        Return and initialize the Captum attribution methods corresponding with
        the method type.

        Note: As of now, DeepLift can only take a torch.nn module as an input.
        Hence, we need to define a torch.nn wrapper.
        """
        if self.method == "ig":
            return LayerIntegratedGradients(self.forward_pass, self.layer)
        elif self.method == "dl":
            wrapper = ModelWrapper(self.model, self)
            return DeepLiftWrapper(wrapper, self.layer)
        elif self.method == "sv":
            return ShapleyValue(self.forward_pass, self.tokenizer, None)
        elif self.method == "sv2":
            return KernelShapWrapper(self.forward_pass)
    
    def forward_pass(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_phenotype_category: str,
        target_phen: Optional[str] = "max",
        return_prediction: bool = False,
        topk = 3,
    ) -> float | Tuple[float, int]:
        """
        Return the model's prediction (if specified) and its associated probability.

        Args:
            - input_ids (torch.Tensor): a tensor of an input's input_ids
            - token_type_ids (torch.Tensor): a tensor of an input's token_type_ids
            - attention_mask (torch.Tensor): a tensor of an input's attention_mask
            - masked_phenotype_category (str): a phenotypic category that the model predicts (this phenotype will be masked)
            - target_phen (str): this optional parameter indicates probability from which phenotype to be returned from the model.
                               if not specified, it will return the probability of the most probable labels.
                               if it is "max", we will return the highest probability across all the phenotypes.
                               otherwise, it will return the probability associated with that particular phenotype
            - return_prediction (bool): a boolean indicating whether to return the predicted label.

        Description:
            If gene_blending has not been called yet, this function will first blend the token_type_ids.
            Given an input to the model, masked_phenotype_category is masked and the model is tasked
            with predicting it. If phen is specified, the model will return the probability predicting phen.
            Otherwise, it will give the most probable prediction.
        """

        category_idx = 1 + self.tokenizer.phenotypic_types.index(
            masked_phenotype_category
        )
        masked_input_ids = input_ids.clone()
        # Perform masking on the cloned tensor to avoid in-place modification

        masked_input_ids[:, category_idx] = 2

        output = self.model(
            masked_input_ids.to(self.device),
            token_type_ids=token_type_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        ).logits
        normalized = torch.softmax(output[:, category_idx, :], dim=-1)

        # Get attribution
        if target_phen == "max":
            value, pred = torch.max(normalized, dim=-1)

        if return_prediction:
            return value, torch.topk(normalized, topk).indices
        return value

    @torch.no_grad()
    def eval_attribution_method(
        self,
        masked_phenotype_category: str,
        internal_batch_size: Optional[int] = 1,
        target_phen: Optional[str] = "max",
        n_steps: Optional[int] = 10,
    ) -> (
        Tuple[torch.tensor, int, torch.tensor, int]
        | Tuple[torch.tensor, int, None, None]
    ):
        """
        Return attribution values and convergence error calculated by an attribution analysis method
        (Integrated Gradient/DeepLift). If self.two_way is true, then it calculates both target -> baseline
        and baseline -> target attributions and stores them in its attributes.

        Args:
            - internal_batch_size (int): a bsz for the attribuion method
            - masked_phenotype_category (str): a phenotypic category that the model predicts (this phenotype will be masked)
            - n_steps (int): a number of steps for Integrated Gradients method (it is not required for DeepLift)
            - target_phen (str): this optional parameter indicates probability from which phenotype to be returned from the model.
                               if it is "max", we will return the highest probability across all the phenotypes.
        """
        for k in self._target:
            for _input in [self._target, self._baseline]:
                _input[k] = _input[k].unsqueeze(0)
        
        attributions, delta = self._eval_method.attribute(
            inputs=(
                self._target["input_ids"],
                self._target["token_type_ids"],
                self._target["attention_mask"],
            ),
            baselines=(
                self._baseline["input_ids"],
                self._baseline["token_type_ids"],
                self._baseline["attention_mask"],
            ),
            internal_batch_size=internal_batch_size,
            additional_forward_args=(masked_phenotype_category, target_phen),
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        # Normalize values: ok
        if self.method != "sv2":
            attributions_by_token = attributions.sum(dim=-1).squeeze(0)
            attributions_by_token = attributions_by_token / torch.norm(
                attributions_by_token
            )
        else:
            attributions_by_token = attributions.squeeze(0)

        self.attributions, self.delta = (
            attributions_by_token,
            delta.item(),
        )
        return (
            self.attributions,
            self.delta,
        )


    def gene_pruning_experiment(self, pruning_steps=100, subset_of_phenotype_values=[], global_attribution=False, mask=True, positive_attributions=True, results_path="", topk=1):
        """
        yucky function for the moment to get Big's banger pruning plot
        """
        assert self.mask_phenotype_category is not None, "need a category to verify"

        phenotypes = [phenotype for phenotype in self.cells.obs[self.mask_phenotype_category].value_counts().index if not subset_of_phenotype_values or phenotype in subset_of_phenotype_values]
        results = defaultdict(nest_dict)
        pruning_range = np.linspace(0, 1, num=pruning_steps + 1)

        for idx, cell in enumerate(tqdm(self.cells, "Pruning experiment")):
            if cell.obs[self.mask_phenotype_category].values[0] not in phenotypes: continue
            cell_dict = prepare_cell(cell, self.tokenizer)
            true = cell_dict['input_ids'][1 + self.tokenizer.phenotypic_types.index(self.mask_phenotype_category)]
            del cell_dict['str_labels']
            self._target = cell_dict
            if mask: self._target['input_ids'][np.arange(1, 1+len(self.tokenizer.phenotypic_types))] = 2 
            self._target = {k:v.unsqueeze(0) for k,v in cell_dict.items()}
            self.attributions = torch.Tensor(self.cell_attributions[idx][1])
            if positive_attributions: self.attributions = torch.clamp(self.attributions, min=0)

            for prune_ratio in pruning_range:
                x_attr, x_random = self.gene_pruning(prune_ratio), self.gene_pruning(prune_ratio, heuristic="random")
                logits_attr, pred_attr = self.forward_pass(**x_attr, masked_phenotype_category=self.mask_phenotype_category, return_prediction=True, topk=topk)
                logits_rand, pred_rand = self.forward_pass(**x_random, masked_phenotype_category=self.mask_phenotype_category, return_prediction=True, topk=topk)
                results[cell.obs[self.mask_phenotype_category].values[0]][prune_ratio].append([logits_attr.detach().cpu().item(), int(true in pred_attr), logits_rand.detach().cpu().item(), int(true in pred_rand)])
        
        self.pruning_results = results
        pd.to_pickle(results, f"{results_path}pruning_results-{self.mask_phenotype_category}.pkl")
        return results

    def gene_pruning(
        self,
        prune_ratio: int,
        heuristic: str = "attributions",
    ):
        """
        Gene Pruning Algorithm
        """
        
        if heuristic == "attributions":
            func = identity_importance
        elif heuristic == "random":
            func = random_importance
        else:
            raise Exception("Invalid heuristic")

        # Only prune genotypes not phenotypes
        offset = len(self.tokenizer.phenotypic_types) + 1
        num_genes = (
            self._target['input_ids'].shape[1] - offset - 1
        )  # remove heading (CLS + phenptypes) and [EOS]
        remaining_genes = int((1 - prune_ratio) * num_genes)
    
        importance = func(
            self.attributions[offset:], 
        )

        kept_gene_indices = torch.topk(importance, remaining_genes, largest=False, sorted=False).indices.detach().cpu().numpy()

        # Prune the blended genes
        pruned_target = {k:v[:, np.concatenate([np.arange(offset), kept_gene_indices + offset])] for k,v in self._target.items()}
        return pruned_target
    
    def _construct_blank_instance(self):
        mask = torch.arange(
            len(self.tokenizer.phenotypic_types) + 1
        )  # keep format and phenotypes
        target = self._target
        number_of_padding_tokens = len(target['input_ids']) - len(mask) - 1
        pad_id = self.tokenizer.flattened_tokens.index('[pad]')
        return {
            "input_ids": torch.cat(
                (target["input_ids"][mask], torch.tensor([pad_id for _ in range(number_of_padding_tokens)], device=target['input_ids'].device), target["input_ids"][[-1]])
            ),
            "token_type_ids": torch.cat(
                (target["token_type_ids"][mask],  torch.tensor([0 for _ in range(number_of_padding_tokens)], device=target['input_ids'].device), target["token_type_ids"][[-1]])
            ),
            "attention_mask": torch.cat(
                (target["attention_mask"][mask], torch.tensor([0 for _ in range(number_of_padding_tokens)], device=target['input_ids'].device), target["attention_mask"][[-1]])
            ),
        }

    def save(self, path="analysis.pkl"):
        backups = self.model, self.tokenizer, self._eval_method, self.cells
        self.model = self.tokenizer = self._eval_method = self.cells = None
        try:
            with open(path, "wb") as file_handle:
                pickle.dump(self, file_handle)
        finally:
            self.model, self.tokenizer, self._eval_method, self.cells = backups

    @classmethod
    def load(cls, model, tokenizer, path="analysis.pkl"):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        obj.model = model
        obj.tokenizer = tokenizer
        obj.initialize_attribution_method()
        return obj

class DeepLiftWrapper:
    """
    DeepLIFT wrapper
    """
    def __init__(self, model, layer):
        self.deeplift = LayerDeepLift(model, layer, multiply_by_inputs=True)

    def attribute(
        self,
        inputs,
        baselines,
        additional_forward_args=None,
        return_convergence_delta=False,
        n_steps=None,
        internal_batch_size=None,
    ):
        attributions, delta = self.deeplift.attribute(
            inputs=inputs,
            baselines=baselines,             
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
        )
        return attributions, delta
    
class ModelWrapper(torch.nn.Module):
    """
    Model Wrapper for DeepLIFT, since it only takes nn.Module
    """

    def __init__(
        self, model, attribution_analysis_instance: GeneAttribution
    ) -> None:
        super().__init__()
        self.model = model
        self.attribution_analysis_instance = attribution_analysis_instance

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_phenotype_category: str,
        target_phen: Optional[str],
    ):
        return self.attribution_analysis_instance.forward_pass(
            input_ids,
            token_type_ids,
            attention_mask,
            masked_phenotype_category,
            target_phen,
        )

class KernelShapWrapper:
    """
    shapley values good
    """
    def __init__(self, forward_func, n_samples: int = 200):
        self._ks = KernelShap(forward_func)
        self.n_samples = n_samples

    def attribute(
        self,
        inputs,
        baselines,
        internal_batch_size=None,
        additional_forward_args=None,
        n_steps=None,
        return_convergence_delta=False,
    ):
        masked_phenotype_category, target_phen = additional_forward_args
        attributions = self._ks.attribute(
            inputs=inputs,# inputs IDs only
            baselines=baselines,             
            additional_forward_args=(masked_phenotype_category, target_phen, ),
            n_samples=self.n_samples,
            return_input_shape=False,
            feature_mask=tuple(torch.arange(0, inputs[0].shape[-1]).unsqueeze(0) for _ in range(3))
        )
        delta = torch.tensor(0.) # Shapley doesnt have a delta
        return attributions, delta

class ShapleyValue:
    def __init__(self, forward_func, tokenizer:GeneTokenizer, intersection_genes=None): # list of token type ids
        self.forward_func = forward_func
        self.tokenizer= tokenizer
        #self.intersection_genes=set(intersection_genes.detach().cpu().tolist())

    def attribute(
        self,
        inputs,
        baselines,
        internal_batch_size=None,
        additional_forward_args=None,
        n_steps=None,
        return_convergence_delta=False,
    ):
        coalition_baseline = self.forward_func(*inputs, *additional_forward_args)
        feature_attributions = [torch.tensor([0], device=coalition_baseline.device)] * self.tokenizer.gene_token_type_offset
        for i in range(self.tokenizer.gene_token_type_offset, len(inputs[0].squeeze())):
            masked_inputs = tuple(tensor.clone() for tensor in inputs)
            masked_inputs[0][0, i] = 2
            masked_inputs[1][0, i] = 0
            coalition_value = self.forward_func(*masked_inputs, *additional_forward_args)
            feature_attributions.append(coalition_baseline - coalition_value)
        attribution_tensor = torch.cat(feature_attributions)
        
        return attribution_tensor, (coalition_baseline - attribution_tensor.sum())

def nest_dict():
    return defaultdict(list)

def identity_importance(x):
    return x

def random_importance(attr1: torch.tensor):
    return torch.randperm(attr1.size(0), device=attr1.device)
