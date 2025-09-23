import re
import json
import torch
import warnings
import unicodedata 
import numpy as np
import scipy.sparse
from polygene.configs import BaseConfig

def _prepend_bin(arr):
    """
    Prepends 'bin_' to every element of a numpy array.
    E.g. [1, 1, 2, 3] -> ['bin_1', 'bin_1', 'bin_2', 'bin_3']
    """
    return np.char.add("bin_", arr.astype(str))

def normalise_str(phenotype_str: str) -> str:
    """Convert phenotype string to a standardized format.""" #chat gipidiy
    s = unicodedata.normalize('NFKD', phenotype_str).encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r"'s\b", "", s.lower())
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = '_'.join(s.split())
    return f"[{s}]"

class GeneTokenizer:
    """ Tokenizer for individual AnnData cells. """
    def __init__(self, config: BaseConfig):
        """
        Initializes the vocabulary. This includes:
            - special tokens (currently 4)
            - phenotype tokens, loaded from config.vocab_path
            - binned expression tokens (currently `len(config.bin_edges)`)
        """
        self.config = config

        with open(config.vocab_path, "r") as f:
            self.phenotypic_tokens_map = json.load(f)

        self.num_bins = len(self.config.bin_edges)

        self.flattened_tokens = ["[cls]", "[eos]", "[mask]", "[pad]"]  # special tokens
        self.cls_token, self.eos_token, self.mask_token, self.pad_token = self.flattened_tokens
        self.num_special_tokens = len(self.flattened_tokens)  # might be pointless since we've hardcoded 4 above anyway

        # sort for consistency, so that all tokenizers initialized on identical vocab have the same `token_to_id_map`
        self.phenotypic_types = sorted([phenotype for phenotype in self.config.obs_included_phenotypes if phenotype in self.phenotypic_tokens_map])
        assert self.phenotypic_types, f"Empty list of valid phenotypic types, select from {list(self.phenotypic_tokens_map.keys())}"
        for phenotypic_type in self.phenotypic_types:
            self.flattened_tokens.extend(self.phenotypic_tokens_map[phenotypic_type])

        # For clarity, the gene expression tokens will be: 'bin_0', 'bin_1', 'bin_2', ..., 'bin_n'
        # 'bin_0' corresponds to not expressed genes and should not be included in the input, but helpful for MLM
        self.flattened_tokens = [normalise_str(token) for token in self.flattened_tokens]
        self.flattened_tokens.extend(_prepend_bin(np.arange(self.num_bins + 1)).tolist())  # Important! - breaking backwards compatability by adding bin_0
        self.token_to_id_map = {token: i for i, token in enumerate(self.flattened_tokens)}

        # keep a genotype map because HVG ranks are inconsistent between datasets, oh lord
        with open("polygene/data_utils/vocab/gene_ranking_map.json", "r") as f: #TODO path as argument
            self.gene_type_id_map = json.load(f)
        
        # Tag to be manually changed for models to grow with unseen vocabulary
        self.flexible = False # During training do you want to be able to grow vocabularies and add new phenotypes
        self.neural_updates = {
                        "token_values":[], "token_types": [],
                        "token_value_str":[], "token_type_of_values":[]
                        }


    def __call__(self, cell) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Represents `cell` as a combination of `input_ids` and `token_type_ids`.

        E.g.
                           [CLS] [AGE_TOKEN] [CELL_TOKEN] ... [TISSUE_TOKEN] gene0 gene100 ... gene57991 [EOS]
        `input_ids` =      [0,   16,         32,          ... 256,           276,  275,    ... 276,      1]
        `token_type_ids` = [0,   1,          2,           ... 5,             6,    106,    ... 57992,    0]

        In the above example, only genes 5, 7, ... are expressed, so only these genes are included in the input.
        Note that special tokens share the same token_type_id (0),
        but each phenotype category has a distinct token_type_id.
        This distinction is important during MLM, because the model can only determine the phenotype category
        it is expected to predict through the token_type_id.

        The actual order and input ids in the example are arbitrary.
        In fact, I believe the genes are not always in ascending order.

        Returns:
            (input_ids, token_type_ids)
        """

        input_tokens = []  # Will have special tokens, phenotypic tokens, and gene expression tokens
        token_type_ids = []

        # [CLS]
        input_tokens.append(self.cls_token)
        token_type_ids.append(0)


        # e.g. [AGE_TOKEN] [CELL_TOKEN] ... [TISSUE_TOKEN]
        input_tokens.extend([normalise_str(cell.obs[phenotypic_type].item() + ("_" + phenotypic_type if cell.obs[phenotypic_type].item() == "unknown" else ""))
                              if phenotypic_type in cell.obs.columns else self.mask_token for phenotypic_type in self.phenotypic_types])
        
        # Distinct token_type_id for each phenotype token
        token_type_ids.extend([1 + self.phenotypic_types.index(phenotypic_type) for phenotypic_type in self.phenotypic_types])

        # e.g. gene0 gene100 ... gene57991
        bin_ids, gene_ids = self._bin_genes(cell.X, self.config.sparse)

        # Map to the correct HVG ranking in case not training on CellxGene
        gene_ensembl_ids  = cell.var.index[gene_ids]
        gene_ids = np.array([self.gene_type_id_map.setdefault(ensembl_id, -1) for ensembl_id in gene_ensembl_ids])
        known_genes = gene_ids != -1
        if (known_genes == False).sum():
            #warnings.warn(f"Unknown Ensembl Gene IDs: {gene_ensembl_ids[~known_genes]}") Warning commented bc theres actually a lot
            gene_ids, bin_ids = gene_ids[known_genes], bin_ids[known_genes] # Remove unknown Ensembl IDs
        o = np.argsort(gene_ids) # Sort genes so HVG are first
        gene_ids, bin_ids = gene_ids[o], bin_ids[o]

        input_tokens.extend(_prepend_bin(bin_ids))
        token_type_ids.extend(gene_ids + self.gene_token_type_offset)

        assert len(input_tokens) == len(token_type_ids)

        if len(input_tokens) > self.config.max_length: # Truncating to upperbound computational complexity of transformer O(S^2)
            input_tokens = input_tokens[:self.config.max_length]
            token_type_ids = token_type_ids[:self.config.max_length]

        invalid_tokens = self._check_valid_tokens(input_tokens)
        labels = input_tokens[:]
        if invalid_tokens:
            if self.flexible: # Grow network for out of vocabulary tokens 
                # get the phenotype category they are a part of
                invalid_token_type_ids = [token_type_ids[input_tokens.index(invalid_token)] for invalid_token in invalid_tokens]
                self.add_token_values(invalid_tokens, invalid_token_type_ids)
            else: 
                raise ValueError(f"Out of vocabulary tokens: {invalid_tokens}")

        return (torch.LongTensor(self.convert_tokens_to_ids(input_tokens)),
                 torch.LongTensor(token_type_ids),
                   labels)
    
    def _bin_genes(self, sparse_expr_arr: scipy.sparse.csr_matrix, sparse=True) -> tuple:
        """
        Args:
            sparse_expr_arr: the sparse matrix containing the gene expression levels for a single cell

        Returns: (bin_ids, gene_ids)
            bin_ids: the indices of gene expression bins, with
                bin_ids[i] corresponding to the expression level of gene gene_ids[i]
            gene_ids: the indices of the expressed genes

        Note that while `bin_ids` and `gene_ids` loosely correspond with the input_ids and token_type_ids respectively,
        they are both an offset away from the correct index in our vocabulary, so
        additional postprocessing is performed on both `bin_ids` and `gene_ids` in __call__().
        """
        # We avoid loading the dense gene expression array
        sparse_expr_arr = sparse_expr_arr.tocsr() # Other datasets have column sparse matrices
        gene_indices = sparse_expr_arr.indices  # corresponds to the indices of genes with nonzero expression
        gene_expressions = sparse_expr_arr.data  # corresponds to the gene expression values for these genes

        gene_expression_bins = np.digitize(gene_expressions, self.config.bin_edges)  # bin gene expression values
        if sparse:
            expressed_genes_mask = np.flatnonzero(gene_expression_bins) # remove bin_0
            return gene_expression_bins[expressed_genes_mask], gene_indices[expressed_genes_mask]
        return gene_expression_bins, gene_indices

    def _check_valid_tokens(self, tokens: str | list[str]) -> tuple[bool, list[str]]:
        """ 
        Checks if all tokens are valid (i.e. in the vocabulary).
        Returns a tuple of (is_valid, invalid_tokens).
        """
        if isinstance(tokens, str):
            tokens = [tokens]

        invalid_tokens = [token for token in tokens if token not in self.token_to_id_map]
        return invalid_tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id_map[token]  # will fail on unknown tokens

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """
        If tokens is a string, returns the integer index it corresponds to in the vocabulary.
        If tokens is a list of strings, returns the list of indices each string corresponds to in the vocabulary.
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        else:
            return list(map(self._convert_token_to_id, tokens))

    def get_phenotypic_tokens_mask(self, token_type_ids: torch.LongTensor) -> torch.Tensor:
        """
        Returns:
            A tensor of booleans: True for a phenotype token, False for a special/gene token.
        """
        return torch.ge(token_type_ids, 1) & torch.lt(token_type_ids, self.gene_token_type_offset)

    def get_gene_tokens_mask(self, token_type_ids: torch.LongTensor) -> torch.Tensor:
        """
        Returns:
            A tensor of booleans: True for a gene expression (bin_id) token, False for a special/phenotype token.
        """
        return torch.ge(token_type_ids, self.gene_token_type_offset)

    @property
    def vocab_size(self) -> int:
        """ Number of unique `input_ids` """
        return len(self.flattened_tokens)

    @property
    def type_vocab_size(self) -> int:
        """ Number of unique `token_type_ids` """
        # update if new genes got added
        # LL: new genes are ranked last, we don't know their HVG rank, so it will be low priority, we can include with the first bin value threshold
        return self.gene_token_type_offset + self.config.num_top_genes
    
    @property
    def gene_token_type_offset(self): # token_type_id for gene 0, after accounting for special and phenotype tokens
        return 1 + len(self.phenotypic_types)

    def add_phenotype(self, new_phenotypes=[]): # next step is to automatically detect the new phenotypes to learn
        non_duplicate_phenotypes =  list(set(new_phenotypes) - set(self.phenotypic_types))
        self.neural_updates['token_types'].extend([self.gene_token_type_offset + i for i in range(len(non_duplicate_phenotypes))]) # order of lines is important here
        self.phenotypic_types.extend(non_duplicate_phenotypes)

    def add_token_values(self, new_tokens_values=[], token_type_ids=[]): #token type ids of the new token values 
        #LL: probably need to refactor whole code with better variable names
        """
        When you add token values - you need to add it to
        1. self.flattened_tokens
        2. self.token_to_id_map
        3. self.phenotypic_tokens_map
        """
        for i, new_token in enumerate(new_tokens_values):
            self.token_to_id_map[new_token] = self.vocab_size
            self.neural_updates['token_values'].append(self.vocab_size)
            self.neural_updates['token_value_str'].append(new_token) #order of lines is important
            self.neural_updates['token_type_of_values'].append(token_type_ids[i])
            self.flattened_tokens.append(new_token)
            self.phenotypic_tokens_map.setdefault(self.phenotypic_types[token_type_ids[i]-1], []).append(new_token)

    def update_from_model_memory(self, updates_memory: dict):
         """
         config.updates_memory = {
                "token_values": [],
                "token_value_str": [],
                "token_type_of_values": [],
            }
         """
         token_value_vocab = set(self.flattened_tokens) # O(n)
         for token_type, token_id, token_str in zip(updates_memory['token_type_of_values'],
                                                    updates_memory['token_values'],
                                                    updates_memory['token_value_str']):
             
            if token_str in token_value_vocab: continue #ignore if already in vocabulary
            self.token_to_id_map[token_str] = token_id
            self.flattened_tokens.append(token_str)
            self.phenotypic_tokens_map.setdefault(self.phenotypic_types[token_type-1], []).append(token_str)    

    def sync(self, updates_memory, num): # num = 1082 essentially
        self.flattened_tokens = self.flattened_tokens[:num]
        for token_type, token_id, token_str in zip(updates_memory['token_type_of_values'],
                                        updates_memory['token_values'],
                                        updates_memory['token_value_str']):
            self.token_to_id_map[token_str] = token_id
            self.flattened_tokens.append(token_str)
