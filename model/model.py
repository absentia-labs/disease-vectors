"""
Essentially an adaptation of BERT where the positional embeddings are removed.
However, this is implemented by overriding DistilBERT classes because only DistilBERT supports Flash Attention.
Hence, there is significantly more code because vanilla DistilBERT has no `token_type_ids`.
"""

from typing import Optional, Tuple, Union

import torch
import numpy as np
import transformers
from torch import nn
from transformers.activations import get_activation
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils import logging
from polygene.data_utils.tokenization import GeneTokenizer
logger = logging.get_logger(__name__)


class PositionlessEmbeddings(nn.Module):
    """
    Copied from `BertEmbeddings`, not DistilBERT embeddings because DistilBERT does not use `token_type_ids`.
    Differences from `BertEmbeddings` are:
        - `position_embeddings` are removed.
        - registered_buffers are removed because we assume `token_type_embeddings` are always passed in

    An expressed gene is the sum of:
        the "word_embedding", which represents the expression level (i.e. bin number)
        the "token_type_embedding", which represents the specific gene

    Therefore, in our case we will likely have significantly more distinct token_types (genes) than words (bins).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_value_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None: return inputs_embeds
        inputs_embeds = self.token_value_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        

class  MultiPredictionHead(nn.Module):
    def __init__(self, config, included_phenotypes, phenotypic_token_map, n_bins, loss):
        super().__init__()
        self.loss = loss
        self.n_bins = n_bins
        self.included_phenotypes = included_phenotypes
        self.phenotypic_token_map = phenotypic_token_map
        self.obs_prediction_heads = nn.ModuleList([nn.Linear(config.dim, len(phenotypic_token_map[phenotype])) for phenotype in included_phenotypes])

        self.genotype_prediction_head =  nn.Linear(config.dim, n_bins)
        self.phenotype_offset_in_vocab = np.cumsum([4] + [len(phenotypic_token_map[phenotype]) for phenotype in included_phenotypes])

    def forward(self, x, y=None):
        # x (B, S, E) and y (B, S,)
        classification_token, genotype_tokens = x[:, 0, :], x[:, 1+len(self.included_phenotypes):, :]
        obs_y_pred = [head(classification_token) for head in self.obs_prediction_heads] # (P, B, P_i)
        genotype_y_pred = self.genotype_prediction_head(genotype_tokens) # (B, G, BINS)
        
        total_loss = None
        if y is not None:
            phenotype_loss = sum([self.loss(obs_y_pred[idx], (y[:, 1+idx] - self.phenotype_offset_in_vocab[idx]).clamp(min=-100)) for idx in range(len(self.included_phenotypes))])
            genotype_loss = self.loss(genotype_y_pred.view(-1, self.n_bins), (y[:, 1+len(self.included_phenotypes):] - self.phenotype_offset_in_vocab[-1]).clamp(min=-100).view(-1))
            total_loss = 0.5 * phenotype_loss/len(self.included_phenotypes) + 0.5 * genotype_loss
        
        reconstruct_x = x.new_zeros(x.shape[0], x.shape[1], self.phenotype_offset_in_vocab[-1] + self.n_bins)
        reconstruct_x[:, 1+len(self.included_phenotypes):, -self.n_bins:] = genotype_y_pred
        for idx, phenotype_pred in enumerate(obs_y_pred):
            reconstruct_x[:, 1+idx, self.phenotype_offset_in_vocab[idx]:self.phenotype_offset_in_vocab[idx+1]] = phenotype_pred
        return reconstruct_x, total_loss


class Polygene(transformers.DistilBertPreTrainedModel):
    """
    Equivalent to `DistilBertModel` but with `PositionlessEmbeddings`
    and allowing `token_type_ids` argument in `forward()`.
    """
    def __init__(self, config: transformers.PretrainedConfig):
        """
        Only modification is using `PositionlessEmbeddings`.
        """
        super().__init__(config)

        self.mlm_loss_fct = nn.CrossEntropyLoss(label_smoothing=0.25)


        if not hasattr(config, "updates_memory") or config.updates_memory is None: # Load in the checkpoint version
            config.updates_memory = {
                "token_values": [],
                "token_value_str": [],
                "token_type_of_values": [],
            }
        
        self.embeddings = PositionlessEmbeddings(config)  # changed to `PositionlessEmbeddings`
        self.transformer = transformers.models.distilbert.modeling_distilbert.Transformer(config)  # encoder
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        if not self.config.classification_token:
            if not hasattr(self.config, "head_hidden_layers"): # error handling for different versions of the class
                self.config.head_hidden_layers = 1
            head_layers = [layer for _ in range(self.config.head_hidden_layers) for layer in (nn.Linear(config.dim, config.dim), get_activation(config.activation))]
            self.prediction_head = nn.Sequential(*(head_layers + [
                nn.LayerNorm(config.dim, eps=1e-12),
                nn.Linear(config.dim, config.vocab_size),
            ]))
        else:
            self.prediction_head = MultiPredictionHead(config, config.obs_included_phenotypes, config.phenotypic_tokens_map, config.n_bins, self.mlm_loss_fct)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        neural_updates: Optional[dict] = None,
        tokenizer: Optional[GeneTokenizer] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        # still a little voodoo
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers) # Prepare head mask if needed

        if self.training and any(neural_updates[n] for n in neural_updates):
            self.update_network(neural_updates, input_ids, tokenizer)

        if not self.training and tokenizer is not None and tokenizer.vocab_size < self.config.vocab_size:
            tokenizer.sync(self.config.updates_memory, num=self.config.vocab_size - len(self.config.updates_memory['token_value_str']))

        embeddings = self.embeddings( # (B, S, D)
            input_ids=input_ids,
            token_type_ids=token_type_ids, 
            inputs_embeds=inputs_embeds
        )  if inputs_embeds is None else inputs_embeds

        distilbert_output = self.transformer( # encoder-only transformer, don't panic
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state_embeddings = distilbert_output[0] / (torch.linalg.norm(distilbert_output[0],dim=-1, keepdim=True) + 1e-12) # (B, S, D)

        if not self.config.classification_token:
            prediction_logits = self.prediction_head(hidden_state_embeddings) # (B, S, V), V for vocabulary size
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1)) if labels is not None else None
        else:
            prediction_logits, mlm_loss = self.prediction_head(hidden_state_embeddings, labels)

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=hidden_state_embeddings,
            attentions=distilbert_output.attentions,
        )
    
    def update_network(self, neural_updates: dict, input_ids, tokenizer):
        """
        Method to update embeddings and prediction head shape, hopefully without breaking the model

        neural_updates: dict
        neural_updates = {"token_values":[], "token_types": [],
                               "token_value_str":[], "token_type_of_values":[]}
        """
        device = self.embeddings.token_value_embeddings.weight.device
        for idx, token_value_str in enumerate(neural_updates['token_value_str'][:]): 
            if token_value_str in self.config.updates_memory['token_value_str']: 
                if neural_updates['token_values'][idx] != self.config.updates_memory['token_values'][idx]:
                    input_ids[input_ids == neural_updates['token_values'][idx]] = self.config.updates_memory['token_values'][idx]
                    tokenizer.sync(self.config.updates_memory, num=self.config.vocab_size - len(self.config.updates_memory['token_value_str']))

                neural_updates['token_values'].pop(0)
                neural_updates['token_value_str'].pop(0)
                neural_updates['token_type_of_values'].pop(0)
        
        if neural_updates['token_values']: # token values in current implementation are added at the end of the vocabulary
            # first grow the token value embedding lookup
            vocab, dim = self.embeddings.token_value_embeddings.weight.shape 
            new_embeddings = nn.Embedding(vocab + len(neural_updates['token_values']), dim)
            with torch.no_grad():
                new_embeddings.weight[:vocab] = self.embeddings.token_value_embeddings.weight
            self.embeddings.token_value_embeddings = new_embeddings.to(device)
            
            # do the same for the prediction head
            classifier = self.prediction_head[-1]
            new_classifier = nn.Linear(dim, vocab + len(neural_updates['token_values']))
            with torch.no_grad():
                new_classifier.weight[:vocab] = classifier.weight
                new_classifier.bias[:vocab] = classifier.bias
            self.prediction_head[-1] = new_classifier.to(device)

            #if neural_updates['token_values'][0] <= self.config.vocab_size:
            #    neural_updates['token_values'] = [self.config.vocab_size + i for i in range(len(neural_updates['token_values']))]
            #    input_ids[input_ids == '']
            self.config.vocab_size += len(neural_updates['token_values'])
            self.config.updates_memory['token_values'].extend(neural_updates['token_values'])
            self.config.updates_memory['token_value_str'].extend(neural_updates['token_value_str'])
            self.config.updates_memory['token_type_of_values'].extend(neural_updates['token_type_of_values'])

        if neural_updates['token_types']: # token types are added in the middle (and at the end for genes but currently not supported)
            # insert lines
            insert_idx = neural_updates['token_types'][0]
            n_types = len(neural_updates['token_types'])
            type_vocab, dim = self.embeddings.token_type_embeddings.weight.shape
            new_embeddings = nn.Embedding(type_vocab + n_types, dim)

            with torch.no_grad():
                new_embeddings.weight[:insert_idx] = self.embeddings.token_type_embeddings.weight[:insert_idx]
                new_embeddings.weight[insert_idx + n_types:] = self.embeddings.token_type_embeddings.weight[insert_idx:]
            self.embeddings.token_type_embeddings = new_embeddings.to(device)

            self.config.type_vocab_size += n_types


import os
import pickle

def load_trained_model(directory, checkpoint_n=-1):
    with open(directory + "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    model = Polygene.from_pretrained(directory + sorted([x for x in os.listdir(directory) if x.startswith('checkpoint-')])[checkpoint_n], ignore_mismatched_sizes=True)#, attn_implementation="flash_attention_2") # get last checkpoint of run
    model.to("cuda:0")
    model.eval()
    return model, tokenizer