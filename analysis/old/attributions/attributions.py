import sys
sys.path.append('../../../')
from polygene.eval.metrics import prepare_cell
import json
import torch
import requests
import scanpy as sc
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd, numpy as np
from polygene.data_utils.tokenization import GeneTokenizer, normalise_str
from scipy.stats import fisher_exact, ttest_ind
from sklearn.metrics import mutual_info_score

class AttributionAnalysis():
    def __init__(self, model, tokenizer: GeneTokenizer , data = None, device="cuda:0", biotype_json="../data_utils/vocab/gene_biotypes.json", ensembl_json="../data_utils/vocab/ensembl_to_gene.json"):
        """
        Initialize Analyzer that computes feature attributions with different methods for alpha group of cells
        model: PyTorch Model
        tokenizer: tokenizer saved from pickle
        data: str or AnnData h5ad file with cells of interest
        """
        self.model, self.tok = model, tokenizer
        self.device = device
        model.to(device)
        self.data = sc.read_h5ad(data) if isinstance(data, str) else data
        self.list_of_ensembl_ids = list(self.tok.gene_type_id_map.keys())

        self.ensembl_id_to_gene_name = json.load(open(ensembl_json)) if ensembl_json is not None else None
        self.gene_biotypes = json.load(open(biotype_json)) if biotype_json is not None else None
    
    def gradients(self, phenotype="disease", reduction_func = None, only_protein_encoding=True, disable_pbar=False, force_phenotype_value=None, use_data_slice=None):
        # reduction function needs to take alpha (S, D) matrix and return an (S,) vector. 
        reduction_func = (lambda m: np.linalg.norm(m, axis=1)) if reduction_func is None else reduction_func
        
        data = self.data if use_data_slice is None else use_data_slice
        cell_attributions = []
        for cell in tqdm(data, disable=disable_pbar):
            x = prepare_cell(cell, self.tok)
            x['input_ids'][torch.arange(1, 1+len(self.tok.phenotypic_types))] = self.tok.token_to_id_map[self.tok.mask_token]
            phenotype_value = x['str_labels'][1+self.tok.phenotypic_types.index(phenotype)] if force_phenotype_value is None else normalise_str(force_phenotype_value)
            x['inputs_embeds'] = self.model.embeddings(x['input_ids'].to(self.device), x['token_type_ids'].to(self.device)).detach().requires_grad_(True)

            y = F.softmax(self.model(**{k:v.unsqueeze(0).to(self.device) for k,v in x.items() if k != 'str_labels'}).logits, dim=-1) # (B, S, D)
            Ly = y[0, 1+self.tok.phenotypic_types.index(phenotype), self.tok.flattened_tokens.index(phenotype_value)]
            self.model.zero_grad()
            Ly.backward()

            attributions_per_gene = reduction_func(x['inputs_embeds'].grad.detach().cpu().numpy()[self.tok.gene_token_type_offset:])
            gene_ensembl_ids = [self.list_of_ensembl_ids[gene - self.tok.gene_token_type_offset] for gene in x['token_type_ids'].detach().cpu().numpy()[self.tok.gene_token_type_offset:]]
            cell_attributions.append( {k:v for k,v in zip(gene_ensembl_ids, attributions_per_gene)})

        self.cell_attributions_df = pd.DataFrame(cell_attributions).fillna(0)
        if only_protein_encoding: self.cell_attributions_df = self.select_protein_encoding_genes(self.cell_attributions_df)
        self.cell_attributions_df.columns = pd.Series(self.cell_attributions_df.columns).apply(lambda ensembl: self.ensembl_id_to_gene_name[ensembl])
        return self.cell_attributions_df

    def integrated_gradients(self, phenotype="disease", reduction_func=None, steps=10, only_protein_encoding=True, disable_pbar=False, force_phenotype_value=None):
        reduction_func = (lambda m: np.linalg.norm(m, axis=1)) if reduction_func is None else reduction_func
        cell_attributions = []
        for cell in tqdm(self.data, disable=disable_pbar):
            x = prepare_cell(cell, self.tok)
            x['input_ids'][torch.arange(1, 1+len(self.tok.phenotypic_types))] = self.tok.token_to_id_map[self.tok.mask_token]
            phenotype_value = x['str_labels'][1+self.tok.phenotypic_types.index(phenotype)] if force_phenotype_value is None else normalise_str(force_phenotype_value)

            target = self.model.embeddings(x['input_ids'].to(self.device), x['token_type_ids'].to(self.device)).detach()
            baseline = self.get_padded_baseline(x)
            gradients = torch.zeros_like(target)

            for alpha in torch.linspace(0, 1, steps):
                interpolated_input = (baseline + alpha*target).detach().requires_grad_(True)
                x['inputs_embeds']=interpolated_input

                y = F.softmax(self.model(**{k:v.unsqueeze(0).to(self.device) for k,v in x.items() if k != 'str_labels'}).logits, dim=-1) # (B, S, D)
                Ly = y[0,1+self.tok.phenotypic_types.index(phenotype),self.tok.flattened_tokens.index(phenotype_value)]
                self.model.zero_grad()
                Ly.backward(retain_graph=True)
                gradients += interpolated_input.grad.detach()

            integrated_gradients = ( (target - baseline) * gradients / steps).detach().cpu().numpy()
            attributions_per_gene = reduction_func(integrated_gradients[self.tok.gene_token_type_offset:])
            
            gene_ensembl_ids = [self.list_of_ensembl_ids[i-self.tok.gene_token_type_offset] for i in x['token_type_ids'].cpu().numpy()[self.tok.gene_token_type_offset:]]
            cell_attributions.append(dict(zip(gene_ensembl_ids,attributions_per_gene)))
        
        self.cell_attributions_df = pd.DataFrame(cell_attributions).fillna(0)
        if only_protein_encoding: self.cell_attributions_df = self.select_protein_encoding_genes(self.cell_attributions_df)
        self.cell_attributions_df.columns = pd.Series(self.cell_attributions_df.columns).apply(lambda ensembl: self.ensembl_id_to_gene_name[ensembl])
        return self.cell_attributions_df
    
    def deep_lift(self, phenotype="disease", reduction_func=None, only_protein_encoding=True, disable_pbar=True, force_phenotype_value=None):
        from captum.attr import DeepLift
        reduction_func = (lambda m: np.linalg.norm(m, axis=1)) if reduction_func is None else reduction_func
        cell_attributions=[]
        for cell in tqdm(self.data, disable=disable_pbar):
            x = prepare_cell(cell, self.tok)
            phenotype_value = x['str_labels'][1+self.tok.phenotypic_types.index(phenotype)] if force_phenotype_value is None else normalise_str(force_phenotype_value)
            feed = {k:v.unsqueeze(0).to(self.device) for k,v in x.items() if k!='str_labels'}
            feed['inputs_embeds'] = self.model.embeddings(x['input_ids'].to(self.device), x['token_type_ids'].to(self.device)).detach().requires_grad_(True).unsqueeze(0)
            base = self.get_padded_baseline(x).unsqueeze(0)

            class ForwardWrapper(torch.nn.Module):
                def __init__(sself, model): super().__init__(); sself.model=model
                def forward(sself, inputs_embeds, attention_mask): return sself.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
            dl = DeepLift(ForwardWrapper(self.model))

            target = (1+self.tok.phenotypic_types.index(phenotype),self.tok.flattened_tokens.index(phenotype_value))

            DeepLIFT_attributions = dl.attribute(inputs=feed['inputs_embeds'], baselines=base, additional_forward_args=(feed["attention_mask"],), target=target)
            attributions_per_gene = reduction_func(DeepLIFT_attributions[0].detach().cpu().numpy()[self.tok.gene_token_type_offset:])
            gene_ensembl_ids = [self.list_of_ensembl_ids[i-self.tok.gene_token_type_offset] for i in x['token_type_ids'].cpu().numpy()[self.tok.gene_token_type_offset:]]
            cell_attributions.append(dict(zip(gene_ensembl_ids, attributions_per_gene)))
        self.cell_attributions_df = pd.DataFrame(cell_attributions).fillna(0)
        if only_protein_encoding: self.cell_attributions_df = self.select_protein_encoding_genes(self.cell_attributions_df)
        self.cell_attributions_df.columns = pd.Series(self.cell_attributions_df.columns).apply(lambda ensembl: self.ensembl_id_to_gene_name[ensembl])
        return self.cell_attributions_df
    
    def disease_vector(self, phenotype="disease", reduction_func=None, only_protein_encoding=True, disable_pbar=True):
        from captum.attr import DeepLift
        reduction_func = (lambda m: np.linalg.norm(m, axis=1)) if reduction_func is None else reduction_func
        cell_attributions=[]
        for cell in tqdm(self.data, disable=disable_pbar):
            x = prepare_cell(cell, self.tok); phenotype_value = x['str_labels'][1+self.tok.phenotypic_types.index(phenotype)]
            feed = {k:v.unsqueeze(0).to(self.device) for k,v in x.items() if k!='str_labels'}
            feed['inputs_embeds'] = self.model.embeddings(x['input_ids'].to(self.device), x['token_type_ids'].to(self.device)).detach().requires_grad_(True).unsqueeze(0)
            base = self.get_padded_baseline(x).unsqueeze(0)

            class ForwardWrapper(torch.nn.Module):
                def __init__(sself, model): super().__init__(); sself.model=model
                def forward(sself, inputs_embeds, attention_mask): return sself.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
            dl = DeepLift(ForwardWrapper(self.model))

            target = (1+self.tok.phenotypic_types.index(phenotype),self.tok.flattened_tokens.index(phenotype_value))

            DeepLIFT_attributions = dl.attribute(inputs=feed['inputs_embeds'], baselines=base, additional_forward_args=(feed["attention_mask"],), target=target)
            attributions_per_gene = reduction_func(DeepLIFT_attributions[0].detach().cpu().numpy()[self.tok.gene_token_type_offset:])
            gene_ensembl_ids = [self.list_of_ensembl_ids[i-self.tok.gene_token_type_offset] for i in x['token_type_ids'].cpu().numpy()[self.tok.gene_token_type_offset:]]
            cell_attributions.append(dict(zip(gene_ensembl_ids, attributions_per_gene)))
        self.cell_attributions_df = pd.DataFrame(cell_attributions).fillna(0)
        if only_protein_encoding: self.cell_attributions_df = self.select_protein_encoding_genes(self.cell_attributions_df)
        self.cell_attributions_df.columns = pd.Series(self.cell_attributions_df.columns).apply(lambda ensembl: self.ensembl_id_to_gene_name[ensembl])
        return self.cell_attributions_df
    
    def get_padded_baseline(self, x):
        x_baseline = {
            "input_ids": torch.cat([x['input_ids'][:self.tok.gene_token_type_offset],
                                     torch.tensor([self.tok.convert_tokens_to_ids(self.tok.pad_token)] * len(x['input_ids'][self.tok.gene_token_type_offset:]),
                                                   device=x['input_ids'].device, dtype=x['input_ids'].dtype)] ),
            "token_type_ids": torch.cat([x['token_type_ids'][:self.tok.gene_token_type_offset],
                                     torch.tensor([0] * len(x['input_ids'][self.tok.gene_token_type_offset:]), 
                                                  device=x['input_ids'].device, dtype=x['token_type_ids'].dtype)], ),
        }
        baseline_input_embeds = self.model.embeddings(x_baseline['input_ids'].to(self.device), x_baseline['token_type_ids'].to(self.device)).detach()
        return baseline_input_embeds
    
    def select_protein_encoding_genes(self, attributions_df):
        ensembl_ids =  attributions_df.columns.tolist()
        mask = [eid for eid in ensembl_ids if self.gene_biotypes.get(eid, "") == "protein_coding"]
        return attributions_df.loc[:, np.array(mask)]

    @staticmethod
    def get_associated_genes(disease_name, disease_ontology_term_id, top=100):
        url = "https://api.platform.opentargets.org/api/v4/graphql"
        disease_ontology_term_id = disease_ontology_term_id.replace(':', '_')

        query_assoc = """
            query associatedTargets($diseaseId: String!, $size: Int!) {
                disease(efoId: $diseaseId) {
                    id
                    name
                    associatedTargets(page: { index: 0, size: $size }) {
                        count
                        rows { target { id approvedSymbol } score }
                    }
                }
            }
        """

        variables = {"diseaseId": disease_ontology_term_id, "size": top}
        resp = requests.post(url, json={"query": query_assoc, "variables": variables})
        try:
            response_json = resp.json()
            response = response_json.get('data', {}).get('disease')
        except ValueError:
            return {}
        
        if response is None:
            query_search = """
                query search($term: String!) {
                    search(queryString: $term, entityNames: ["disease"]) {
                        hits { id name }
                    }
                }
            """
            #search_resp = requests.post(url, json={"query": query_search, "variables": {"term": disease_name}}).json()
            resp = requests.post(url, json={"query": query_search, "variables": {"term": disease_name}})
            try:
                search_resp = resp.json()
            except ValueError:
                return {}
            hits = search_resp.get('data', {}).get('search', {}).get('hits', [])
            if hits:
                new_id = hits[0]['id']
                variables = {"diseaseId": new_id, "size": top}
                response = requests.post(url, json={"query": query_assoc, "variables": variables}).json()['data']['disease']
            else:
                return {}

        if response:
            return {row['target']['approvedSymbol']: row['score'] for row in response['associatedTargets']['rows'][:top]}
        return {}
    
    def validate_attributions(self, k=100, phenotype_obs_key="disease", phenotype_obs_value="normal", baseline=None, overwrite_ontology_id=None):
        ontology_id = self.data.obs[self.data.obs[phenotype_obs_key] == phenotype_obs_value]['disease_ontology_term_id'].tolist()[0] if overwrite_ontology_id is None else overwrite_ontology_id
        attributed_genes = self.cell_attributions_df.loc[(self.data.obs[phenotype_obs_key] == phenotype_obs_value).tolist(),:].sum(axis=0) if baseline is None else baseline
        
        opentarget_dict = self.get_associated_genes(phenotype_obs_key, ontology_id, k)
        opentarget_genes = set(opentarget_dict.keys())
        topk_attributed = set(attributed_genes.sort_values(ascending=False)[:k].index.tolist())

        overlap_genes = opentarget_genes & topk_attributed
        overlap = len(overlap_genes)
        TP, FP, FN = overlap, k - overlap, k - overlap
        TN = len(attributed_genes) - (TP + FP + FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        _, pval = fisher_exact([[TP,FP],[FN,TN]], alternative="greater") if all(x > 0 for x in  [TP, FP, FN, TN]) else (0, 1)
        candidate_genes = attributed_genes.sort_values(ascending=False)[:k].drop(list(overlap_genes)).index.tolist()

        max_overlap = opentarget_genes & set(attributed_genes.index.tolist())

        return {"recall": recall,
                "max_recall": overlap/max(len(max_overlap), 1),
                "fisher_pvalue": pval,
                "opentarget_genes": pd.Series(opentarget_dict),
                "attributed_genes":  attributed_genes,
                "overlap_candidate_genes": (sorted(overlap_genes, key=lambda x: attributed_genes[x]), candidate_genes)
            }

    def baselines(self, phenotype_obs_key='disease', target_label=None, baseline_label=None, k=100):
        X = self.data.X.toarray()
        nonzero_mask = X.sum(axis=0) > 0
        X = X[:, nonzero_mask]
        var_names = self.data.var_names[nonzero_mask]

        y = self.data.obs[phenotype_obs_key].to_numpy()
        mask_target, mask_baseline = (y == target_label), (y == baseline_label)
        Xc, Xn = X[mask_target], X[mask_baseline]
        t_stat, _ = ttest_ind(Xc, Xn, axis=0, equal_var=False, nan_policy='omit')
        attributions_gwas = pd.Series(t_stat, index=[self.ensembl_id_to_gene_name.get(k,k) for k in var_names])
        gwas_eval = self.validate_attributions(phenotype_obs_key=phenotype_obs_key, phenotype_obs_value=target_label, baseline=attributions_gwas, k=k)

        y_bin = np.where(y == target_label, 1, 0)
        mi_scores = []
        for j in range(X.shape[1]):
            xj = np.asarray(X[:, j]).ravel()
            bins = np.quantile(xj, np.linspace(0, 1, 6))
            xj_disc = np.digitize(xj, bins, right=True)
            mi_scores.append(mutual_info_score(xj_disc, y_bin))

        attributions_mutual_information = pd.Series(mi_scores, index=[self.ensembl_id_to_gene_name.get(k,k) for k in var_names])
        mutual_information_eval = self.validate_attributions(phenotype_obs_key=phenotype_obs_key, phenotype_obs_value=target_label, baseline=attributions_mutual_information, k=k)
        return gwas_eval, mutual_information_eval
