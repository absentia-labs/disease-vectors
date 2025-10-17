import anndata as an
import scanpy as sc
import sys
sys.path.append('../../')
from polygene.eval.metrics import prepare_cell
import argparse
from polygene.data_utils.tokenization import GeneTokenizer
from tqdm import tqdm
import requests
import torch.nn.functional as F
import pandas as pd, numpy as np
from scipy.stats import fisher_exact
import torch
import json


class AttributionAnalysis():
    def __init__(self, model, tokenizer: GeneTokenizer , data = None, device="cuda:0", biotype_json="../data_utils/vocab/gene_biotypes.json", ensembl_json="../data_utils/vocab/ensembl_to_gene.json"):
        """
        Initialize Analyzer that computes feature attributions with different methods for a group of cells
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
    
    def __call__(self, method='integrated_gradients', data=None):
        if data is None and self.data is None: return print("Missing data")

    def gradients(self, phenotype="disease", reduction_func = None, only_protein_encoding=True, disable_pbar=False):
        # reduction function needs to take a (S, D) matrix and return an (S,) vector. 
        reduction_func = (lambda m: np.linalg.norm(m, axis=1)) if reduction_func is None else reduction_func
        
        cell_attributions = []
        for cell in tqdm(self.data, disable=disable_pbar):
            x = prepare_cell(cell, self.tok)
            phenotype_value = x['str_labels'][1+self.tok.phenotypic_types.index(phenotype)]
            x['inputs_embeds'] = self.model.embeddings(x['input_ids'].to(self.device), x['token_type_ids'].to(self.device)).detach().requires_grad_(True)

            y = F.softmax(self.model(**{k:v.unsqueeze(0).to(self.device) for k,v in x.items() if k != 'str_labels'}).logits, dim=-1) # (B, S, D)
            Ly = y[0, 1+self.tok.phenotypic_types.index(phenotype), self.tok.flattened_tokens.index(phenotype_value)]
            self.model.zero_grad()
            Ly.backward()

            attributions_per_gene = reduction_func(x['inputs_embeds'].grad.detach().cpu().numpy()[self.tok.gene_token_type_offset:])
            gene_ensembl_ids = [self.list_of_ensembl_ids[gene - self.tok.gene_token_type_offset]
                                 for gene in x['token_type_ids'].detach().cpu().numpy()[self.tok.gene_token_type_offset:]]
            cell_attributions.append( {k:v for k,v in zip(gene_ensembl_ids, attributions_per_gene)})
        self.cell_attributions_df = pd.DataFrame(cell_attributions).fillna(0)
        
        if only_protein_encoding and self.gene_biotypes is not None: # This can take an extra 2-3 minutes, perhaps save protein encoding information in a json
            ensembl_ids =  self.cell_attributions_df.columns.tolist()
            mask = [eid for eid in ensembl_ids if self.gene_biotypes.get(eid, "") == "protein_coding"]
            #print(f"Computed Attributions for {len(ensembl_ids)} genes and kept {len(mask)} protein encoding genes")
            self.cell_attributions_df = self.cell_attributions_df.loc[:, np.array(mask)]

        self.cell_attributions_df.columns = pd.Series(self.cell_attributions_df.columns).apply(lambda ensembl: self.ensembl_id_to_gene_name[ensembl])
        return self.cell_attributions_df

    def integrated_gradients(self, phenotype="disease", reduction_func=None, steps=10, only_protein_encoding=True, disable_pbar=False):
        reduction_func = (lambda m: np.linalg.norm(m, axis=1)) if reduction_func is None else reduction_func
        cell_attributions = []
        for cell in tqdm(self.data, disable=disable_pbar):
            x = prepare_cell(cell, self.tok); phv = x['str_labels'][1+self.tok.phenotypic_types.index(phenotype)]
            inp = self.model.embeddings(x['input_ids'].to(self.device), x['token_type_ids'].to(self.device)).detach()
            base = torch.zeros_like(inp).to(self.device); diff = inp-base; grads = torch.zeros_like(inp)
            for a in torch.linspace(0,1,steps):
                scaled = (base+a*diff).detach().requires_grad_(True); feed={k:v.unsqueeze(0).to(self.device) for k,v in x.items() if k!='str_labels'}; feed['inputs_embeds']=scaled.unsqueeze(0)
                y = F.softmax(self.model(**feed).logits,dim=-1); Ly=y[0,1+self.tok.phenotypic_types.index(phenotype),self.tok.flattened_tokens.index(phv)]; self.model.zero_grad(); Ly.backward(retain_graph=True); grads+=scaled.grad.detach()
            ig=(diff*grads/steps).detach().cpu().numpy(); vals=reduction_func(ig[self.tok.gene_token_type_offset:]); ids=[self.list_of_ensembl_ids[i-self.tok.gene_token_type_offset] for i in x['token_type_ids'].cpu().numpy()[self.tok.gene_token_type_offset:]]
            cell_attributions.append(dict(zip(ids,vals)))
        self.cell_attributions_df=pd.DataFrame(cell_attributions).fillna(0)
        if only_protein_encoding and self.gene_biotypes is not None: # This can take an extra 2-3 minutes, perhaps save protein encoding information in a json
            ensembl_ids =  self.cell_attributions_df.columns.tolist()
            mask = [eid for eid in ensembl_ids if self.gene_biotypes.get(eid, "") == "protein_coding"]
            #print(f"Computed Attributions for {len(ensembl_ids)} genes and kept {len(mask)} protein encoding genes")
            self.cell_attributions_df = self.cell_attributions_df.loc[:, np.array(mask)]

        self.cell_attributions_df.columns = pd.Series(self.cell_attributions_df.columns).apply(lambda ensembl: self.ensembl_id_to_gene_name[ensembl])
        return self.cell_attributions_df
    
    def disease_vector_ig(self, index_pairs, phenotype="disease", reduction_func=None, only_protein_encoding=True, steps=2, disable_pbar=False):
        reduction_func = (lambda m: np.linalg.norm(m, axis=1)) if reduction_func is None else reduction_func
        phenotype_index = 1 + self.tok.phenotypic_types.index(phenotype)
        pair_attributions = []
        pair_labels = []
        for i_base, i_target in tqdm(index_pairs, disable=disable_pbar):
            base_cell = self.data[i_base]; target_cell = self.data[i_target]
            xb = prepare_cell(base_cell, self.tok); xt = prepare_cell(target_cell, self.tok)
            phenotype_value = xt['str_labels'][phenotype_index]

            offset = self.tok.gene_token_type_offset
            pad_id = getattr(self.tok, "pad_token_id", self.tok.convert_tokens_to_ids(self.tok.pad_token))
            d, idt, ttd, amd = xt["input_ids"].device, xt["input_ids"].dtype, xt["token_type_ids"].dtype, xt["attention_mask"].dtype

            t1, i1 = xt["token_type_ids"][offset:].tolist(), xt["input_ids"][offset:].tolist()
            t2, i2 = xb["token_type_ids"][offset:].tolist(), xb["input_ids"][offset:].tolist()
            m1, m2 = {t: i for t, i in zip(t1, i1)}, {t: i for t, i in zip(t2, i2)}
            s1, s2 = set(t1), set(t2)
            inter, u1, u2 = list(s1 & s2), list(s1 - s2), list(s2 - s1)

            ids1, tt1, am1 = [m1[t] for t in inter + u1] + [pad_id] * len(u2), inter + u1 + [0] * len(u2), [1] * (len(inter) + len(u1)) + [0] * len(u2)
            ids2, tt2, am2 = [m2[t] for t in inter] + [pad_id] * len(u1) + [m2[t] for t in u2], inter + [0] * len(u1) + u2 , [1] * len(inter) + [0] * len(u1) + [1] * len(u2)

            blended_xt = {
                "input_ids": torch.cat([xt["input_ids"][:offset], torch.tensor(ids1, device=d, dtype=idt)]),
                "token_type_ids": torch.cat([xt["token_type_ids"][:offset], torch.tensor(tt1, device=d, dtype=ttd),]),
                "attention_mask": torch.cat([xt["attention_mask"][:offset], torch.tensor(am1, device=d, dtype=amd),])
            }
            blended_xb = {
                "input_ids": torch.cat([xb["input_ids"][:offset], torch.tensor(ids2, device=d, dtype=idt)]),
                "token_type_ids": torch.cat([xb["token_type_ids"][:offset], torch.tensor(tt2, device=d, dtype=ttd),]),
                "attention_mask": torch.cat([xb["attention_mask"][:offset], torch.tensor(am2, device=d, dtype=amd),])
            }
            xt, xb = blended_xt, blended_xb
            target_embeddings = self.model.embeddings(xt['input_ids'].to(self.device), xt['token_type_ids'].to(self.device)).detach()
            baseline_embeddings = self.model.embeddings(xb['input_ids'].to(self.device), xb['token_type_ids'].to(self.device)).detach()
            if target_embeddings.shape != baseline_embeddings.shape: continue
            difference = target_embeddings - baseline_embeddings
            gradients = torch.zeros_like(target_embeddings)
            for a in torch.linspace(0,1,steps):
                scaled = (baseline_embeddings + a * difference).detach().requires_grad_(True)
                feed = {k: v.unsqueeze(0).to(self.device) for k, v in xt.items() if k != 'str_labels'}
                feed['inputs_embeds'] = scaled.unsqueeze(0)
                y = F.softmax(self.model(**feed).logits, dim=-1)
                L = y[0, phenotype_index, self.tok.flattened_tokens.index(phenotype_value)]
                self.model.zero_grad()
                L.backward(retain_graph=True)
                gradients += scaled.grad.detach()
            ig = (difference * gradients / steps).detach().cpu().numpy()
            values = reduction_func(ig[self.tok.gene_token_type_offset:])
            token_types = xt['token_type_ids'].cpu().numpy()
            ensembl_ids = [self.list_of_ensembl_ids[i - self.tok.gene_token_type_offset] for i in token_types[self.tok.gene_token_type_offset:]]
            pair_attributions.append(dict(zip(ensembl_ids, values)))
            pair_labels.append(f"{i_base}->{i_target}")
        df = pd.DataFrame(pair_attributions, index=pair_labels).fillna(0)
        if only_protein_encoding and self.gene_biotypes is not None:
            keep = [eid for eid in df.columns if self.gene_biotypes.get(eid, "") == "protein_coding"]
            df = df.loc[:, np.array(keep)]
        df.columns = pd.Series(df.columns).apply(lambda ensembl: self.ensembl_id_to_gene_name.get(ensembl, ensembl))
        self.cell_attributions_df = df
        return df

    def deep_lift(self, phenotype="disease", reduction_func=None, only_protein_encoding=True, disable_pbar=True):
        from captum.attr import DeepLift
        reduction_func = (lambda m: np.linalg.norm(m, axis=1)) if reduction_func is None else reduction_func
        cell_attributions=[]
        for cell in tqdm(self.data, disable=disable_pbar):
            x = prepare_cell(cell, self.tok); phv = x['str_labels'][1+self.tok.phenotypic_types.index(phenotype)]
            feed = {k:v.unsqueeze(0).to(self.device) for k,v in x.items() if k!='str_labels'}
            feed['inputs_embeds'] = self.model.embeddings(x['input_ids'].to(self.device), x['token_type_ids'].to(self.device)).detach().requires_grad_(True).unsqueeze(0)
            base = torch.zeros_like(feed['inputs_embeds']).to(self.device)

            class ForwardWrapper(torch.nn.Module):
                def __init__(sself, model): super().__init__(); sself.model=model
                def forward(sself, inputs_embeds, attention_mask): return sself.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
            dl = DeepLift(ForwardWrapper(self.model))

            target = (1+self.tok.phenotypic_types.index(phenotype),self.tok.flattened_tokens.index(phv))

            out = dl.attribute(inputs=feed['inputs_embeds'], baselines=base, additional_forward_args=(feed["attention_mask"],), target=target)
            
            vals = reduction_func(out[0].detach().cpu().numpy()[self.tok.gene_token_type_offset:])
            ids = [self.list_of_ensembl_ids[i-self.tok.gene_token_type_offset] for i in x['token_type_ids'].cpu().numpy()[self.tok.gene_token_type_offset:]]
            cell_attributions.append(dict(zip(ids, vals)))
        self.cell_attributions_df = pd.DataFrame(cell_attributions).fillna(0)
        if only_protein_encoding and self.gene_biotypes is not None: # This can take an extra 2-3 minutes, perhaps save protein encoding information in a json
            ensembl_ids =  self.cell_attributions_df.columns.tolist()
            mask = [eid for eid in ensembl_ids if self.gene_biotypes.get(eid, "") == "protein_coding"]
            #print(f"Computed Attributions for {len(ensembl_ids)} genes and kept {len(mask)} protein encoding genes")
            self.cell_attributions_df = self.cell_attributions_df.loc[:, np.array(mask)]

        self.cell_attributions_df.columns = pd.Series(self.cell_attributions_df.columns).apply(lambda ensembl: self.ensembl_id_to_gene_name[ensembl])
        return self.cell_attributions_df
        
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
        response = requests.post(url, json={"query": query_assoc, "variables": variables}).json()['data']['disease']

        if response is None:
            query_search = """
                query search($term: String!) {
                    search(queryString: $term, entityNames: ["disease"]) {
                        hits { id name }
                    }
                }
            """
            search_resp = requests.post(url, json={"query": query_search, "variables": {"term": disease_name}}).json()
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

    
    def validate_attributions(self, k=100, method_top_attr = "above_mean", phenotype_obs_key="disease", phenotype_obs_value="normal", baseline=None, 
                              start_with_top_X_genes=None):
        # assuming all cells are part of same phenotype
        ontology_id = self.data.obs[self.data.obs[phenotype_obs_key] == phenotype_obs_value]['disease_ontology_term_id'].tolist()[0]
        attributed_genes = self.cell_attributions_df.loc[(self.data.obs[phenotype_obs_key] == phenotype_obs_value).tolist(),:].sum(axis=0) if baseline is None else baseline
        if start_with_top_X_genes is not None: attributed_genes = attributed_genes.sort_values(ascending=False)[:start_with_top_X_genes]
        
        opentarget_dict = self.get_associated_genes(phenotype_obs_key, ontology_id, k)
        opentarget_genes = set(opentarget_dict.keys())

        if method_top_attr == "above_mean":
            #print("attribution mean:", round(attributed_genes.mean(), 5))
            topk_attributed = set(attributed_genes[attributed_genes > attributed_genes.mean()].index.tolist())
        elif method_top_attr == "Q3":
            topk_attributed = set(attributed_genes[attributed_genes > attributed_genes.quantile(q=0.75)].index.tolist())
        elif method_top_attr == "IQR":
            q1, q3 = attributed_genes.quantile([0.25, 0.75])
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
            topk_attributed = set(attributed_genes[attributed_genes > threshold].index.tolist())
        else: 
            topk_attributed = set(attributed_genes.sort_values(ascending=False)[:k].index.tolist())

        overlap_genes = opentarget_genes & topk_attributed
        overlap = len(overlap_genes)
        TP, FP, FN = overlap, len(topk_attributed) - overlap, len(opentarget_genes) - overlap
        TN = len(attributed_genes) - (TP+FP+FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        odds, pval = fisher_exact([[TP,FP],[FN,TN]], alternative="greater") if all(x > 0 for x in  [TP, FP, FN, TN]) else (0, 1)
        if baseline is None:
            print(f"{overlap} common genes from {k} open target genes and {len(topk_attributed)} attributed genes")
            print(f"\nRecall: {recall:.3f}, Precision/Novelty: {precision:.3f}, Fisher's Exact p: {pval:.5f}, " 
                f"Overlap Strength: {sum([opentarget_dict.get(gene) for gene in overlap_genes]):.2f}, Overlap Genes: {sorted(overlap_genes)}")
            print(f"Candidate novel genes:{attributed_genes.drop(list(overlap_genes)).sort_values(ascending=False).index.tolist()[:3]}")
        max_overlap = opentarget_genes & set(attributed_genes.index.tolist())

        return opentarget_genes, attributed_genes, overlap/max(len(max_overlap), 1), recall, precision, pval, sorted(overlap_genes)

    def baselines(self, phenotype_obs_key='disease', case_label=None, control_label=None, k=50, method_top_attr="Q3", start_with_X=None):
        from scipy.stats import ttest_ind
        from sklearn.metrics import mutual_info_score
        X = self.data.X.toarray()

        # fair comparison
        #print('Lower bin edge:', self.tok.config.bin_edges[0])
        X[X < self.tok.config.bin_edges[0]] = 0
        nonzero_mask = X.sum(axis=0) > 0
        X = X[:, nonzero_mask]
        var_names = self.data.var_names[nonzero_mask]

        y = self.data.obs[phenotype_obs_key].to_numpy()
        mask_case, mask_ctrl = (y == case_label), (y == control_label)
        Xc, Xn = X[mask_case], X[mask_ctrl]
        t_stat, _ = ttest_ind(Xc, Xn, axis=0, equal_var=False, nan_policy='omit')

        y_bin = np.where(y == case_label, 1, 0)
        mi_scores = []
        for j in range(X.shape[1]):
            xj = np.asarray(X[:, j]).ravel()
            bins = np.quantile(xj, np.linspace(0, 1, 6))
            xj_disc = np.digitize(xj, bins, right=True)
            mi_scores.append(mutual_info_score(xj_disc, y_bin))

        #print("GWAS baseline:")
        baseline = pd.Series(t_stat, index=[self.ensembl_id_to_gene_name.get(k,k) for k in var_names])
        out_gwas = self.validate_attributions(phenotype_obs_key=phenotype_obs_key, phenotype_obs_value=case_label, baseline=baseline, method_top_attr=method_top_attr, k=k, start_with_top_X_genes=start_with_X)
        #print("Mutual Information baseline:")
        baseline = pd.Series(mi_scores, index=[self.ensembl_id_to_gene_name.get(k,k) for k in var_names])
        out_mi = self.validate_attributions(phenotype_obs_key=phenotype_obs_key, phenotype_obs_value=case_label, baseline=baseline, method_top_attr=method_top_attr, k=k, start_with_top_X_genes=start_with_X)
        return out_gwas, out_mi

if __name__== "__main__":

    parser = argparse.ArgumentParser("attributions program")
    parser.add_argument("--model_path", default="../../runs/gesam_polygene_run_4/")
    parser.add_argument("--data_path")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n_cells", type=int, default=100)
    parser.add_argument("--method_top_attr", default="Q3")
    parser.add_argument("--disease_name")
    parser.add_argument("--baselines", action="store_true")
    args = parser.parse_args()

    from polygene.model.model import load_trained_model
    import scanpy as sc

    #model_path = "../../../runs/gesam_polygene_run_4/" #'/media/lleger/LaCie/POLYGENE/'
    model, tokenizer = load_trained_model(args.model_path, checkpoint_n=-1)

    cells = sc.read_h5ad(args.data_path)
    idx = cells.obs.groupby('disease', group_keys=False).apply(lambda x: x.sample(n=min(args.n_cells, len(x)))).index
    cells = cells[idx[cells.obs.loc[idx, 'disease'].isin([args.disease_name, 'normal'])]]
    analyzer = AttributionAnalysis(model, tokenizer, data=cells)
    
    # BASELINE
    r = analyzer.baselines(phenotype_obs_key="disease", case_label="Alzheimer disease", control_label="normal", method_top_attr="Q3", k=50)
    
    cells = cells[cells.obs.loc[idx, 'disease'].isin([args.disease_name])]
    analyzer.cells = cells

    # Gradients
    df = analyzer.gradients(only_protein_encoding=True)
    r = analyzer.validate_attributions(k=50, method_top_attr="Q3", phenotype_obs_value="Alzheimer disease")
    # Integrated Gradients
    df = analyzer.integrated_gradients(only_protein_encoding=True)
    r = analyzer.validate_attributions(k=50, method_top_attr="Q3", phenotype_obs_value="Alzheimer disease")
    # DeepLIFT Gradients
    df = analyzer.deep_lift(only_protein_encoding=True)
    r = analyzer.validate_attributions(k=50, method_top_attr="Q3", phenotype_obs_value="Alzheimer disease")
