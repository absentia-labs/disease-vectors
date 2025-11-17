import os, sys
sys.path.append('../../../')
from tqdm import tqdm
from sklearn.decomposition import PCA
from polygene.model.model import load_trained_model
from polygene.data_utils.tokenization import normalise_str
import torch.nn.functional as F
import pandas as pd, numpy as np
import torch, torch.nn as nn, scanpy as sc
from polygene.analysis.attributions.attributions import AttributionAnalysis
log_partition_function = lambda xi: torch.log( torch.sum( torch.exp( xi ) ) )

# the fisher rao metric is the second order taylor approximation of the KL divergence, it is the second derivative of the convex potential for natural parameters xi
def riemannian_metric(z, decoder):
    jacobian_decoder = torch.func.jacfwd(decoder)(z)
    hessian_log_partition = torch.func.hessian(log_partition_function)(decoder(z))
    G = jacobian_decoder.T @ hessian_log_partition @ jacobian_decoder
    return G

def christoffel_symbols(metric_function, z, decoder): #alternatively called the coefficients of affine connection for coordinates z. 
    di_Gjk= torch.func.jacfwd(metric_function, argnums=0)(z, decoder).permute(2, 0, 1) # stored as the last index so (j, k, i) but dimensions permuted to match varname
    
    levi_civita_connection = 1/2 * (di_Gjk + di_Gjk.permute(1, 0, 2) - di_Gjk.permute(2, 0, 1))
    return levi_civita_connection.detach().cpu().numpy()

def skewness_tensor(z, decoder):
    amari_chentsov_tensor_eta =  torch.func.jacfwd(torch.func.hessian(log_partition_function))(decoder(z))
    jacobian_decoder = torch.func.jacfwd(decoder)(z)
    amari_chentsov_tensor = ((amari_chentsov_tensor_eta @ jacobian_decoder).permute(2, 0, 1) @ jacobian_decoder).permute(0, 2, 1) @ jacobian_decoder
    del amari_chentsov_tensor_eta, jacobian_decoder
    torch.cuda.empty_cache()
    #primal_connection = levi_civita_connection + 1/2 * amari_chentsov_tensor
    #dual_connection = levi_civita_connection - 1/2 * amari_chentsov_tensor
    return amari_chentsov_tensor.detach().cpu().numpy()

EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/vector_data/'
EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/geodesic_data/'
mod, tok = load_trained_model("../../../runs/gesam_polygene_run_4/")
decoder = mod.prediction_head
torch.manual_seed(3)
device = "cuda:0"

class Geodesic(nn.Module):
    def __init__(self, z0, z1, total_t, decoder, device=None, dtype=torch.float32):
        super().__init__()
        self.decoder = decoder.eval()
        for p in self.decoder.parameters(): p.requires_grad_(False)

        z0 = torch.as_tensor(z0, dtype=dtype, device=device).view(1, -1)
        z1 = torch.as_tensor(z1, dtype=dtype, device=device).view(1, -1)
        self.register_buffer("z0", z0)
        self.register_buffer("z1", z1)

        self.total_t = total_t
        t = torch.linspace(0, 1, total_t, device=device, dtype=dtype)[1:-1].unsqueeze(1)
        self.interior_positions = nn.Parameter(z0 + t * (z1 - z0))
        self.energy_steps = []

    @property
    def positions(self):
        return torch.cat([self.z0, self.interior_positions, self.z1], dim=0)

    @property
    def optimization_steps(self):
        return len(self.energy_steps) - 1

    def energy(self):
        z = self.positions
        logits1 = self.decoder(z[:-1])
        logits2 = self.decoder(z[1:])
        logp = F.log_softmax(logits1, dim=-1)
        logq = F.log_softmax(logits2, dim=-1)
        p = logp.exp()
        kl_divergence = (p * (logp - logq)).sum(dim=-1).sum()
        return 2 * kl_divergence

    def optimize(self, steps=200, lr=1e-2, disable_tqdm=True):
        params = [self.interior_positions]
        opt = torch.optim.Adam(params, lr=lr)
        
        pbar = tqdm(range(steps), disable=disable_tqdm)
        for _ in pbar:
            opt.zero_grad(set_to_none=True)
            loss = self.energy()
            pbar.set_description(f"minimizing energy: {loss.item():.4f}")
            loss.backward()
            opt.step()
            self.energy_steps.append(loss.item())
        return self.positions.detach().cpu().numpy()
    
    def discretize(self, manifold, k=1):
        coords = self.positions
        cells = torch.as_tensor(manifold, dtype=coords.dtype, device=coords.device)
        d = torch.cdist(coords, cells)
        t_assign = d.argmin(dim=0)
        mask = torch.zeros_like(d, dtype=torch.bool)
        for t in range(coords.shape[0]):
            idx = torch.where(t_assign == t)[0]
            if idx.numel() > k:
                idx = idx[d[t, idx].topk(k, largest=False).indices]
            mask[t, idx] = True
        return mask.cpu().numpy()
    
    def path_integrated_gradients(self, cells, mask, model, tok, phenotype_value, disease_ontology_id,
                              biotype_json="../../data_utils/vocab/gene_biotypes.json",
                              ensembl_json="../../data_utils/vocab/ensembl_to_gene.json",
                              disable_pbar=True, k=100):
        """
        cells: anndata (n, g)
        mask: (t, n) generated by discretize function.
        """

        analyzer = AttributionAnalysis(model, tok, biotype_json=biotype_json, ensembl_json=ensembl_json, device=device)

        per_step_attributions = []
        per_step_gene_expression = []
        per_step_overlap, per_step_candidate = [], []

        for t in tqdm(range(mask.shape[0]), "IGIG", disable=disable_pbar):
            analyzer.data = cells[mask[t]]
            attributions = analyzer.gradients(phenotype="disease", only_protein_encoding=True, disable_pbar=True, force_phenotype_value=normalise_str(phenotype_value)).sum(axis=0)
            expr = analyzer.data.to_df()
            expr.columns = [analyzer.ensembl_id_to_gene_name.get(e, e) for e in expr.columns]
            avg_expr = expr[attributions.index.tolist()].mean(axis=0)

            per_step_attributions.append(attributions)
            per_step_gene_expression.append(avg_expr)

        path_attributions = pd.concat(per_step_attributions, axis=1).T.fillna(0)
        gamma_dot = pd.concat(per_step_gene_expression, axis=1).T.fillna(0).diff().fillna(0)
        cumulative_integrated_gradients = (path_attributions * gamma_dot).cumsum()
        opentarget = set(analyzer.get_associated_genes(phenotype_value, disease_ontology_id, k).keys())

        for t in range(cumulative_integrated_gradients.shape[0]):
            ranked = cumulative_integrated_gradients.iloc[t].sort_values(ascending=False)
            topk = ranked.index[:k]
            overlap = [g for g in topk if g in opentarget]
            candidates = [g for g in topk if g not in opentarget]
            per_step_overlap.append(overlap)
            per_step_candidate.append(candidates)

        return dict(
            per_step_overlap=per_step_overlap,
            per_step_candidate=per_step_candidate,
            cumulative_integrated_gradients=cumulative_integrated_gradients.iloc[-1]
        )

    def get_stats(self, decoder):
        gamma = self.positions
        entropy = []
        amari_norm = []
        for gamma_t in gamma:
            
            G = riemannian_metric(gamma_t, decoder).detach().cpu().numpy()
            amari_tensor = skewness_tensor(gamma_t, decoder)
            G = 0.5 * (G + G.T) + 1e-6 * np.eye(G.shape[0]) # neural network fisher rao is extremely ill conditionned low rank. so need 1e-6 for numerical stability
            eigvals = np.linalg.eigvalsh(G)
            p = eigvals / eigvals.sum()
            entropy.append(-(p * np.log(p + 1e-12)).sum())
            amari_norm.append(np.sqrt(np.sum(amari_tensor**2)))
        
        return entropy, amari_norm

if __name__ == "__main__":
    results = {}
    k_discrete = 10
    n_geodesics = 50
    n_timepoints = 50

    for disease in [f.split('_')[0] for f in os.listdir(EMBEDDINGS_DIR) if "_0_embedding" in f]:
        #saved_embeddings = pd.read_pickle(EMBEDDINGS_DIR + disease + '_embeddings.pkl')
        #saved_cells = sc.read_h5ad(EMBEDDINGS_DIR + disease + "_cells.h5ad")
        saved_embeddings = pd.read_pickle(EMBEDDINGS_DIR + disease + '_0_embeddings.pkl')
        saved_cells = sc.read_h5ad(EMBEDDINGS_DIR + disease + "_0_cells.h5ad")
        df = pd.DataFrame({'embedding': saved_embeddings[0].tolist()} | {tok.phenotypic_types[i]: saved_embeddings[2][:, i] for i in range(len(tok.phenotypic_types))})

        healthy_cells =  df[df['disease'] == '[normal]'].sample(n=n_geodesics, random_state=3)['embedding'].tolist()
        disease_cells =  df[df['disease'] == normalise_str(disease)].sample(n=n_geodesics, random_state=3)['embedding'].tolist()
        M = np.array(df['embedding'].tolist())
        paths, losses, discretizations, ig_paths, stats = [], [], [], [], []

        disease_ontology_id = saved_cells.obs[saved_cells.obs['disease'] == disease]['disease_ontology_term_id'].tolist()[0]
        print(disease_ontology_id)
        for healthy_embedding, disease_embedding in zip(healthy_cells, disease_cells):
            gamma = Geodesic(healthy_embedding, disease_embedding, total_t=n_timepoints, decoder=decoder, device=device)
            path_coordinates = gamma.optimize(steps=1000, lr=1e-3)
            discrete_mask = gamma.discretize(M, k=k_discrete)
            ig_paths.append( gamma.path_ig(saved_cells, discrete_mask, mod, tok, phenotype_value=disease, disease_ontology_id = disease_ontology_id) ) 

            stats.append( gamma.get_stats(decoder) )
            paths.append(path_coordinates)
            losses.append(gamma.energy_steps)
            discretizations.append(discrete_mask)

        proj = PCA(2, whiten=True, svd_solver="full", random_state=3)
        df[['x','y']] = proj.fit_transform(np.array(df['embedding'].tolist())).tolist()
        explained_var=proj.explained_variance_ratio_

        results[disease] = {
            "df": df,
            "paths": paths,
            "pca_paths": [proj.transform(path) for path in paths],
            "losses": losses,
            "explained_var": explained_var,
            "discretizations": discretizations,
            "ig_paths": ig_paths,
            "total_cells":len(saved_cells),
            "fisher_entropy": [s[0] for s in stats],
            "amari_norm": [s[1] for s in stats],
            "cell_type": saved_cells.obs['cell_type'].tolist()[0].title()
        }

    pd.to_pickle(results, EMBEDDINGS_DIR + f"results_{n_geodesics}.pkl")
    #import pandas as pd, numpy as np
#EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/geodesic_data/'
#results = pd.read_pickle(EMBEDDINGS_DIR + "results.pkl")
