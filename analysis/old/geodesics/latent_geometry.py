import os, sys
sys.path.append('../../../')
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.nn.functional import softmax
from polygene.model.model import load_trained_model
from polygene.data_utils.tokenization import normalise_str
import scanpy as sc, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes


EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_geometry/vectors/'
mod, tok = load_trained_model("../../../runs/gesam_polygene_run_4/")
decoder = mod.prediction_head


n_grad = 30
whiten=False
surround_border = 20
stabilizing_eps = 1e-6  # stabilizing coefficient pulling the metric toward the Euclidean identity

torch.manual_seed(3)

device = "cuda:0"
log_partition_function = lambda xi: torch.log( torch.sum( torch.exp( xi ) ) )

# the fisher rao metric is the second order taylor approximation of the KL divergence, it is the second derivative of the convex potential for natural parameters xi
def riemannian_metric(z, decoder):
    jacobian_decoder = torch.func.jacfwd(decoder)(z)
    hessian_log_partition = torch.func.hessian(log_partition_function)(decoder(z))
    G = jacobian_decoder.T @ hessian_log_partition @ jacobian_decoder
    return G

def christoffel_symbols(metric_function, z, decoder): #alternatively called the coefficients of affine connection parametrised in the coordinate system.
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

def curvature_tensors(metric, levi_civita_p, levi_civita_q, levi_civita_r, eps): 
    # this tensor quantifies the notion of holonomy and the round the world transport
    # how does your vector change as its being parallel transported on a closed curve. 
    # from it we can also compute some pretty elegant measures of curvature.d

    derivative_gamma_k = (levi_civita_q - levi_civita_p) / eps
    derivative_gamma_l = (levi_civita_r - levi_civita_p) / eps
    ricci_tensor = np.trace(derivative_gamma_k, axis1=0, axis2=1) - np.trace(derivative_gamma_l, axis1=0, axis2=1)
    inverse_metric = np.linalg.inv(metric)
    return np.sum(inverse_metric * ricci_tensor)

from scipy.linalg import cho_factor, cho_solve
def curvature_tensors(metric, levi_civita_p, levi_civita_q, levi_civita_r, eps):
    # this tensor quantifies the notion of holonomy and the round the world transport
    # how does your vector change as its being parallel transported on a closed curve. 
    # from it we can also compute some pretty elegant measures of curvature
    derivative_gamma_k = (levi_civita_q - levi_civita_p) / eps
    derivative_gamma_l = (levi_civita_r - levi_civita_p) / eps

    term1 = np.einsum('iil->l', derivative_gamma_k)
    term2 = np.einsum('ijl->l', derivative_gamma_l)

    term3 = np.einsum('iim,jml->jl', levi_civita_p, levi_civita_p, optimize=True)
    term4 = np.einsum('ijm,iml->jl', levi_civita_p, levi_civita_p, optimize=True)

    ricci_tensor = term1 - term2 + term3 - term4
    inverse_metric = cho_solve(cho_factor(metric), np.eye(metric.shape[0]))
    scalar_curvature = np.einsum('ij,ij->', inverse_metric, ricci_tensor)
    # directional derivative approximation of Ricci tensor. 
    return scalar_curvature # Discrete Riemannian Curvature on a Statistical manifold. 


for file_path in os.listdir(EMBEDDINGS_DIR):
    if "embeddings.pkl" not in file_path: continue
    saved_embeddings = pd.read_pickle(os.path.join(EMBEDDINGS_DIR, file_path))
    df = pd.DataFrame({'embedding': saved_embeddings[0].tolist()} | {tok.phenotypic_types[i]: saved_embeddings[2][:, i] for i in range(len(tok.phenotypic_types))})
    disease_list = sorted(df['disease'].unique().tolist(), key=lambda x: 0 if 'normal' in x else 1) 

    proj = PCA(2, whiten=whiten, svd_solver='full')
    df[['x', 'y']] = proj.fit_transform(np.array(df['embedding'].tolist())).tolist()
    explained_var = proj.explained_variance_ratio_.copy()

    x_min, x_max, y_min, y_max = df['x'].min()-surround_border, df['x'].max()+surround_border, df['y'].min()-surround_border, df['y'].max()+surround_border
    xg, yg = np.linspace(x_min, x_max, n_grad), np.linspace(y_min, y_max, n_grad)
    X, Y = np.meshgrid(xg, yg)

    phase_space_pca_coordinates = np.stack([X, Y], axis=-1).reshape(-1, 2)
    phase_space_embeddings = proj.inverse_transform(phase_space_pca_coordinates)

    gradients_per_embedding = []
    for embedding in tqdm(phase_space_embeddings, file_path + " - Gradients"):
        embedding = torch.tensor(embedding, device=device, dtype=torch.float32).requires_grad_(True)
        prob = softmax(decoder(embedding.unsqueeze(0)), dim=1).squeeze()
        gradients_per_disease = []
        for disease in disease_list: 
            prob_y =  prob[tok.token_to_id_map[normalise_str(disease)]]
            decoder.zero_grad()
            prob_y.backward(retain_graph=True)
            gradients_per_disease.append( embedding.grad.detach().cpu().numpy() )
            embedding.grad.zero_()
        gradients_per_embedding.append(np.array(gradients_per_disease))
    
    gradients = np.array(gradients_per_embedding).reshape(-1, phase_space_embeddings.shape[-1])
    grad_proj = PCA(2, whiten=whiten, svd_solver="full").fit(gradients)
    R, _ = orthogonal_procrustes(grad_proj.components_.T, proj.components_.T)
    phase_space_gradients = (grad_proj.transform(gradients) @ R).reshape(phase_space_embeddings.shape[0], len(disease_list), 2).reshape(n_grad, n_grad, len(disease_list), 2)

    #phase_space_gradients = proj.fit_transform(np.array(gradients_per_embedding).reshape(-1, phase_space_embeddings.shape[-1])).reshape(phase_space_embeddings.shape[0], len(disease_list), -1).reshape(n_grad, n_grad, len(disease_list), 2)
    
    results = { 
                "meshgrid": (X, Y), 
                "gradients": phase_space_gradients, # shape (mesh_num, mesh_num, [normal, disease], [direction_x, direction_y])
                "df": df, # original pca x,y coords and class information
                "explained_var": explained_var,
                "surround_border": surround_border,
                "n_grad": n_grad,
            }
    information_geometry_stats = []
    for embedding in tqdm(phase_space_embeddings, "Statistical Curvature"):
        
        z = torch.tensor(embedding, dtype=torch.float32, device=device).requires_grad_(True)
        G = riemannian_metric(z, decoder).detach().cpu().numpy()
        G = 0.5 * (G + G.T) + 1e-6 * np.eye(G.shape[0]) # neural network fisher rao is extremely ill conditionned low rank. so need 1e-6 for numerical stability
        levi_civita_connection = christoffel_symbols(riemannian_metric, z, decoder)
        amari_chentsov_tensor = skewness_tensor(z, decoder)

        # RC tensor approximation in infinitessimal rectangle
        eps = 1e-3
        levi_civita_connection_q = christoffel_symbols(riemannian_metric, z + eps*torch.randn_like(z, device=device), decoder)
        levi_civita_connection_r = christoffel_symbols(riemannian_metric, z + eps*torch.randn_like(z, device=device), decoder)
        curvature = curvature_tensors(G, levi_civita_connection, levi_civita_connection_q, levi_civita_connection_r, eps)
        
        eigvals = np.linalg.eigvalsh(G)
        spectral_p = eigvals / eigvals.sum()
        information_geometry_stats.append({
            "mean": G.mean(),
            "anistropy_ratio": eigvals.max() / eigvals.min(), # measure of anistropy (aka if some directions/variances dominate)
            "spectral_entropy": -(spectral_p * np.log(spectral_p + 1e-12)).sum(), # measure of isotropy
            "log_volume": 0.5 * np.log(eigvals).sum(),
            "ricci_curvature": curvature,
            "amari_norm":  np.sqrt(np.sum(amari_chentsov_tensor**2)), 
            "levi_civita_norm": np.sqrt(np.sum(levi_civita_connection**2)),
            })
    
    results["geometrical_stats"] = pd.DataFrame(information_geometry_stats)

    pd.to_pickle(results, EMBEDDINGS_DIR + file_path.split('_embeddings')[0]+ "_geometry_ricci.pkl")
    