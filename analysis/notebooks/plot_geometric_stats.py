SAVE = True
fontsize = 12


import os, sys
import scanpy as sc, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
from matplotlib.ticker import ScalarFormatter

EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/vector_data/'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

for file_path in os.listdir(EMBEDDINGS_DIR):#[]:
    if "_geometrical" not in file_path: continue
    stats = pd.read_pickle(os.path.join(EMBEDDINGS_DIR, file_path))
    X,Y = stats['meshgrid']
    explained_var = stats['explained_var']
    phase_space_gradients = stats["gradients"]
    surround_border, n_grad = stats['surround_border'], stats['n_grad']
    
    df = stats['df']
    geometry_df = stats['geometrical_stats']
    jacobian_field = 1
    step = 2
    geometry_df['healthy_gradient_norm'] = np.linalg.norm(phase_space_gradients, axis=-1)[:, :, 0].reshape(-1).tolist()
    geometry_df['disease_gradient_norm'] = np.linalg.norm(phase_space_gradients, axis=-1)[:, :, 1].reshape(-1).tolist() 

    x_min, x_max, y_min, y_max = df['x'].min()-surround_border, df['x'].max()+surround_border, df['y'].min()-surround_border, df['y'].max()+surround_border
    
    metrics = ['ricci_curvature', 'amari_norm', 'levi_civita_norm', 'anistropy_ratio', 'spectral_entropy', 'log_volume']
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=300)
    axes = axes.flatten()

    for idx, metric_name in enumerate(metrics):
        ax = axes[idx]
        heatmap_stat = np.array(geometry_df[metric_name].tolist()).reshape(n_grad, n_grad)
        phase_space_gradients_norm = phase_space_gradients / np.linalg.norm(phase_space_gradients, axis=-1, keepdims=True)# np.log(1 + 0.2*np.linalg.norm(phase_space_gradients, axis=-1, keepdims=True))
        gx, gy = phase_space_gradients_norm[:, :, jacobian_field, 0], phase_space_gradients_norm[:, :, jacobian_field, 1]

        ax.quiver(X[::step, ::step], Y[::step, ::step], gx[::step, ::step], gy[::step, ::step],
                color='gray', alpha=.5, scale_units='xy', scale=0.8, width=0.00375,
                headwidth=3, headlength=5, headaxislength=4, zorder=-5)

        cmap = "vlag" if metric_name == "ricci_curvature" else "Blues"
        im = ax.imshow(heatmap_stat, extent=[x_min, x_max, y_min, y_max],
                    origin='lower', aspect='auto', alpha=.5, zorder=-10, cmap=cmap)

        sns.scatterplot(data=df, x='x', y='y', hue=df['disease'],
                        palette=sns.color_palette('Paired', n_colors=4)[2:], s=5,
                        alpha=0.2, ax=ax, linewidth=0.1, zorder=1)

        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.spines[['top','right']].set_linewidth(0); ax.spines[['bottom','left']].set_linewidth(2)
        ax.set_xlim(df['x'].quantile(.01)-surround_border, df['x'].quantile(.99)+surround_border)
        ax.set_ylim(df['y'].quantile(.01)-surround_border, df['y'].quantile(.99)+surround_border)
        ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}\%)', fontsize=fontsize-2)
        ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}\%)', fontsize=fontsize-2)
        #plt.rc('font', family='serif')
        eq = {key:val for key,val in zip(metrics, ["Curvature of Ricci Tensor", "Amari-Chentsov Tensor Norm: $||T_{ijk}||_F$", "Levi-Civita Connection Norm: $||\Gamma^i_{jk}||_F$",
                                                   "Condition Number: $\lambda_{max}/\lambda_{min}$", "Spectral Entropy:  $H(p) = -\sum p_i \log(p_i) \;\; p_i = \lambda_i/\sum_j\lambda_j$",
                                                   "Log Volume: $\log( \sqrt{\det(G)}) $"])}
        ax.set_title(eq.get(metric_name, ' '.join(metric_name.split('_')).title()), fontsize=fontsize)


        handles, legend_labels_ = ax.get_legend_handles_labels()
        arrow_proxy = FancyArrowPatch((0, 0), (1, 0), arrowstyle='-|>', mutation_scale=15, color="gray", linewidth=1)
        def make_legend_quiver(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            return FancyArrowPatch((0, height*0.5), (width, height*0.5),
                                arrowstyle='-|>', mutation_scale=fontsize, color="gray", linewidth=1)
        handles.append(arrow_proxy)
        legend_labels_.append("Gradient Flow\n$\partial p (disease | z)$")
        legend_labels_ = [' '.join([word.capitalize() for word in x[1:-1].split('_')]).replace(' Right Ventricular', '')
                        if '[' in x else x for x in legend_labels_]
        legend_labels_ = [x.replace(' ', '\n') if len(x) > 17 and "\n" not in x else x for x in legend_labels_]
        legend = ax.legend(handles, legend_labels_, frameon=True, bbox_to_anchor=(1,1),
                        borderaxespad=0.5, edgecolor='none', framealpha=0.8,
                        fontsize=fontsize-2, handler_map={FancyArrowPatch: HandlerPatch(patch_func=make_legend_quiver)})
        for handle in legend.legendHandles:
            if hasattr(handle, "set_markersize"):
                handle.set_markersize(6)
                handle.set_alpha(1)

        cbar = fig.colorbar(im, ax=ax, shrink=0.4, aspect=15, pad=0.04)
        cbar.solids.set_alpha(1)
        cbar.ax.tick_params(labelsize=fontsize-3)
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_powerlimits((-2, 3))
        cbar.ax.yaxis.set_major_formatter(formatter)
        offset = cbar.ax.yaxis.get_offset_text()
        offset.set_x(2.5); offset.set_y(0); offset.set_fontsize(fontsize-3)

    plt.tight_layout()
    if SAVE: plt.savefig("../figures/phase_space/" + file_path.split('.')[0] + "_all_metrics.png", dpi=300)
    plt.close()

    #break