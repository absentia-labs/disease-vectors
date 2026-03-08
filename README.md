# Information geometry of disease

This repository collects code and results for analyzing the representations generated from a multimodal foundation model of scRNA with information geometry.

Information geometry studies the properties: lengths, angles, curvature of statistical or information manifolds. Manifolds made up of probability distributions. scRNA-seq foundation models have exploded in popularity in recent years and neural networks in general models a point as a probability distribution. We offer the first efforts in bridging these two disciplines of modeling high dimensional genetic expression as probability distributions of a given phenotype of interest e.g. disease and the study of statistical manifolds.

Model used that embeds a cell vector onto the manifold or organises data along a statistical axis of variation is a transformer encoder inspired off of BERT. 

Model for using the transformer architecture on tabular data. Learning relationships in scRNA metadata and scRNA gene expression with masked learning.
Training bash: ./train.sh


