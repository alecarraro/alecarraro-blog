---
layout: post
title: Hub laplacian operators for directional Graph Neural Networks
image: /assets/images/cover_gnn.png
---

<div class="message">
  <p>
    In this post, I want to summarize some ideas and results we explored last year during a self-proposed research project on for a course on machine learning for graph data. The code is available <a href="https://github.com/alecarraro/hub-laplacian-4-anisotropic-gnns">here</a>.
  </p>

  <p style="margin-top: 1em; font-style: italic; color: #555;">
    Friendly head-up: the code is a bit messy in places, especially the GAD part, which will be refactored at some point. Check at your own risk!
  </p>
</div>


## Table of Contents
1. [Introduction](#introduction)
2. [The idea: Hub-Laplacian operators](#The-idea:-Hub-Laplacian-operators)
3. [Methodology and Results](#methodology-and-results)
4. [Mathematical Analysis](#mathematical-analysis)
5. [References](#references)

## Introduction

In recent years, diffusion-based message passing models have become a popular paradigm for tasks in graph machine learning, with notable works by Bronstein [1], Chamberlain [2], and others. However, these models suffer from two opposing problems:

- **Over-smoothing:** After many layers, all node embeddings become indistinguishable [3].  
- **Over-squashing:** Long-range information gets compressed through narrow paths, limiting expressivity [4].

Much of the recent literature focuses on solving these issues by proposing anisotropic diffusion [5] or message passing based on advection-diffusion PDEs [6]. We decided to take a different route: instead of modifying existing schemes, can we use a different graph shift operator (instead of the normalized adjacency or Laplacian) that encodes geometry differently?  

To this end, we employed hub-biased diffusion via the Hub-Laplacian operator [7], a new degree-weighted variant of the graph Laplacian that can *attract* or *repel* information flow from highly connected nodes.

## The idea: Hub-Laplacian operators

The standard combinatorial graph Laplacian is
$$
L = D - A
$$
where $$A$$ is the adjacency matrix and $$D = \mathrm{diag}(d_v)$$ the degree matrix. The *Hub-Laplacian* (Miranda et al. [7, 8, 9]) biases diffusion toward or away from high-degree nodes by re-weighting neighbour contributions:

$$
L_{\alpha} = \Xi_{\alpha} - D^{\alpha} A D^{-\alpha},
\qquad \text{where} \qquad
\Xi_{\alpha} = \mathrm{diag}\Bigg(\sum_{w\in\mathcal N(v)} \Big(\frac{d_w}{d_v}\Big)^{\alpha}\Bigg)
$$

Intuitively:

* $$\alpha>0$$ is **hub-attracting**: edges are biased so that information preferentially flows *toward* higher-degree neighbors.
* $$\alpha<0$$ is **hub-repelling**: information tends to avoid high-degree neighbors.
* $$\alpha=0$$ recovers the usual combinatorial Laplacian (since $$(d_w/d_v)^0 = 1$$ and $$\Xi_0 = D$$).

Tuning $$\alpha$$ gives a continuous family of graph shift operators that interpolates between symmetric, undirected diffusion $$\alpha = 0$$ and increasingly directional, degree-biased diffusion as $$\alpha$$ grows. 
In the extreme limit $$\alpha \to +\infty$$ the dynamics concentrate on edges toward strictly larger-degree neighbors (the opposite happens for $$\alpha \to -\infty$$), acting intuitively like a sort of weighted directed graph.

Thus learning a scalar $$\alpha$$ lets the model smoothly move between (a) standard undirected diffusion and (b) strongly degree-directed flows. This provides a way of incorporating directionality tuning only one parameter, so it is a compromise between fixed GSOs and fully learned shift operators.

We can also form a normalized hub operator that mirrors the normalized Laplacian. One convenient form is

$$
\widehat L_{\alpha} = \Xi_{\alpha} - D^{-1/2-\alpha} A D^{-1/2+\alpha}
$$

which is the natural degree-scaled analogue of $$L_\alpha$$. When $$\alpha=0$$ this operator is congruent to the usual normalized Laplacian $$I - D^{-1/2} A D^{-1/2}$$ (they share the same nonzero spectrum up to the congruence $$D^{1/2} (\cdot) D^{1/2}$$).

## Methodology
We focused on **graph regression** using a subset of the [QM9 dataset](https://www.kaggle.com/datasets/zaharch/quantum-machine-9-aka-qm9), commonly used to benchmark diffusion-based message passing schemes.  
To keep the experiments simple and interpretable, we experimented with two main architectures:

### 1. Graph Convolutional Neural Network (GCNN)

Our GCNN layers follow the standard polynomial filter formulation:

$$
H(X) = \sum_{k=0}^{K} S^k X W_k
$$

where $$S$$ is the graph shift operator (in our case, $$L_{\alpha}$$), and $$W_k$$ are learnable weights.  

The values of $$\alpha$$ are learnable and updated during training, requiring recomputation of $$L_{\alpha}$$ at each epoch. This allows the network to adaptively determine whether hub-attracting or hub-repelling diffusion is beneficial.

To obtain graph-level representations, we experimented with **mean**, **max**, and **sum pooling**, keeping results interpretable.

### 2. Graph Anisotropic Diffusion
Graph Anisotropic Diffusion (GAD) was introduced in [1]. In each layer, it performs one discrete step of

$$\frac{d}{dt} \mathbf{X}^l = \mathbf{D}^{-1} \mathbf{L} \mathbf{X}^l$$ 

and then propagates information through message passing using the aggregators $$\{\text{mean, max, min, } dx1\}$$ where $$dx1$$ denotes the operation:

$$\mathbf{M}^l = \mathbf{A}\left\{\left[\mathbf{X}^l\right]_j \mid j \in N_i\right\} = \left[\mathbf{X}^l, \mathbf{B}_{dx1} \mathbf{X}^l\right]^T$$ 

where 

$$\mathbf{B}_{dx1} = \mathbf{\hat{F}}_1 - \text{diag}\left(\sum_j \mathbf{\hat{F}}_{:,j}\right)$$

and $$\mathbf{\hat{F}}_1$$ is the row normalized map induced by the gradient of the Laplacian's first eigenvector (fiedler vector) values along the graph, as defined in [5].

We expanded this approach by allowing the use of $$\mathbf{L}_\alpha$$ and the hub-advection diffusion operator $$\mathbf{T}$$ in the diffusion step with or without learnable parameters, as well as for the eigenmaps $$\mathbf{F}$$.

## Results
I will spare the details of the hyperparameters and the grid search, but you can find more details in the [report](https://github.com/alecarraro/hub-laplacian-4-anisotropic-gnns/blob/master/Report_GNN_project.pdf) we wrote last year. 
Furthermore to keep this post coincise, I will show some results only for the GCNN architecture and

| Target | Validation MAE (Best) | Validation MAE (Baseline) | Test MAE (Best) | Test MAE (Baseline) | Initial $$\alpha$$ | Pooling |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Target 0 | **0.9687** | 1.0025 | **0.7805** | 1.0407 | 0.98 | Mean |
| Target 1 | **2.9704** | 3.2886 | **2.6446** | 3.3338 | 0.5 | Sum |
| Target 2 | **0.4952** | 0.5556 | **0.4269** | 0.5001 | 0.93 | Mean |


While these results show that the hub Laplacian consistently outperforms the baseline when paired with the optimal (grid-searched) pooling strategy for each target, it is important to recall the standard caveats of  machine learning. The superior performance of the hub Laplacian does not necessarily imply its general superiority, as our experiments were performed on a subset of a single dataset. Moreover a lot of optimization and better architectural choices can be made. Neverthless the results are somewhat interesting and promising.

Furthermore, as is often the case in machine learning, it is difficult to provide an explanation for these effects that goes beyond initial intuition. However, we can make some mathematical guesses regarding the hub laplacian efficacy, which will be explored in the following section.


## References
[1] Ahmed A. A. Elhag, Gabriele Corso, Hannes Stärk, and Michael M. Bronstein. *Graph Anisotropic Diffusion*, 2022. [arXiv:2205.00354](https://arxiv.org/abs/2205.00354)  

[2] Benjamin Paul Chamberlain, James Rowbottom, Maria I. Gorinova, Stefan Webb, Emanuele Rossi, and Michael M. Bronstein. *GRAND: Graph Neural Diffusion*. CoRR, abs/2106.10934, 2021. [arXiv:2106.10934](https://arxiv.org/abs/2106.10934)  

[3] T. Konstantin Rusch, Michael M. Bronstein, and Siddhartha Mishra. *A survey on oversmoothing in graph neural networks*. [arXiv:2303.10993](http://arxiv.org/abs/2303.10993)  

[4] Singh Akansha. *Over-squashing in graph neural networks: A comprehensive survey*. [arXiv:2308.15568](http://arxiv.org/abs/2308.15568)   

[5] Dominique Beaini, Saro Passaro, Vincent Létourneau, William L. Hamilton, Gabriele Corso, and Pietro Liò. *Directional graph networks*. [arXiv:2010.02863](http://arxiv.org/abs/2010.02863)  

[6] Yifan Qu, Oliver Krzysik, Hans De Sterck, and Omer Ege Kara. *First-order PDEs for graph neural networks: Advection and Burgers equation models*, 2024. [arXiv:2404.03081](https://arxiv.org/abs/2404.03081)

[6] Moshe Eliasof, Eldad Haber, and Eran Treister. *Graph Neural Reaction Diffusion Models*. SIAM Journal on Scientific Computing, 46(4):C399–C420, 2024. [DOI:10.1137/23M1576700](https://doi.org/10.1137/23M1576700)  

[7] Ernesto Estrada and Delio Mugnolo. *Hubs-biased resistance distances on graphs and networks*, 507(1):125728. [arXiv:2101.07103](http://arxiv.org/abs/2101.07103)  

[8] Manuel Miranda and Ernesto Estrada. *Degree-biased advection–diffusion on undirected graphs/networks*, 17:30. [MMNP Journal](https://www.mmnp-journal.org/10.1051/mmnp/202203

[9] Lucia Valentina Gambuzza, Mattia Frasca, and Ernesto Estrada. *Hubs-attracting Laplacian and related synchronization on networks*, 19(2):1057–1079. [SIAM Journal](https://epubs.siam.org/doi/10.1137/19M1287663)  

[10] Mingqi Yang, Yanming Shen, Rui Li, Heng Qi, Qiang Zhang, and Baocai Yin. *A new perspective on the effects of spectrum in graph neural networks*, 2022. [arXiv:2112.07160](https://arxiv.org/abs/2112.07160)  

[11] Weichen Zhao, Chenguang Wang, Xinyan Wang, Congying Han, Tiande Guo, and Tianshu Yu. *Understanding oversmoothing in diffusion-based GNNs from the perspective of operator semigroup theory*, 2025. [arXiv:2402.15326](https://arxiv.org/abs/2402.15326)  
