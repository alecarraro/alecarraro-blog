---
layout: post
title: TLR Matrix Multiplication
---

## Introduction

Let $$A \in \mathbb{R}^{N \times N}$$ be a square matrix stored in tile low-rank (TLR) format. We assume that $$A$$ is partitioned into an $$n \times n$$ grid of tiles,

$$
A =
\begin{bmatrix}
A_{11} & \widetilde A_{12} & \cdots & \widetilde A_{1n} \\
\widetilde A_{21} & A_{22} & \cdots & \widetilde A_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\widetilde A_{n1} & \widetilde A_{n2} & \cdots & A_{nn}
\end{bmatrix}.
$$

The diagonal tiles $$A_{ii}$$ are stored densely, whereas the off-diagonal tiles $$\widetilde A_{ij}$$, with $$i \neq j$$, are stored in compressed low-rank form. In particular, each off-diagonal tile is represented as

$$
\widetilde A_{ij} = U_{ij} V_{ij}^{T},
$$

where $$U_{ij} \in \mathbb{R}^{b_i \times r_{ij}}$$, $$V_{ij} \in \mathbb{R}^{b_j \times r_{ij}}$$, and $$r_{ij}$$ denotes the numerical rank of the tile. Here, $$b_i$$ and $$b_j$$ denote the row and column tile sizes, respectively. For interior tiles, $$b_i = b_j = b$$, while boundary tiles may have smaller dimensions when $$N$$ is not an exact multiple of $$b$$.

The objective is to compute the generalized matrix multiplication

$$
C \leftarrow \alpha \operatorname{op}(A)\operatorname{op}(B) + \beta C,
$$

where $$\alpha$$ and $$\beta$$ are scalars, and

$$
\operatorname{op}(X) \in \{X, X^{T}\}
$$

denotes either the matrix itself or its transpose. The input matrices $$A$$ and $$B$$ may be stored either densely or in TLR format, and the output matrix $$C$$ may also be stored either densely or in TLR format.

A further distinction concerns the ranks of the compressed tiles. In the fixed-rank case, all off-diagonal tiles are assumed to have the same rank $$r$$. This may arise either from fixed-rank compression or from padding variable-rank factors to a common rank. In the variable-rank case, each tile is allowed to have an individual rank $$r_{ij}$$.

The implementation complexity increases significantly when the output matrix is both stored in TLR format with the tiles of the operands having variable-rank case. Therefore, the development is organized into three stages:

$$
\begin{aligned}
&\text{1. fixed rank:} \qquad &&\text{dense} \leftarrow \text{TLR} \times \text{TLR}, \\
&\text{2. fixed rank:} \qquad &&\text{TLR} \leftarrow \text{TLR} \times \text{TLR}, \\
&\text{3. variable rank:} \qquad &&\text{TLR} \leftarrow \text{TLR} \times \text{TLR}.
\end{aligned}
$$

We first describe the implementation of the fixed-rank dense-output case. The TLR-output cases are then obtained by extending this formulation.

### Dense Accumulation with Fixed Rank

The goal of the first implementation stage is to compute

$$
C \leftarrow AB,
$$

where both $$A$$ and $$B$$ are stored in fixed-rank TLR format and $$C$$ is stored densely. The main implementation objective is to express the computation in terms of vendor-optimized BLAS primitives, in particular `GEMM` and batched `GEMM` variants.

To expose such regular operations, the memory layout of the TLR matrices must be chosen carefully. A TLR matrix can be stored using three buffers, one for the dense diagonal tile, represented by a 3d array of size $$b\times b \times n$$ such that each tile is written in memory in column major storage.

Similarly the factors $$U$$ and $$V$$ can also be written in memory in two distinct buffers, where each factor is laid out column major and the factors are linearized according to either tile row major or tile col major order. In the case where the tiles have the same blocksize $$b$$ and rank $$r$$, there is a constant stride among consecutive factors and can be written as a 3d array as well. In any other case we can store an array with the ranks $$r$$ and block size $$b$$ and find the ptr location using prefix sum.

For the case the global matrix dimension $$N$$ is not an exact multiple of the nominal tile size $$b$$. Let

$$
N = mb + b_{\mathrm{rem}},
$$

where $$m = \lfloor N/b \rfloor$$ and $$0 \leq b_{\mathrm{rem}} < b$$. If $$b_{\mathrm{rem}} = 0$$, all tiles have size $$b \times b$$. Otherwise, the last block row and last block column contain boundary tiles with smaller dimensions. In that case, the interior part of the matrix has dimension $$mb \times mb$$, while the boundary block has size $$b_{\mathrm{rem}}$$.

For this reason, it is convenient to decompose the matrix into a regular interior TLR block and an irregular boundary part. When $$b_{\mathrm{rem}} > 0$$, we write

$$
A =
\begin{bmatrix}
A_{\mathrm{int}} & u_A \\
v_A^{T} & \gamma_A
\end{bmatrix},
$$

where

$$
A_{\mathrm{int}} \in \mathbb{R}^{mb \times mb}, 
\qquad
u_A \in \mathbb{R}^{mb \times b_{\mathrm{rem}}},
\qquad
v_A \in \mathbb{R}^{mb \times b_{\mathrm{rem}}},
\qquad
\gamma_A \in \mathbb{R}^{b_{\mathrm{rem}} \times b_{\mathrm{rem}}}.
$$

Here, $$A_{\mathrm{int}}$$ denotes the regular TLR submatrix formed by the full-size tiles, while $$u_A$$, $$v_A^{T}$$, and $$\gamma_A$$ collect the boundary column, boundary row, and bottom-right corner block, respectively. Similarly, we write

$$
B =
\begin{bmatrix}
B_{\mathrm{int}} & u_B \\
v_B^{T} & \gamma_B
\end{bmatrix}.
$$

With this notation, the product $$C = AB$$ can be written as

$$
C =
\begin{bmatrix}
A_{\mathrm{int}} & u_A \\
v_A^{T} & \gamma_A
\end{bmatrix}
\begin{bmatrix}
B_{\mathrm{int}} & u_B \\
v_B^{T} & \gamma_B
\end{bmatrix}.
$$

Carrying out the block multiplication gives

$$
C =
\begin{bmatrix}
A_{\mathrm{int}} B_{\mathrm{int}} + u_A v_B^{T}
&
A_{\mathrm{int}} u_B + u_A \gamma_B
\\
v_A^{T} B_{\mathrm{int}} + \gamma_A v_B^{T}
&
v_A^{T} u_B + \gamma_A \gamma_B
\end{bmatrix}.
$$

Equivalently, this can be decomposed as

$$
C =
\begin{bmatrix}
A_{\mathrm{int}} B_{\mathrm{int}} & A_{\mathrm{int}} u_B \\
v_A^{T} B_{\mathrm{int}} & v_A^{T} u_B
\end{bmatrix}
+
\begin{bmatrix}
u_A v_B^{T} & u_A \gamma_B \\
\gamma_A v_B^{T} & \gamma_A \gamma_B
\end{bmatrix}.
$$

This decomposition separates the computation into four output regions:

$$
C =
\begin{bmatrix}
C_{\mathrm{int}} & C_{\mathrm{right}} \\
C_{\mathrm{bottom}} & C_{\mathrm{corner}}
\end{bmatrix}.
$$

The interior block $$C_{\mathrm{int}}$$ contains the product of the regular TLR interiors, together with a dense boundary update. The right and bottom boundary regions correspond to products between a TLR interior and a dense boundary panel. The corner block is a small dense matrix product.

This structure is useful because the dominant computation is isolated in the regular interior block $$A_{\mathrm{int}} B_{\mathrm{int}}$$, whose tiles all have size $$b \times b$$ and whose low-rank factors all have fixed rank $$r$$. As a result, the corresponding tile products can be grouped into batches with identical dimensions and lowered to batched BLAS kernels. The remaining boundary updates involve dense panels and small dense blocks, and can be handled separately using standard `GEMM` calls. Thus, the decomposition enables the implementation to maintain regular computation shapes for the main workload, while still supporting arbitrary matrix dimensions $$N$$.
