---
layout: post
title: GSoC 2025 Project with JuliaReach
---

<div class="message">
  Hey! This post summarizes my GSoC 2025 work with the JuliaReach organization on ReachabilityAnalysis.jl under the guidance of mentors 
  <a href="https://github.com/schillic">Christian Schilling</a> and 
  <a href="https://github.com/mforets">Marcelo Forets</a>.
</div>

## Table of Contents
1. [Outline and Motivation](#outline-and-motivation)
2. [Example](#Example)
3. [Contribution Overviews](#contributions-overview)
4. [References](#references)

## Outline and Motivation

In complex or safety-critical systems, it is often impossible to predict exactly how the system will behave due to uncertainties in initial states, inputs, or dynamics. Reachability analysis addresses this by computing *flowpipes* that enclose all possible behaviors over time, allowing engineers to rigorously verify safety, detect potential failures, and guide robust system design.  

In [ReachabilityAnalysis.jl](https://github.com/JuliaReach/ReachabilityAnalysis.jl), several algorithms exist to compute these enclosures for different types of dynamical systems with uncertain inputs and initial states, including hybrid systems.  

During my project, I focused on a fundamental class of models: linear time-invariant (LTI) systems with uncertain parameters.  
An LTI system describes the evolution of the state vector $$x(t)$$ under a fixed matrix $$A$$:

$$x' = A x$$

where $$x \in \mathbb{R}^n$$.  
Such systems are widely used because they arise naturally when linearizing nonlinear models. In practice, however, the dynamics are rarely known exactly. Parameters in $$A$$ may be uncertain due to modeling errors, noisy identification, or faulty sensors. To capture this, we consider a parametric LTI system of the form:  

$$x' = A x, \quad x(0) \in X_0, \quad A \in \mathcal{A}$$

Here:  
- $$X₀$$ is the set of possible initial states, reflecting uncertainty at $$t=t_0$$.
- $$\mathcal{A}$$ is a set of possible system matrices, representing parameter uncertainty.

The goal is to compute a tight over-approximation of all trajectories that can arise from *any* choice of initial state in $$X₀$$ and *any* admissible dynamics in $$\mathcal{A}$$. This is significantly more challenging than the classical case with a fixed $$A$$, since the solution space must account for both state uncertainty and dynamics uncertainty simultaneously.

Huang et al. [1] introduced a dedicated algorithm for these parametric LTI systems with uncertain dynamics, which combines two modern set representations: matrix zonotopes for modeling uncertainty in the system matrix and sparse polynomial zonotopes (SPZs) for representing the reachable states.  
The motivation for this choice becomes clear when we compare possible uncertainty models for $$A$$ and discuss how to best enclose the evolving states.

---

### Why Matrix Zonotopes Instead of Interval Matrices?  

A classical way to represent uncertainty in $$A$$ is through an interval matrix:

$$ \mathcal{A} = \{ A \mid a_{ij} \in [\underline{a}_{ij}, \overline{a}_{ij}] \} $$

This representation assumes that each entry of the matrix can vary independently within its interval. While simple and convenient, this independence assumption can be overly conservative: it includes combinations of parameters that may never actually occur in practice, often resulting in loose over-approximations of the reachable set.  

To address this, Althoff [2] introduced matrix zonotopes, which generalize standard zonotopes to matrices:

$$ \mathcal{Z} = \{ A_0 + \sum_{i=1}^p A_i \alpha_i \;\mid\; \alpha_i \in [-1,1] \} $$

Here, $$A_0$$ is the center matrix, and the $$A_i$$ are generator matrices. Matrix zonotopes preserve correlations between matrix entries, unlike interval matrices that treat them as independent. This leads to significantly tighter enclosures for uncertain dynamics, making them a more accurate and efficient choice in reachability analysis.

---

### Why Sparse Polynomial Zonotopes for States?  

Once the dynamics are modeled with matrix zonotopes, the reachable states need a representation that can handle the resulting complexity without becoming overly conservative.  
Huang et al. use aparse polynomial zonotopes for this task.

SPZs extend zonotopes with polynomial terms, allowing them to capture non-convex sets while remaining closed under Minkowski sums and linear maps, the core operations in reachability analysis. Their sparse structure keeps the representation efficient and makes them a natural fit alongside matrix zonotopes.  

For a more detailed description of SPZ see [3], or for a friendly introduction, see Luca’s previous GSoC [blog post](https://www.lucaferranti.com/posts/2022/09/gsoc22/)  

## Example

## Contributions Overview
In the tables below I summarize my contributions to the different packages in the `JuliaReach` ecosystem to implement the reachability algorithm described above.

### 1. `LazySets.jl`: Matrix Zonotopes  

| PR Title | PR # | Category |
|----------|------|----------|
| Add `overapproximate` for matrix zonotope exponential | [#4000](https://github.com/JuliaReach/LazySets.jl/pull/4000) | Feature |
| Add operations on matrix zonotopes | [#3999](https://github.com/JuliaReach/LazySets.jl/pull/3999) | Feature |
| Add `overapproximate` for matrix zonotope multiplication | [#3996](https://github.com/JuliaReach/LazySets.jl/pull/3996) | Feature |
| Preserve `indexvector` in linear map for MZ | [#3985](https://github.com/JuliaReach/LazySets.jl/pull/3985) | Fix |
| Extend `ExponentialMap` to support `MatrixZonotope`s | [#3970](https://github.com/JuliaReach/LazySets.jl/pull/3970) | Feature |
| Add `linear_map` between MatrixZonotope and SPZ | [#3969](https://github.com/JuliaReach/LazySets.jl/pull/3969) | Feature |
| Fix in-place scaling for `MatrixZonotope` | [#3967](https://github.com/JuliaReach/LazySets.jl/pull/3967) | Fix |
| Add methods and type for `MatrixSets` | [#3966](https://github.com/JuliaReach/LazySets.jl/pull/3966) | Feature |
| Add `norm` and `overapproximate_norm` for matrix zonotope | [#3941](https://github.com/JuliaReach/LazySets.jl/pull/3941) | Feature |
| Refactor and extend `MatrixSets` module | [#3933](https://github.com/JuliaReach/LazySets.jl/pull/3933) | Refactor |

---

### 2. `LazySets.jl`: General improvements and fixes  

| PR Title | PR # | Category |
|----------|------|----------|
| Optimize `remove_redundant_generators` | [#3998](https://github.com/JuliaReach/LazySets.jl/pull/3998) | Feature |
| Improve `remove_redundant_generators` for `SPZ` | [#3986](https://github.com/JuliaReach/LazySets.jl/pull/3986) | Improvement |
| Add `overapproximate` for exponential map for `SPZ` and `Zonotope` | [#3979](https://github.com/JuliaReach/LazySets.jl/pull/3979) | Feature |
| Fix `isuniversal` and constructor for `Polygon` | [#3978](https://github.com/JuliaReach/LazySets.jl/pull/3978) | Fix |
| Preserve ID in `minkowski_sum` and `cartesian_product` between `SPZ` and `Zonotope` | [#3977](https://github.com/JuliaReach/LazySets.jl/pull/3977) | Improvement |
| Make `overapproximate` of ASPZ with `UnionSetArray{Zonotope}` more robust | [#3972](https://github.com/JuliaReach/LazySets.jl/pull/3972) | Improvement |
| Add overapproximation of the l1 norm of a `Zonotope` | [#3925](https://github.com/JuliaReach/LazySets.jl/pull/3925) | Feature |
| Faster l1 norm for `AbstractZonotope` | [#3924](https://github.com/JuliaReach/LazySets.jl/pull/3924) | Improvement |
| Add `scale` for `SPZ` | [#3908](https://github.com/JuliaReach/LazySets.jl/pull/3908) | Feature |
| Add `merge_id` and generalize `exact_sum` for `SPZ` | [#3905](https://github.com/JuliaReach/LazySets.jl/pull/3905) | Feature |
| Add sampler for sparse polynomial zonotopes | [#3847](https://github.com/JuliaReach/LazySets.jl/pull/3847) | Feature |

---

### 3. `MathematicalSystems.jl` and `ReachabilityAnalysis.jl`

| PR Title | PR # | Notes |
|----------|------|-------|
| Add `parametric` systems | [#332](https://github.com/JuliaReach/MathematicalSystems.jl/pull/332) | New feature |
| Add HLBS25 algorithm for linear parametric systems| #XXX | New feature |

## References
[1] Yushen Huang, Ertai Luo, Stanley Bak, Yifan Sun, *Reachability analysis for linear systems with uncertain parameters using polynomial zonotopes*, Nonlinear Analysis: Hybrid Systems, Volume 56, 2025, 101571, ISSN 1751-570X. [DOI](https://doi.org/10.1016/j.nahs.2024.101571)

[2] Matthias Althoff, *Reachability Analysis and its Application to the Safety Assessment of Autonomous Cars*, PhD thesis, Technische Universität München, July 2010.

[3] Niklas Kochdumper, Matthias Althoff, *Sparse Polynomial Zonotopes: A Novel Set Representation for Reachability Analysis*, IEEE Transactions on Automatic Control, Vol. 66, No. 9, Sept. 2021, pp. 4043–4058. [DOI](https://doi.org/10.1109/TAC.2020.3024348)
