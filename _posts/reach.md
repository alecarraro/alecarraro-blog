---
layout: post
title: GSoC 2025 Final Report – ReachabilityAnalysis.jl
---

<div class="message">
  Howdy! This post summarizes my GSoC 2025 work with the JuliaReach organization on ReachabilityAnalysis.jl under the guidance of mentors Christian Schilling and Marcelo Forets.
</div>

## Outline and Motivation

Safety-critical systems—such as autonomous vehicles and control systems—often operate under uncertainty, making the computation of **reachable sets** a critical task. In [ReachabilityAnalysis.jl](https://github.com/JuliaReach/ReachabilityAnalysis.jl), several algorithms exist to compute set enclosures (flowpipes) for dynamical systems, including hybrid systems with uncertain inputs and initial states.

During this project, I implemented an algorithm for dynamical systems of the form:

<div style="text-align:center;">
  <em>x' = A x</em>
</div>

where both the **initial state** and the **state matrix A** are uncertain. Multiple set representations exist for modeling uncertainty, including **interval matrices**, which capture bounded uncertainty in each entry of A.  

Althoff introduced **matrix zonotopes**, which are the matrix counterpart of standard zonotopes and can be written as:

<div style="text-align:center;">
  <!-- Add your equation here -->
  \( \mathcal{Z} = \{ M_0 + \sum_{i=1}^p M_i \alpha_i \mid \alpha_i \in [-1,1] \} \)
</div>

The main advantage of matrix zonotopes is that they allow us to model the uncertainty in **each entry of the state matrix independently**, often yielding a **tighter enclosure** than an interval matrix.  

This representation is particularly powerful for reachability analysis of linear systems with uncertain parameters, as it can reduce over-approximation while remaining computationally efficient.
