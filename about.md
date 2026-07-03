---
layout: page
title: About Me
---

<style>
  .about-container {
    display: grid;
    grid-template-columns: minmax(250px, 300px) minmax(0, 1fr);
    gap: 2.5rem;
    align-items: flex-start;
  }

  .about-left {
    padding: 1.25rem;
    border: 1px solid #e7e1d8;
    border-radius: 8px;
    background: #fffdfa;
  }

  .about-left img {
    width: 144px;
    height: 144px;
    border-radius: 50%;
    object-fit: cover;
    margin-bottom: 1rem;
    border: 3px solid #fff;
    box-shadow: 0 6px 18px rgba(40, 37, 34, .12);
  }

  .profile-info {
    list-style: none;
    padding: 0;
    font-size: 0.95rem;
    color: #4c4742;
  }

  .profile-info li {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    min-width: 0;
  }

  .profile-info svg {
    flex: 0 0 auto;
    width: 20px;
    height: 20px;
    fill: currentColor;
  }

  .profile-info a {
    flex: 0 0 auto;
  }

  .about-title {
    margin-top: 0;
    margin-bottom: .35rem;
  }

  .about-kicker {
    margin-bottom: 1rem;
    color: #7a7168;
    font-family: "PT Sans", Helvetica, Arial, sans-serif;
    font-size: .9rem;
    letter-spacing: .02em;
    text-transform: uppercase;
  }

  .about-copy {
    line-height: 1.7;
  }

  .focus-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: .75rem;
    margin-top: 1.25rem;
  }

  .focus-item {
    padding: .85rem;
    border: 1px solid #e7e1d8;
    border-radius: 8px;
    background: #fffdfa;
    font-family: "PT Sans", Helvetica, Arial, sans-serif;
    font-size: .88rem;
    color: #514b45;
  }

  .focus-item strong {
    display: block;
    margin-bottom: .15rem;
    color: #2f2b28;
  }

  @media (max-width: 38em) {
    .about-container,
    .focus-list {
      grid-template-columns: 1fr;
    }
  }

</style>

<div class="about-container">
  <div class="about-left">
    <!-- Profile picture -->
    <img src="{{ '/assets/images/profile/profile.png' | relative_url }}" alt="Alessandro Carraro">

    <!-- Profile info -->
    <ul class="profile-info">
      <li>
        <!-- GitHub icon SVG -->
        <a href="https://github.com/alecarraro" target="_blank" title="GitHub">
          <svg viewBox="0 0 24 24"><path d="M12 0.5C5.65 0.5 0.5 5.65 0.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.3.8-.6v-2.1c-3.2.7-3.8-1.5-3.8-1.5-.5-1.2-1.2-1.5-1.2-1.5-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 .1 1.5-.8 1.5-.8.9-1.6 2.4-1.2 3-.9.1-.7.4-1.2.8-1.5-2.5-.3-5.2-1.2-5.2-5.5 0-1.2.4-2.2 1-3-.1-.3-.5-1.4.1-2.8 0 0 .8-.2 2.7 1 .8-.2 1.6-.3 2.5-.3s1.7.1 2.5.3c1.9-1.2 2.7-1 2.7-1 .6 1.4.2 2.5.1 2.8.6.8 1 1.8 1 3 0 4.3-2.7 5.2-5.3 5.5.4.3.7 1 .7 2v3c0 .3.2.7.8.6 4.6-1.5 7.9-5.8 7.9-10.9 0-6.35-5.15-11.5-11.5-11.5z"/></svg>
        </a> alecarraro
      </li>
      <li>
        <!-- LinkedIn icon SVG -->
        <a href="https://www.linkedin.com/in/alessandro-carraro-162761387/" target="_blank" title="LinkedIn">
          <svg viewBox="0 0 24 24"><path d="M4.98 3.5a2.5 2.5 0 1 1-.001 5.001A2.5 2.5 0 0 1 4.98 3.5zM3 9h4v12H3V9zm7 0h3.6v1.7h.05c.5-.9 1.7-1.8 3.45-1.8 3.7 0 4.4 2.4 4.4 5.5V21h-4v-5.5c0-1.3-.03-3-1.85-3-1.85 0-2.14 1.5-2.14 2.9V21h-4V9z"/></svg>
        </a>
        Alessandro Carraro
      </li>
      <li>
        📍 Stockholm, Sweden
      </li>
    </ul>
  </div>

  <div class="about-right about-copy">
    <p class="about-kicker">Computational mathematics, HPC, and scientific software</p>
    <h2 class="about-title">Hi, I am Alessandro.</h2>

    <p>
      I completed a master's degree in Computational Mathematics through the COSSE programme at TU Delft and KTH. I will soon start a PhD at the University of Edinburgh, working on structured linear algebra for efficient representation learning.
    </p>

    <p>
      My interests sit at the intersection of high-performance linear algebra, GPU programming, and numerical software. I mostly work in C++ and Julia, and I enjoy turning mathematical structure into implementations that are both fast and understandable.
    </p>

    <p>
      I contribute to open-source scientific computing projects, including work on <a href="https://github.com/NextLinearAlgebra/NextLA.jl">NextLA.jl</a> and previous contributions to the <a href="https://juliareach.github.io/">JuliaReach</a> ecosystem for reachability analysis of dynamical systems.
    </p>

    <div class="focus-list">
      <div class="focus-item">
        <strong>Research</strong>
        Low-rank approximations, mixed precision, and parallel Jacobi methods.
      </div>
      <div class="focus-item">
        <strong>Software</strong>
        CUDA programming, high-performance linear algebra, Julia, and C++.
      </div>
    </div>
  </div>
</div>
