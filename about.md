---
layout: page
title: About Me
---

<style>
  .about-container {
    display: flex;
    flex-wrap: wrap;
    gap: 2rem;
    align-items: flex-start;
  }

  .about-left {
    flex: 1 1 250px;
    max-width: 250px;
  }

  .about-left img {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    object-fit: cover;
    margin-bottom: 1rem;
  }

  .profile-info {
    list-style: none;
    padding: 0;
    font-size: 0.95rem;
    color: #333;
  }

  .profile-info li {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }

  .profile-info svg {
    width: 20px;
    height: 20px;
    fill: currentColor;
  }

  .about-right {
    flex: 2 1 400px;
    min-width: 300px;
  }

  .message {
    line-height: 1.6;
    margin-bottom: 1rem; /* space between paragraphs */
  }

  .resume-link {
    display: inline-block;
    margin-top: 1rem;
    font-weight: bold;
  }
</style>

<div class="about-container">
  <div class="about-left">
    <!-- Profile picture -->
    <img src="{{ '/assets/images/profile.jpg' | relative_url }}" alt="Alessandro Carraro">

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
        <a href="https://linkedin.com/in/alessandro-carraro" target="_blank" title="LinkedIn">
          <svg viewBox="0 0 24 24"><path d="M4.98 3.5a2.5 2.5 0 1 1-.001 5.001A2.5 2.5 0 0 1 4.98 3.5zM3 9h4v12H3V9zm7 0h3.6v1.7h.05c.5-.9 1.7-1.8 3.45-1.8 3.7 0 4.4 2.4 4.4 5.5V21h-4v-5.5c0-1.3-.03-3-1.85-3-1.85 0-2.14 1.5-2.14 2.9V21h-4V9z"/></svg>
        </a>
        Alessandro Carraro
      </li>
      <li>
        📍 Stockholm, Sweden
      </li>
    </ul>
  </div>

  <div class="about-right">
    <p class="message">
      Hi, I’m Alessandro Carraro. I am a 2nd-year Master’s student at KTH and TU Delft, enrolled in the joint programme Computer Simulation for Science and Engineering (COSSE).

      As a part of my education, I have followed courses on numerical methods, scientific and graph machine learning, and control theory.
  
      In my free time, I contribute to the <a href="https://juliareach.github.io/">JuliaReach</a> ecosystem, an open-source framework specialized in reachability analysis for dynamical systems.
    </p>

    <p class="resume-link">
      📄 You can view my resume here:  
      <a href="{{ '/assets/pdfs/resume_2025_website.pdf' | relative_url }}" target="_blank">PDF</a>
    </p>
  </div>
</div>
