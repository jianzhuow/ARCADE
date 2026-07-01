# Automated Rational Catalyst Design (ARCADE)

ARCADE (Automated Rational CAtalyst DEsign) is a generalizable framework integrated within the Digital Catalysis Platform (DigCat: www.digcat.org) to address the multifaceted challenges in multi-element catalyst design. This generalizable framework encompasses thermodynamically stable nanoparticle generation by introducing SRO-based structural features, automated active-site identification and binding energy evaluation, selectivity analysis through active-site matching and adsorption energies comparisons, activity analysis via advanced pH-dependent microkinetic modeling, and comprehensive mechanism elucidation across stability, selectivity, and activity via SRO-derived contribution scores, representing methodological breakthroughs across different aspects of catalyst design.

This repository serves as a functional programmatic demonstration of ARCADE, containing the core code for executing the workflow alongside the datasets reported in our study.

---

## Repository Structure

The workflow is organized into core programmatic modules and a comprehensive data repository. **For specific usage instructions, environment configurations, and technical notes for each component, please refer to the README file within the respective directory.**

* **`01_ApolloX_v2/`**: Core structure generation engine. Contains the Monte Carlo generative engines integrated with Particle Swarm Optimization iterative algorithm for high-throughput exploration of vast configurational spaces.
* **`02_adsorption_site_processing/`**: Adsorption configuration search engine. Handles the automated candidate construction for intermediates based on Voronoi analysis and the structural pattern matching protocol based on local environment descriptors.
* **`03_contribution_score_evaluation/`**: Interpretability engine. Computes the statistical contribution scores to quantitatively map short-range order metrics to macroscopic catalytic properties.
* **`data/`**: Complete multi-scale dataset tracking the workflow, partitioned into four key stages:
    * `structure_modeling/`: Ensembles, configurations, and thermodynamic metrics from ApolloX.
    * `adsorption_sites_screening/`: DFT-relaxed adsorption structures and binding energies.
    * `activity_analysis/`: Microkinetic modeling outputs, and volcano curve projections.
    * `selectivity_analysis/`: Thermodynamic energy landscapes and kinetic selectivity matching matrices.

