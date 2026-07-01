# Automated Rational Catalyst Design (ARCADE)
### A Workflow for Fully Automated and Interpretable Design of Complex Catalysts

ARCADE is a fully automated and physics-interpretable computational workflow explicitly tailored for decoding and designing complex multi-element catalysts (e.g., medium- and high-entropy alloys). By seamlessly bridging machine learning accelerated structure exploration with rigorous multi-scale catalytic screening, ARCADE shifts the paradigm from Edisonian trial-and-error to the rational discovery of optimal local chemical environments.

---

## 🛠️ Computational Workflow

The ARCADE ecosystem is orchestrated into three core methodological engines and a benchmarking dataset repository. To navigate the code, follow the pipeline execution order below:

[ 01_ApolloX_v2 ]  --> Structure Generation & Energetic Ranking
                      │
                      ▼
    [ 02_adsorption_site_processing ] --> Screening & Matching Candidates
                      │
                      ▼
   [ 03_contribution_score_evaluation ] --> SRO-Property Quantitative Mapping

### 📂 Repository Structure

* **[`01_ApolloX_v2/`](./01_ApolloX_v2/)**: *Core Structure Generation Engine.* Contains the Monte Carlo engines integrated with Particle Swarm Optimization (PSO) and Machine Learning Potentials (MLP) for high-throughput exploration of vast configurational spaces. 📖 *[See detailed documentation](./01_ApolloX_v2/README.md)*
* **[`02_adsorption_site_processing/`](./02_adsorption_site_processing/)**: *Adsorption Coordination Search Engine.* Contains automated pipelines for intermediate screening and geometric structural pattern matching. 📖 *[See detailed documentation](./02_adsorption_site_processing/README.md)*
    * `screening/`: Automated 100% full-relaxation candidate construction for intermediates ($^*\text{COOH}$, $^*\text{OCHO}$, $^*\text{H}$).
    * `matching/`: Structural pattern matching protocol based on local environment descriptors.
* **[`03_contribution_score_evaluation/`](./03_contribution_score_evaluation/)**: *Interpretability Engine.* Computes the statistical contribution scores ($S_{\mathit{stab}}$, $S_{\mathit{act}}$, $S_{\mathit{sel}}$) to quantitatively map short-range order (SRO) metrics (Warren–Cowley parameters $\alpha_{ij}$ and Local Density Deviation $\delta_j$) to macroscopic catalytic properties.
* **[`data/`](./data/)**: *Comprehensive Multi-Scale Dataset.*
    * `structure_modeling/`: Ensembles, configurations, and thermodynamic metrics from ApolloX.
    * `adsorption_sites_screening/`: Complete DFT-relaxed adsorption structures and raw binding energies.
    * `activity_analysis/`: Microkinetic modeling inputs/outputs and volcano curve projections.
    * `selectivity_analysis/`: Thermodynamic $\Delta\Delta G$ landscapes and kinetic selectivity matching matrices.

---

## 🚀 Getting Started

### Prerequisites
* Python $\ge$ 3.9
* ASE (Atomic Simulation Environment)

### Installation
Clone the repository and set up the local environment:
```bash
git clone [https://github.com/jianzhuow/ARCADE.git](https://github.com/jianzhuow/ARCADE.git)
cd ARCADE
