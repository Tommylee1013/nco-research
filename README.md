## From Priors to Posteriors: Bayesian Views in Nested Clustered Optimization

This repository contains research code and experiments for **Posterior-NCO**,  
an extension of **Nested Clustered Optimization (NCO)** that incorporates **Managers’ Views**  
via Bayesian posterior covariance/correlation matrices.

The project is designed as an **undergraduate research paper** but aims at a level of rigor  
comparable to graduate-level portfolio optimization studies.

---

### Motivation

Classical **Markowitz mean–variance optimization** suffers from estimation error, especially in high-dimensional settings and under heavy-tailed distributions.  

Lopez de Prado (2016, 2020) proposed **NCO (Nested Clustered Optimization)** as a robust alternative using hierarchical clustering on correlation structures.  

In this work, we extend NCO by incorporating **posterior moments** informed by **Managers’ Views** (e.g., Black–Litterman, Entropy Pooling, correlation views).  
This yields what we call:
 
- **Posterior-NCO = NCO on posterior-informed covariance/correlation matrices.**

---

### Repository Structure

```angular2html
nco-research/
├── README.md
├── requirements.txt
└── setup.py
```
---

### Installation

```bash
git clone https://github.com/tommylee1013/nco-research.git
cd nco-research
pip install -r requirements.txt
```

Optional (editable install):

```bash
pip install -e .
```

### 🚀 Usage

#### 1. Run Monte Carlo Experiment

- use Black & Litterman Style update

```bash
python run_montecarlo_experiment.py --n_assets 40 --n_clusters 4 --n_in_sample 252 \
  --n_out_of_sample 252 --n_trials 50 --rho_in 0.65 --rho_out 0.05 --df 5 \
  --shrinkage lw_constant_corr --denoising mp_constant --detone \
  --use_views --view_branch black_litterman --bl_view_type pairwise --n_views 10 \
  --view_noise_std 1e-4 --view_confidence_scale 0.5
```

- use Correlation View Blending

```bash
python run_montecarlo_experiment.py --n_assets 40 --n_clusters 4 --n_in_sample 252 \
  --n_out_of_sample 252 --n_trials 50 --rho_in 0.65 --rho_out 0.05 --df 5 \
  --shrinkage lw_constant_corr --denoising mp_constant --detone \
  --use_views --view_branch corr_blend --beta_view 0.3 --intra_scale_view 1.1 \
  --inter_scale_view 0.7 --corr_view_noise_std 0.02
```

This simulates block-structured covariance matrices and compares OOS Sharpe ratios across:
- Markowitz (long-only max Sharpe)
- post-NCO (posterior-informed correlation)
- NCO (empirical correlation)
- IVP (inverse variance baseline)

#### 2. Analyze Results

```bash
python analyze_results.py
```

Generates tables and figures for OOS performance, Sharpe distribution, and win-rate comparisons (NCO vs Markowitz).

### 📊Experiment Design

- True Model: Block-structured covariance with intra-cluster correlation. 
- Noise: Small sample size, heavy-tailed returns (Student-t). 
- Methods:
  1. Markowitz
  2. Posterior-NCO (via BL posterior or correlation blending)
  3. NCO
  4. IVP
- Metrics: OOS Sharpe, volatility, max drawdown, concentration (HHI).
- Monte Carlo: Repeat 200–1000 trials, report distributions & win rates. 

This design follows Lopez de Prado’s methodology in demonstrating the robustness of NCO versus Markowitz, but adds a Bayesian twist.

### 📄 Paper

- **Tentative Title** :
  *Posterior-NCO* : A Bayesian Extension of Nested Clustered Optimization in Portfolio Selection
- Structure:
  1. Introduction (Markowitz limitations, NCO motivation)
  2. Literature Review (Markowitz, Black–Litterman, NCO)
  3. Methodology (Posterior covariance, posterior correlation, NCO recursion)
  4. Monte Carlo Simulation (Design + Results)
  5. Conclusion (Posterior-NCO outperforms Markowitz in OOS stability)

### 📚 References
- Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample. 
- Lopez de Prado, M. (2020). Advances in Financial Machine Learning. 
- Black, F., & Litterman, R. (1992). Global Portfolio Optimization. 
- Meucci, A. (2008). The Black–Litterman Approach. 
- Meucci, A. (2009). Managing Diversification.


