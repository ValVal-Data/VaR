
# Value at Risk (VaR) and Expected Shortfall (ES) Estimation for a Multi-Asset Portfolio

## Overview
This project develops and validates a market risk engine for estimating 1-week Value at Risk (VaR) and Expected Shortfall (ES) for a diversified multi-asset portfolio.

The framework benchmarks historical simulation, parametric approaches, and Monte Carlo simulation under increasingly realistic assumptions. In particular, it evaluates constant-volatility and time-varying volatility specifications, as well as Gaussian, Student-t, and skewed-t innovation distributions.

The analysis is designed as a model selection exercise rather than a single-model implementation. Each specification is assessed using distribution diagnostics, heteroskedasticity tests, dependence diagnostics, convergence analysis, and VaR backtesting. Over the sample considered, CCC-GARCH(1,1) with skewed-t innovations provides the most credible tail-risk estimates and the strongest validation results.

## Main findings
- Return distributions are non-Gaussian, asymmetric, and heavy-tailed.
- Volatility is time-varying, making constant-volatility models inadequate.
- Heavy-tailed and asymmetric innovations materially improve risk measurement.
- The skewed-t CCC-GARCH specification performs best in backtesting.
- Portfolio tail risk is more sensitive to volatility stress than to correlation stress.


# 1.1 Table of content
1. Overview
    1. Table of content
2. Key features
3. Methodology
    1. Risk measures
    2. Model comparison
    3. Parametric and simulation specifications
    4. Model selection logic
    5. Key assumptions and limitations
4. Technologies used
5. How to run the project
6. Results
    1. Distributions
    2. VaR and ES
7. Validation and testing
    1. Distribution diagnostics
    2. Monte Carlo convergence
    3. Backtesting
    4. Benchmarking
    5. Stress testing
        1. Stress multiplier
        2. Tail risk ratio
    6. Model selection summary
8. Model limitations
9. Background and motivation
10. Project structure
11. Future improvements

# 2. Key features
- Multi-method VaR/ES framework: historical, parametric, and Monte Carlo
- Dynamic volatility modeling via CCC-GARCH(1,1)
- Alternative innovation distributions: Gaussian, Student-t, and skewed-t
- Cross-asset dependence modeling through conditional correlation and Cholesky decomposition
- Model validation through normality diagnostics, heteroskedasticity testing, dependence testing, and residual analysis
- VaR backtesting using Kupiec and Christoffersen-style coverage checks
- Monte Carlo convergence analysis for tail quantile stability
- Stress testing of volatility and dependence assumptions
- Modular Python implementation for reproducibility and extension

# 3. Methodology

### 3.1 Risk measures
- VaR is defined as the negative 2.5th percentile of the 1-week profit-and-loss distribution.
- ES is defined as the conditional mean loss beyond the VaR threshold.

### 3.2 Model comparison
The analysis compares three classes of risk models:
- Historical simulation
- Parametric models
- Monte Carlo simulation

### 3.3 Parametric and simulation specifications
The following specifications are evaluated:
- Historical simulation
- Gaussian parametric model
- Exponentially weighted moving average (EWMA)
- Geometric Brownian motion with historical volatility
- CCC-GARCH(1,1) with Gaussian innovations
- CCC-GARCH(1,1) with Student-t innovations
- CCC-GARCH(1,1) with skewed-t innovations

### 3.4 Model selection logic
The models are assessed sequentially using:
- distribution diagnostics,
- heteroskedasticity tests,
- dependence diagnostics,
- convergence analysis,
- and VaR backtesting.
The final model is selected on the basis of empirical adequacy rather than model complexity alone. In this dataset, CCC-GARCH(1,1) with skewed-t innovations delivered the most credible tail-risk estimates and the strongest overall backtesting performance.

### 3.5 Key assumptions and limitations
- Volatility is time-varying and modeled dynamically.
- Cross-asset dependence is captured through a constant conditional correlation (CCC) structure.
- Innovation distributions may be non-Gaussian, heavy-tailed, and asymmetric.
- Results are conditional on the selected sample period and portfolio composition; they should be interpreted as model-based estimates rather than exact forecasts.


# 4. Technologies used
- Python (NumPy, SciPy, Pandas)
- Matplotlib
- Jupyter Notebook

# 5. How to run the project
- Create a python virtual environment
- Run: pip install -r Requirements.txt, to install required libraries
- Run the jupyter notebook: Main.ipynb

# 6. Results
## 6.1 Distributions
![Return distributions](Figures/ReturnDistribution.png)
Returns are not gaussian distributed for all assets. They also show some asymmetry and large tail. This explain why skewed-t distribution, which provide both the tailing of student-t distribution and asymetry, supports the adequacy of the specification.

## 6.2 VaR and ES
![VaR from Historical data](Figures/HistoricalVaR.png)
![VaR from MC simulation](Figures/GARCHVaR.png)
The gap between historical and Monte Carlo estimates suggests that a volatility-sensitive model places more weight on recent market conditions than a long-window historical estimator. This is consistent with the idea that historical simulation can dilute current risk when calm and stressed regimes are pooled together.

# 7. Validation and testing
### 7.1 Distribution diagnostics
![Residual distribution](Figures/ResidualDistribution.png)
![Residual QQ plots](Figures/ResidualQQ.png)
Residual diagnostics indicate that the GARCH-based specifications materially improve the fit relative to simpler models. In particular, allowing for heavy tails and asymmetry produces residual behavior that is more consistent with the empirical return distribution.

### 7.2 Monte Carlo convergence
![Simulation with n=100](Figures/Returns100.png)
![Simulation with n=1000](Figures/Returns1000.png)
![Simulation with n=10000](Figures/Returns10000.png)
![Simulation with n=100000](Figures/Returns100000.png)
Monte Carlo convergence was assessed by comparing simulated return distributions and VaR estimates across increasing numbers of simulation paths (10^2, 10^3, 10^4, and 10^5). As the number of simulations increases, both the shape of the distribution and tail quantiles stabilize, indicating diminishing Monte Carlo error.
Although convergence appears satisfactory with approximately 1,000–10,000 simulations, 100,000 paths are retained in the final analysis to reduce estimation noise in extreme quantiles and improve the stability of ES estimates.

### 7.3 Backtesting
![Cumulative Exception rate](Figures/CumulativeExceptionRate.png)
Backtesting shows that the selected skewed-t GARCH specification provides exception rates and exception dynamics that are consistent with the target confidence level. This supports the use of the model for portfolio-level tail-risk measurement over the sample considered.

## 7.4 Benchmarking
![VaR from Historical data](Figures/HistoricalVaR.png)
![VaR from MC simulation](Figures/GARCHVaR.png)
Again, there is a difference that might be explained by current economic situation.

## 7.5 Stress testing
### 7.5.1 Stress multiplier
| Scenario | Global-CorpBond | MSCI-ACWI | Gold | SXI-RealEstate | EUR |
|----------|-----------------|-----------|------|----------------|-----|
| Model | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Correlation stress | 1.0 | 1.3 | 1.2 | 1.0 | 1.0 |
| Volatility stress | 2.3 | 3.0 | 2.3 | 4.0 | 2.9 |
| Volatility+Correlation stress | 3.5 | 2.9 | 2.4 | 3.0 | 3.1 |

Volatility stress produces a significantly higher VaR/ES multiplier than correlation stress. This indicates that the portfolio's tail risk is primarily driven by volatility shocks rather than by correlation breakdowns. 

### 7.5.2 Tail risk ratio
| Scenario | Global-CorpBond | MSCI-ACWI | Gold | SXI-RealEstate | EUR |
|----------|-----------------|-----------|------|----------------|-----|
| Model | 1.3 | 1.3 | 1.3 | 1.3 | 1.3 |
| Correlation stress | 1.1 | 1.4 | 1.1 | 1.1 | 1.2 |
| Volatility stress | 1.5 | 1.2 | 1.2 | 1.1 | 1.2 |
| Volatility+Correlation stress | 1.2 | 1.3 | 1.1 | 1.3 | 1.2 |

The ES/VaR ratio remains relatively stable across stress scenarios, suggesting that stress primarily scales loss magnitude rather than materially altering tail shape. This is consistent with the linear nature of the portfolio, which does not contain strongly nonlinear payoffs such as options.

## 7.6 Model selection summary
The following table summarizes the risk models considered, their main assumptions, validation results, and the rationale for model acceptance or rejection.

| Model | Key assumptions | Diagnostics & backtesting | Assessment |
|-------|-----------------|---------------------------|------------|
| Historical simulation | Past distribution representative of future | Fails to adapt to current volatility regime | Benchmark only |
| Parametric Gaussian | Normal, i.i.d. returns | Rejected by normality tests | Inadequate |
| EWMA | Short-memory volatility dynamics | Residual non-normality remains | Partial improvement |
| GBM with constant volatility | Constant volatility, Gaussian shocks | Heteroskedasticity detected | Rejected |
| CCC-GARCH (Gaussian) | Time-varying volatility, normal innovations | Volatility modeled well but tails underestimated | Inadequate tail risk |
| CCC-GARCH (Student‑t) | Heavy-tailed innovations | Improved tail fit, mixed backtesting results | Acceptable |
| CCC-GARCH (Skewed‑t) | Heavy-tailed and asymmetric shocks | Best diagnostics and backtesting performance | **Selected model** |

Based on statistical diagnostics, backtesting results, and economic interpretability, CCC-GARCH(1,1) with skewed‑t innovations was selected as the final specification. This model provides the most credible representation of tail risk for the portfolio considered.

# 8. Model limitations
Several limitations should be kept in mind when interpreting the results of this analysis:
- **Constant conditional correlation:** Cross‑asset dependence is modeled using a CCC framework. While supported by diagnostics over the sample period, correlations may become time‑varying during periods of acute market stress.
- **Model‑based tail estimates:** VaR and ES are conditional on the chosen distributional assumptions. Misspecification of innovation distributions or volatility dynamics would directly affect tail‑risk estimates.
- **Sample dependence:** Parameter estimates and validation results depend on the selected historical window, which spans multiple market regimes. Different calibration periods may lead to different risk estimates.
- **Linear portfolio structure:** The portfolio does not include nonlinear instruments such as options. As a result, tail‑risk amplification due to convex payoffs is not captured.
These limitations reflect standard trade‑offs in tractable market risk modeling and highlight areas where more advanced or computationally intensive approaches could be explored.


# 9. Background and Motivation
This project studies the estimation of portfolio tail risk under realistic return dynamics. The focus is on how distributional asymmetry, heavy tails, time-varying volatility, and cross-asset dependence affect 1-week VaR and ES estimates.

Rather than relying on a single methodology, the analysis benchmarks historical simulation, parametric models, and Monte Carlo simulation under progressively richer assumptions. This makes it possible to identify which modeling choices materially improve risk measurement and which assumptions fail under diagnostic testing.


# 10. Project structure
VaR/
|- Data/
|- Figures/
|- Source/
|   |- Distribution.py
|   |- MonteCarlo.py
|   |- Plot.py
|   |- StatTest.py
|   |- Utils.py
|- DataSource.txt
|- Main.ipynb
|- Portfolio.txt
|- README.txt
|- Requirements.txt

# 11. Future improvements
- Use deep learning for volatility prediction
- Nonlinear instruments
- Dashboard