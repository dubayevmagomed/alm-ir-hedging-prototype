# ALM Interest Rate Hedging Prototype

Quantitative prototype for modelling and hedging interest rate risk in an Asset-Liability Management (ALM) context using EUR interest rate swaps (IRS).

This project implements a full end-to-end framework for curve construction, risk modelling, scenario simulation, and hedge optimization.

---

## Overview

The model simulates and evaluates interest rate risk for a stylized pension fund liabilities, represented by projected cashflows and hedged using EURIBOR 6M interest rate swaps.

The framework supports:
- Curve calibration and bootstrapping
- Risk factor modelling using PCA
- Monte Carlo scenario generation
- Full revaluation of assets and liabilities
- Sensitivity analysis (PV01, gamma)
- Construction and evaluation of hedging strategies

---

## Key Features

### Curve Construction
- Bootstrapping of EUR swap curves (EURIBOR 6M)
- Built on top of `Rateslib`

### Risk Factor Modelling
- Principal Component Analysis (PCA) of swap rate movements
- Dimensionality reduction of the yield curve

### Monte Carlo Simulation
- Scenario generation using historical covariance
- Cholesky decomposition for correlated shocks

### Liability Modelling
- Pension-style liability profile
- Modelled as a portfolio of short zero-coupon bonds

### Pricing
- Interest rate swap (IRS) pricing
- Curve-based discounting and forward rate projection

### Risk Metrics
- Spot PV01 and Forward PV01
- Key rate sensitivities
- Gamma (second-order sensitivity)

### Hedging Framework
- Full immunization hedge (benchmark)
- Variance-minimizing hedge using liquid tenors (2Y, 5Y, 10Y, 30Y)
- Optimization via General Least Squares (GLS)

---