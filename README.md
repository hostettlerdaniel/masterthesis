# Master Thesis Code (Simulation Study)

This repository contains the **Python simulation code** used for the master thesis experiments on **proper scoring rules under right censoring** with Weibull models.

The code simulates right-censored survival data, fits Weibull models under different scoring approaches and compares:

- **Hazard-based log-score**
- **IPCW log-score (oracle)**
- **plug-in IPCW log-score (plug-in Kaplan–Meier)**

It produces:

- simulation summaries (`results_csv/`)
- plots in PDF and PNG (`plots_pdf/`, `plots_png/`)

---

## Requirements

The required packages are listed in the file 'requirements.txt'.

---

## Repository structure

```text
Code/
    Simulation.py          # main simulation script (runs all experiments)
    helperfunctions.py     # Weibull helpers, KM/IPCW helpers, fitting helpers
    plots_pdf/             # generated figures (PDF)
    plots_png/             # generated figures (PNG)
    results_csv/           # generated CSV summaries
```

---

## Main files

### `helperfunctions.py`
Contains utility functions

### `Simulation.py`
Contains the main simulation, automatic figure generation

---
