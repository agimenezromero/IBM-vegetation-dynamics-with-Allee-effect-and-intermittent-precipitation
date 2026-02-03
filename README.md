# Intermittent precipitation and spatial Allee effects drive irregular vegetation patterns in semiarid ecosystems
**A stochastic individual-based model of vegetation dynamics in semiarid ecosystems**

This repository contains the simulation code and example notebooks developed for the study:

> *Intermittent precipitation and spatial Allee effects drive irregular vegetation patterns in semiarid ecosystems* (Submitted)

The model explores how the interplay between **local facilitation** (Allee effect), **competition**, and **environmental stochasticity** influences the spatial organization and persistence of vegetation in semiarid ecosystems. By explicitly representing individual plants and their spatial interactions, the simulations provide mechanistic insight into the processes driving the irregular emergence and disappearance of vegetation clusters.

---

## Overview

The code implements a **spatially explicit, stochastic individual-based model (IBM)** in which:

- Birth and death rates depend on the **local density** of individuals, mediated by **Gaussian competition and facilitation kernels**.  
- **Environmental fluctuations** are represented by stochastic alternation between low and high facilitation regimes, simulating irregular precipitation events.  
- Vegetation clustering is quantified using a **pair-correlation function**, allowing comparison with spatially uniform (mean-field) dynamics.  

---

## Repository Structure

The repository is organized as follows:

- `IBM_Allee_lib.jl`  
  Julia library implementing the core individual-based model, including birth–death dynamics, spatial interactions, environmental switching, and data collection routines.

- `Simulations.ipynb`  
  Jupyter notebook illustrating example simulations of the model. This notebook reproduces key results of the study, including the emergence of irregular clusters, persistence–extinction transitions, and the dependence of spatial structure on precipitation intermittency.

- `Image_processing.ipynb`  
  Jupyter notebook demonstrating how spatial statistics derived from the model can be compared with those extracted from vegetation images. This includes preprocessing of binary vegetation maps and the computation of pair-correlation functions for empirical–model comparison.

---

## Requirements

- Julia (≥ 1.8 recommended)  
- Standard Julia scientific libraries (as specified in `IBM_Allee_lib.jl`)  
- Jupyter Notebook or JupyterLab for running the example notebooks  

---

## Usage

1. Load or include `IBM_Allee_lib.jl` in your Julia environment to access the model functions.  
2. Run `Simulations.ipynb` to explore the behavior of the model under different environmental regimes and parameter values.  
3. Use `Image_processing.ipynb` to compare model-generated patterns with vegetation spatial data.

---

## Scope and Limitations

The model is designed to emphasize **individual-level stochasticity and local interactions** rather than to provide site-specific predictions. It abstracts complex ecohydrological processes into effective facilitation and competition kernels and represents precipitation variability through stochastic switching, rather than explicit rainfall–runoff dynamics.

---

## Citation

If you use this code, please cite:

> *Intermittent precipitation and spatial Allee effects drive irregular vegetation patterns in semiarid ecosystems* (submitted)

---

