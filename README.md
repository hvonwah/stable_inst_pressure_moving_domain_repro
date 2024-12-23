# Reproduction Code

This repository contains the scripts to reproduce the numerical examples presented in the paper *Stability of instantaneous pressures in an Eulerian finite element method for moving boundary flow problems* by M. Olshanskii and H. von Wahl.

To run the code an appropriate version of NGSolve and ngsxfem are required. Parameters may be passed to the python scripts implementing the individual examples.

### Installation
To run the python scripts locally, a compatible combination of `Netgen/NGSolve` and `ngsxfem` are required. These can be installed by building from sources or the provided pip wheels. For detailed installation instructions, we refer to the installation guidelines of [NGSolve](https://docu.ngsolve.org/latest/install/install_sources.html), and [ngsxfem](https://github.com/ngsxfem/ngsxfem/blob/release/INSTALLATION.md). Our numerical results are realized using the following versions:

| Package | git commit
|-|-|
| NGSolve | `825e6e0e301c3cae15141f4b9687255b730c9dc3`
| ngsxfem | `2330fa9109c3c9a2b2e52c4c8f65c455608f53e1`


### Content

This repository contains the following files:

| Filename | Description | 
|-|-|
| [`README.md`](README.md) | This file. |
| [`CutFEMMovingDomain_convergence.py`](CutFEMMovingDomain_convergence.py) | Example 1: Convergence study over a given range of mesh and time-step refinements. |
| [`CutFEMMovingDomainTH.py`](CutFEMMovingDomainTH.py) | Implementation for convergence study using unfitted (isoparametric) Taylor-Hood finite elements. |
| [`CutFEMMovingDomainBDF2_TH_Example1.py`](CutFEMMovingDomainBDF2_TH_Example1.py) | Example 2: Oscillating cylinder in a cross flow. |
| [`CutFEMMovingDomainBDF2_TH_Example2.py`](CutFEMMovingDomainBDF2_TH_Example2.py) | Example 3: Oscillating square in a cross flow. |
| [`postprocess.py`](postprocess.py) | Measure the spurious pressure oscillations from examples 2 and 3 |
| [`newton.py`](newton.py) | Implementation of a quasi Newton scheme. |
| [`barycentric.py`](barycentric.py) | Construct Alfeld split meshes. |
| [`other_methods/CutFEMMovingDomainSV.py`](other_methods/CutFEMMovingDomainSV.py) | Implementation for convergence study using unfitted Scott-Vogelius finite elements. |
| [`other_methods/CutFEMMovingDomainBDF2_EO_Example1.py`](other_methods/CutFEMMovingDomainBDF2_EO_Example1.py) | Example 2: Oscillating cylinder in a cross flow using equal order elements with CIP stabilization. |
| [`other_methods/CutFEMMovingDomainBDF2_EO_Example2.py`](other_methods/CutFEMMovingDomainBDF2_EO_Example2.py) | Example 3: Oscillating square in a cross flow using unfitted equal order elements with CIP stabilization. |
| [`other_methods/CutFEMMovingDomainBDF2_SV_Example1.py`](other_methods/CutFEMMovingDomainBDF2_SV_Example1.py) | Example 2: Oscillating cylinder in a cross flow using unfitted Scott-Vogelius elements. |
| [`other_methods/CutFEMMovingDomainBDF2_SV_Example2.py`](other_methods/CutFEMMovingDomainBDF2_SV_Example2.py) | Example 3: Oscillating square in a cross flow using unfitted Scott-Vogelius elements. |
