# FloodFlow

FloodFlow is a numerical-core hydrological modelling suite designed to model coastal, fluvial and pluvial scenarios across large scale rural and urban terrain.

It will eventually incorporate 1D, 2D and 3D models using, respectively, the one-dimensional Saint-Venant equations, the two-dimensional Shallow Water Equations and the three-dimensional Navier Stokes Equations on Cartesian grids.

# Numerics

FloodFlow is initially designed around the solution of the two-dimensional Shallow Water Equations. It makes use of robust high-resolution shock-capturing schemes such as the MUSCL-Hancock scheme with HLLC approximate Riemann solver.

This ensures minimised numerical diffusion that would otherwise be present in a first-order scheme, such as the approach of Godunov (1959).

# Development Roadmap

FloodFlow is in its infancy at this stage. This public repository has been set up to allow other researchers to follow development progress and suggest enhancements.

The numerics will initially be prototyped in Python 3 to aid development speed and also to minimise bugs that are easily introduced when working with a compiled language, such as C++.

Once all tests have passed for the Python 3 version a C++ core will be developed in order to allow rapid execution speed on CPUs.

Eventually the goal is to utilise MPI and Nvidia CUDA to create multi-CPU and GPU versions of the code for modern parallelisation.

## Development of the 2D SWE solvers will proceed as follows:

* Create and test the Riemann solver class hierarchy, which will include the exact Riemann solver for the 2D SWE, along with the HLLC solver of Toro et al.

* Create and test the numerical Scheme class hierarchy, which will include a 1st-order Godunov scheme (utilising the Riemann solvers above) and a 2nd-order MUSCL-Hancock TVD scheme. Further schemes, such as the Kurganov & Petrova (2007) scheme may also be added in the future.

* Create and test dataset, domain and model configuration tools - along with spatio-temporal boundary condition handling.

* Develop benchmarking and logging utilities to aid in the debugging of the software.

* Port the entire application to C++ and ensure test results are identical.

* Utilise MPI to allow domain decomposition for larger simulation models.

* Create an Nvidia CUDA version of the software for modern GPU support.

## Development of 1D SVE and 3D NSE

* The one-dimensional and three-dimensional aspects of the software will not be considered until the two-dimensional aspect is mature.

* Potential solvers for the three-dimensional Navier-Stokes equations include Smoothed Particle Hydrodynamics, for use cases with challenging boundaries (e.g. urban centres)

# Test Cases

FloodFlow will be validated at all levels:

* Specific analytical test case comparison for solutions of the Riemann problem

* Various 1D and 2D analytical tests of the schemes from a variety of research papers

* Comparison with experimental test cases developed by researchers

* Validation against the UK Environment Agency test cases

* Comparison against real flood incidents, where DEMs and flood extent are easily availble

These tests will ultimately ensure FloodFlow is fit for use in challenging "industrial" applications.