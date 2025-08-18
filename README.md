#"Sparse Surrogate Modeling for Combustion Chemistry via Time-Lag Autoencoders and Gradient-Based Clustering". 

<img width="2000" height="1500" alt="graphical_abstract" src="https://github.com/user-attachments/assets/c48faa22-036c-40e0-9975-f1cf3850c57c" />

** Schematic of the proposed methodology for surrogate modeling of chemical kinetics using Time-Lag Autoencoders' dimensionality reduction, and Gradient-Based 
Clustering**. This methodology provides accurate surrogate models, even in conditions of sparse data and reduced input size; a test case is shown for the GRI-Mech 2.11 mechanism, allowing for a reduction of 42% in input size, while maintaining accuracy when evaluating in interpolation and extrapolation conditions. 
#Abstract
----------------------------------------------

This work presents a data-driven framework for efficient and interpretable surrogate modeling of combustion chemistry, combining time-lag autoencoders (TAEs) with gradient-based clustering. The methodology targets applications where high-fidelity simulations are computationally prohibitive, enabling accurate prediction of thermochemical states with minimal input variables. Ignition trajectories are first encoded into low-dimensional, temporally informed latent spaces, from which physically meaningful chemical carriers are identified via statistical correlation. Cases are then grouped by progress-variable gradient similarity, and cluster-specific regression models reconstruct the full thermochemical state from the carriers and equivalence ratio. The proposed framework is demonstrated on methane/air combustion with GRI-Mech 2.11 over a broad range of equivalence ratios. The approach achieves a coefficient of determination greater than 0.9 in most cases and maintains robustness in interpolation and sparse-data scenarios. This work shows that combining dynamical feature learning with gradient-informed clustering yields generalizable, interpretable, and computationally efficient surrogates for complex reaction systems, with potential for real-time energy system modeling.                                                                      
                                              
----------------------------------------------




This repository contains the codes related to the publication "Sparse Surrogate Modeling for Combustion Chemistry via Time-Lag
Autoencoders and Gradient-Based Clustering". The codes are organized as follows:

-Clustering: contains a jupyter notebook with the kmeans algorithm application, it has also the indexes calculation for each 
 clustering strategy. The clustering pictures available in the paper are generated with this notebook. 

-cross_validations: here you will find a set of different folders, one for each ignition trajectory under analysis. Each of 
  these folders has a .py file, with the code used for the networks tunning with cross validations using the k-fold algorithm, 
  and with different quality criteria (loss, accuracy, validation loss, validation accuracy). 

-Datasets: this folder contains the .csv files resulting from the Cantera simulations; more specifications can be found in the 
file 'Datasets_description.txt', inside this folder. 

-Manifold_extractions: again, here a folder is available for each of the ignition trajectories under study; in each folder it 
 is possible to find the chosen TAE model, and some jupyter notebooks that evaluate the correlation indexes of each case and 
 associate the latencies to the chemical carriers. 

-Permutation_importante_test: this folder holds some python notebooks, which produce the permutation features importance results
 displayed in the paper. 

-Regressions: this folder contains some regression cases using all the identified chemical carriers. Further folders are found
 inside, as well as further explanation files. 

-Regression_selected_carriers: here the regression files related to the regression using each cluster carriers are avialable. 
 Further folders and explanation files are available inside. 
