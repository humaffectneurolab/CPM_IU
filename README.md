# CPM_IU
Whole-brain Networks Dynamics Underlying Intolerance of Uncertainty

Contains code for analyses in the manuscript "Whole-brain Networks Dynamics Underlying Intolerance of Uncertainty". Code mainly involves Connectome-based Predictive Modeling (CPM) and Edge Timeseries Analyses, adapted from Shen et al. (2017) and Ye et al. (2025) to fit the research questions of this work.

cpm_kfold_robustEdges_eTS.m
: Carries out CPM, extracts the most robust edges, and applies edge timeseries to extract network timeseries.

eTS_dynamics.m
: Calculates five dynamics metrics from the network timeseries data.

permutation.m
: Permutation test to non-parametrically test the statistical significance of the CPM result.

cpm_stability.m
: Applies repeated k-fold CPM to confirm the stability of the CPM result.



Shen, X., Finn, E. S., Scheinost, D., Rosenberg, M. D., Chun, M. M., Papademetris, X., & Constable, R. T. (2017). Using connectome-based predictive modeling to predict individual behavior from brain connectivity. Nature Protocols, 12(3), 506-518.

Ye, J., Garrison, K. A., Lacadie, C., Potenza, M. N., Sinha, R., Goldfarb, E. V., & Scheinost, D. (2025). Network state dynamics underpin basal craving in a transdiagnostic population. Molecular Psychiatry, 30(2), 619-628.

