# LEAP_ClimSim_Competition

Kaggle competition on the simulation of higher resolution processes within E3SM-MMF model.
The goal is to develop machine learning models to accurately emulate subgrid-scale atmospheric physics in an operational climate model.

## Preprocessing

This folder contains preprocessing steps to format the input data and compute statistics.

* The “statistics” notebook computes the mean, standard deviation, minimum, and maximum of the training dataset. It calculates these statistics for each variable at each vertical level, as well as for the entire atmospheric column (i.e., across all vertical levels) over the spatio-temporal domain. For execution, 3 copies of this notebook where done to process different chunks of data, run in parallel and save the outputs. **statistics_2** is an example of these copies. The variable required to be changed between notebook copies is ```SLICE_START = 2_522_880*copy```, where the variable ```copy``` inidicates which notebook copy is that one (for example, copy=1 for the first copy of the notebook and copy=0 for the original notebook).