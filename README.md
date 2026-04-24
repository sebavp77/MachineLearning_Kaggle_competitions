# LEAP_ClimSim_Competition
Kaggle competition on the simulation of higher resolution processes within E3SM-MMF model.
The goal is to develope machine learning models to accurately emulate subgrid-scale atmospheric physics in an operational climate model.

## Data structure
The training dataset provided by the competition is described on the [LEAP-ClimSIM](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim) competition page. In short, the data consists of different atmospheric variables. Some of these variables have vertical levels, which are expanded into separate columns.

For example, variable ```state_t``` (air temperature) has 60 vertical levels. In the dataset each level is a new column and named as state\_t **\_i** where **i** is the vertical level going from 0 to 59. The vertical levels are ordered in descending altitude, meaning that the first level (index 0) corresponds to the highest altitude in the atmosphere, while the last level (index 59) corresponds to the surface. The size of the training dataset is of ~ **360** GB.

Additionally, each row of the dataset corresponds to a location (latitude and longitude) for a specific time. Every time step has **384** locations.
The training dataset is composed of features (x) and targets (y). In total there are 556 different feature variables consisting of
* **540** columns from variables with vertical levels (**9 variables** x **60 levels**)
* **16** variables without vertical level

The targets consist of **368** columns, divided into:
* **360** columns from variables with vertical levels (**6 variables** x **60 levels**)
* **8** variables without vertical levels
  
For a detailed description of the variables names and units the reader is refered to the documentation of the official competition.

An example of the training dataset is:
|state_t_0|state_t_1|...|state\_ps|...|
|---------|---------|---|---------|---|
| value0(t=t1)   | value0(t=t1)   | value0(t=t1) | value0(t=t1) | value0(t=t1) |
| value1(t=t1)   | value1(t=t1)   | value1(t=t1) | value1(t=t1) | value1(t=t1) |
| ...   | ...   | ... | .... | ... |
| value383(t=t1)   | value383(t=t1)   | value383(t=t1) | value383(t=t1) | value383(t=t1) |
| value0(t=t2)   | value0(t=t2)   | value0(t=t2) | value0(t=t2) | value0(t=t2) |
| value1(t=t2)   | value1(t=t2)   | value1(t=t2) | value1(t=t2) | value1(t=t2) |
| ...   | ...   | ... | .... | ... |
| value383(t=t2)   | value383(t=t2)   | value383(t=t2) | value383(t=t2) | value383(t=t2) |
| ...   | ...   | ... | .... | ... |

_Table 1. Example of the training dataset structure. Variables with vertical levels are expanded into multiple columns using the base variable name with the suffix **_i** (i from 0 to 59). Each row represents a different location at a specific time, and every 384 rows correspond to a new time step._


## Preprocessing
This folder contains preprocessing steps to format the input data and compute statistics.
* The “statistics” notebook computes the mean, standard deviation, minimum, and maximum of the training dataset. It calculates these statistics for each variable at each vertical level, as well as for the entire atmospheric column (i.e., across all vertical levels) over the spatio-temporal domain. For execution, 3 copies of this notebook where done to process different chunks of data, run in parallel and save the outputs. **statistics_2** is an example of these copies. The variable required to be changed between notebook copies is ```SLICE_START = 2_522_880*copy```, where the variable ```copy``` inidicates which notebook copy is that one (for example, copy=1 for the first copy of the notebook and copy=0 for the original notebook).
