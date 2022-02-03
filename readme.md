# [Catchment scale runoff time-series generation and validation using statistical models for the Continental United States](https://www.sciencedirect.com/science/article/pii/S1364815222000275)
## Authors: Patton, Douglas; Smith, Deron; Muche, Muluken; Wolfe, Kurt; Parmar, Rajbir; Johnston, John M. 
### Affiliations: Oakridge Institute of Science and Education; EPA/ORD/CEMM/EPD/LASMB; National Science Foundation 

- the python file [cn_correct.py](cn_correct.py) contains the accuracy assessment code
  - [run_multi_correct.py](https://github.com/quanted/hms-handler/blob/paper-nldas/run_multi_correct.py) contains the options and code to implement the NLDAS results in the paper.
  - [run_multi_correct.py](https://github.com/quanted/hms-handler/blob/paper-gldas/run_multi_correct.py) contains the options and code to implement the GLDAS results in the paper's appendix.
  
The exact environment used to create the results in the paper are in exact_environment.txt

A similar environment compatible with your system can be created using anaconda/miniconda with the following command:

`conda create -c conda-forge --name hms python=3.8 statsmodels descartes geopandas jupyterlab scikit-learn=0.24.0 matplotlib mapclassify`

There are a variety of jupyter notebook files in here that are not part of the results, but which were used to help with code and idea development.