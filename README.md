# Prototaxites-Analysis-Code
This repository stores the code for the submission "Prototaxites fossils are anatomically and chemically distinct from extinct and extant Fungi" to be published in Science Advances.

For any questions about this paper please contact the corresponding authors listed in this work. For questions related to this code please contact Corentin Loron, corentin.loron@ed.ac.uk, or Niall Rodgers, niall.rodgers@ed.ac.uk.


## Analysis Codes - .py files
The repository includes the codes to perform PCA, including outlier detection using Hotelling’s T2 versus Q residual values, DAPC, SVM (both with and without SMOTE for imbalance classes) as well as learning curves. All data analyses were conducted with PyCharm. To run the code you will need standard Python packages: Pandas, Matplotlib, Numpy, Scikit-learn, Seaborn, Imblearn, and Scipy.

For each binary dataset, the excel file is loaded with sample names in first column and one-hot encoded labels in the two last columns. Datasets are available in the supplementary material. All results are discussed in the main text, the supplementary material and the extended methods.


## CCA Code - Notebooks files
The repository includes conducting the CCA of the data and generating the analysis used in the paper. The CCA analysis relies on the well-used R package Vegan. However, for convenience we choose to access the Vegan package through Python using the package rpy2. This is a convenient solution, however requires that R is accessible from within your Python environment, and this will error if this is not the case. All the data analysis was conducted in the attached Jupyter notebook, which we also give as a html. To run this code you will also require standard Python packages like Matplotlib, Numpy and Scikit-learn.

In general, for CCA, the analysis code is relatively short as we mostly call high-level functions derived from established packages. Details of the data used and the description of the procedure are given in the main text and supplementary information. There are a few bespoke lines of code for loading the specific excel file containing the data into the correct format for the packages. 


