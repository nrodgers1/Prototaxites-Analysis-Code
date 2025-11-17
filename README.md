# Prototaxites-Analysis-Code
This repository stores the code for the submission "Prototaxites fossils are anatomically and chemically distinct from extinct and extant Fungi" to be published in Science Advances.

For any questions about this paper please contact the corresponding authors listed in this work. For questions related to this code please contact Corentin Loron, corentin.loron@ed.ac.uk, or Niall Rodgers, niall.rodgers@ed.ac.uk.




## CCA Code - Notebooks files
The repository includes conducting the CCA of the data and generating the analysis used in the paper. The CCA analysis relies on the well used R package Vegan. However, for convience we choose to access the Vegan package through Python using the package rpy2. This is a convient solution, however requires that R is accesible from within you Python environment, and this will error if this is not the case. All the data analysis was conducted in the attached jupyter notebook, which we also give as a html for convience. To run this code you will also require standard Python packages like Matplotlib, Numpy and Scikit-learn. 

In general, CCA the analysis code is relatively short as we mostly call high-level functions derived from established packages. Details of the data used and the descritpion of the procedure are given in the main text and supplmenatry information. There are a few bespoke line of code for loading the specfic excel file containing the data into the correct format for the packages.  


