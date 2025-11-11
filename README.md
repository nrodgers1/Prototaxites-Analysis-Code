# Prototaxites-Analysis-Code
This repository stores the code for the submission "Prototaxites fossils are anatomically and chemically distinct from extinct and extant Fungi" to be published in Science Advances.


The repository includes conducting the CCA of the data and generating the analysis used in the paper. The CCA analysis relies on the well used R package Vegan however for conveience we choose to access through the Vegan package through Python using the package rpy2. This is a convient solution however requires that R is accesible from within you Python environment and this will error if this is not the case. 

In general the rest of the analysis code is relatively short as we mostly call high-level functions derived from established packages. Details of the data used and the descritpion of the procdure are given in the main text and supplmenatry information. There are a few bespoke line of code for loading the specfic excel file containing the data into the correct format for the packages.  
