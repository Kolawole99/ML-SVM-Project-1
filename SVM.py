#================================IMPORTING REQUIRED MODULES, PACKAGES AND LIBRARIES=========================
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %matplotlib inline  #useful in Jupyter notebooks


#============================================LOADING THE DATA================================
"""
    ID	            Clump thickness
    Clump	        Clump thickness
    UnifSize	    Uniformity of cell size
    UnifShape	    Uniformity of cell shape
    MargAdh	        Marginal adhesion
    SingEpiSize	    Single epithelial cell size
    BareNuc	        Bare nuclei
    BlandChrom	    Bland chromatin
    NormNucl	    Normal nucleoli
    Mit	            Mitoses
    Class	        Benign or malignant
"""
cell_df = pd.read_csv("cell_samples.csv")
dataset = cell_df.head()
print(dataset)

# The values are graded from 1 to 10, with 1 being the closest to benign.
# The Class field contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are benign (value = 2) or malignant (value = 4).

#========================distribution based on clump thickness and uniformity of cell size.
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()



#====================================DATA PREPROCESSING AND SELECTION=====================================
column_types = cell_df.dtypes #look at the column types
print(column_types)

#====================================drop non numerical columns=============================
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
column_types = cell_df.dtypes #look at the column types
print(column_types)

#======================================define the feature set===================================
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]
print(X)

#=======================================Target variable binary================================
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y [0:5]
print(y)