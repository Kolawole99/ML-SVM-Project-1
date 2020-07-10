#================================IMPORTING REQUIRED MODULES, PACKAGES AND LIBRARIES=========================
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
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

#======================================look at the column types================================
column_types = cell_df.dtypes 
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



#==========================================TRAIN/TEST SPLIT=====================================
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



#=====================================MODELLING SVM WITH SCIKIT-LEARN===================================

#=========================================training the model=================================
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

#=======================================Predicting with model================================
yhat = clf.predict(X_test)
yhat [0:5]



#==========================================EVALUATING THE MODEL=========================================
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#======================================Compute confusion matrix=====================================
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)
print (classification_report(y_test, yhat))

#===============================Plot non-normalized confusion matrix==================================
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')