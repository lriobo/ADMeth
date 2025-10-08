#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import glob
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


# In[2]:


TrainConMSE = np.load("/home/77462217B/lois/ADMeth/outcomes/ValidateDataset1K_float16/ValidateDataset1K_float16_mse_per_sample_per_position.npy")
MicMSE = np.load("/home/77462217B/lois/ADMeth/outcomes/Michaud_float16/Michaud_float16_mse_per_sample_per_position.npy")
pdMic=pd.read_csv("/mnt/hydra/ubs/shared/users/Lois/DatasetsControles/OnlyMichaud/pdMichaud.txt", sep=",")


# In[3]:


def group_columns_by_mean(data, group_size: int = 10, missing_value: float = -1):

    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("El input debe ser un pandas.DataFrame o un numpy.ndarray.")

    n_cols = df.shape[1]
    grouped_cols = []

    for start in range(0, n_cols, group_size):
        end = min(start + group_size, n_cols)
        subset = df.iloc[:, start:end]
        group_mean = subset.mask(subset.eq(missing_value), np.nan).mean(axis=1, skipna=True)
        group_mean = group_mean.fillna(missing_value)
        grouped_cols.append(group_mean)

    grouped_matrix = pd.concat(grouped_cols, axis=1)
    return grouped_matrix


# In[4]:


MicMSE = group_columns_by_mean(MicMSE, group_size=10)
TrainConMSE = group_columns_by_mean(TrainConMSE, group_size=10)


# In[6]:


keep = (pdMic["Cohort"]=="PHS")
keep = keep.to_numpy()
EVMicMSE = MicMSE.iloc[keep,:]
EVpdMic = pdMic.iloc[keep,:]

keep = (pdMic["Cohort"]!="PHS")
keep = keep.to_numpy()
CVMicMSE = MicMSE.iloc[keep,:]
CVpdMic = pdMic.iloc[keep,:]

keep = (CVpdMic["Status"]==0)
keep = keep.to_numpy()
CVMicConMSE = CVMicMSE.iloc[keep,:]
CVpdMicCon = CVpdMic.iloc[keep,:]

keep = (CVpdMic["Status"]==1)
keep = keep.to_numpy()
CVMicCasMSE = CVMicMSE.iloc[keep,:]
CVpdMicCas = CVpdMic.iloc[keep,:]

keep = (EVpdMic["Status"]==0)
keep = keep.to_numpy()
EVMicConMSE = EVMicMSE.iloc[keep,:]
EVpdMicCon = EVpdMic.iloc[keep,:]

keep = (EVpdMic["Status"]==1)
keep = keep.to_numpy()
EVMicCasMSE = EVMicMSE.iloc[keep,:]
EVpdMicCas = EVpdMic.iloc[keep,:]

keep = (CVpdMicCas["YearsToDiagnosis"]<=5)
keep = keep.to_numpy()
CVMic5CasMSE = CVMicCasMSE.iloc[keep,:]

keep = (CVpdMicCas["YearsToDiagnosis"]<=3) 
keep = keep.to_numpy()
CVMic3CasMSE = CVMicCasMSE.iloc[keep,:]

keep = (CVpdMicCas["YearsToDiagnosis"]>10)
keep = keep.to_numpy()
CVMicM10CasMSE = CVMicCasMSE.iloc[keep,:]

keep = (CVpdMicCas["YearsToDiagnosis"]<=10) & (CVpdMicCas["YearsToDiagnosis"]>5)
keep = keep.to_numpy()
CVMic10CasMSE = CVMicCasMSE.iloc[keep,:]

keep = (EVpdMicCas["YearsToDiagnosis"]<=5)
keep = keep.to_numpy()
EVMic5CasMSE = EVMicCasMSE.iloc[keep,:]

keep = (EVpdMicCas["YearsToDiagnosis"]<=3) 
keep = keep.to_numpy()
EVMic3CasMSE = EVMicCasMSE.iloc[keep,:]

keep = (EVpdMicCas["YearsToDiagnosis"]>10)
keep = keep.to_numpy()
EVMicM10CasMSE = EVMicCasMSE.iloc[keep,:]

keep = (EVpdMicCas["YearsToDiagnosis"]<=10) & (EVpdMicCas["YearsToDiagnosis"]>5)
keep = keep.to_numpy()
EVMic10CasMSE = EVMicCasMSE.iloc[keep,:]


# MSE Analysis

# In[8]:


def BackgroundNormalization(ErrorsMatrix):
    BCErrorsMatrix={}
    for i in range(ErrorsMatrix.shape[0]):
        #BCErrorsMatrix[i] = ErrorsMatrix.loc[i,:]/(ErrorsMatrix.loc[i,:].mean())
        BCErrorsMatrix[i] = ErrorsMatrix.iloc[i,:]/(ErrorsMatrix.iloc[i,:].median())
    BCErrorsMatrix = pd.DataFrame(BCErrorsMatrix).T
    return BCErrorsMatrix


# In[9]:


BCTrainConMSE =  BackgroundNormalization(TrainConMSE)

EVMicMSE = BackgroundNormalization(EVMicMSE)
EVMicConMSE = BackgroundNormalization(EVMicConMSE)
EVMicCasMSE = BackgroundNormalization(EVMicCasMSE)
EVMic5CasMSE = BackgroundNormalization(EVMic5CasMSE)
EVMic3CasMSE = BackgroundNormalization(EVMic3CasMSE)
EVMicM10CasMSE = BackgroundNormalization(EVMicM10CasMSE)
EVMic10CasMSE = BackgroundNormalization(EVMic10CasMSE)

CVMicMSE = BackgroundNormalization(CVMicMSE)
CVMicConMSE = BackgroundNormalization(CVMicConMSE)
CVMicCasMSE = BackgroundNormalization(CVMicCasMSE)
CVMic5CasMSE = BackgroundNormalization(CVMic5CasMSE)
CVMic3CasMSE = BackgroundNormalization(CVMic3CasMSE)
CVMicM10CasMSE = BackgroundNormalization(CVMicM10CasMSE)
CVMic10CasMSE = BackgroundNormalization(CVMic10CasMSE)


# In[10]:


def ScoreRegion(BCErrorsMatrix, BCErrorsMatrixControls):
    ScoreMatrix=((BCErrorsMatrix-BCErrorsMatrixControls.mean(axis=0))/(BCErrorsMatrixControls.std(axis=0)))
    return ScoreMatrix


# In[13]:


EVMicMSE = ScoreRegion(EVMicMSE, BCTrainConMSE)
EVMicConMSE = ScoreRegion(EVMicConMSE, BCTrainConMSE)
EVMicCasMSE = ScoreRegion(EVMicCasMSE, BCTrainConMSE)
EVMic5CasMSE = ScoreRegion(EVMic5CasMSE, BCTrainConMSE)
EVMic3CasMSE = ScoreRegion(EVMic3CasMSE, BCTrainConMSE)
EVMicM10CasMSE = ScoreRegion(EVMicM10CasMSE, BCTrainConMSE)
EVMic10CasMSE = ScoreRegion(EVMic10CasMSE, BCTrainConMSE)

CVMicMSE = ScoreRegion(CVMicMSE, BCTrainConMSE)
CVMicConMSE = ScoreRegion(CVMicConMSE, BCTrainConMSE)
CVMicCasMSE = ScoreRegion(CVMicCasMSE, BCTrainConMSE)
CVMic5CasMSE = ScoreRegion(CVMic5CasMSE, BCTrainConMSE)
CVMic3CasMSE = ScoreRegion(CVMic3CasMSE, BCTrainConMSE)
CVMicM10CasMSE = ScoreRegion(CVMicM10CasMSE, BCTrainConMSE)
CVMic10CasMSE = ScoreRegion(CVMic10CasMSE, BCTrainConMSE)


# In[12]:


EVMicMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/EV/EVMicMSE.csv', index=False)
EVMicConMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/EV/EVMicConMSE.csv', index=False)
EVMicCasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/EV/EVMicCasMSE.csv', index=False)
EVMic5CasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/EV/EVMic5CasMSE.csv', index=False)
EVMic3CasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/EV/EVMic3CasMSE.csv', index=False)
EVMicM10CasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/EV/EVMicM10CasMSE.csv', index=False)
EVMic10CasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/EV/EVMic10CasMSE.csv', index=False)

CVMicMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/CV/CVMicMSE.csv', index=False)
CVMicConMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/CV/CVMicConMSE.csv', index=False)
CVMicCasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/CV/CVMicCasMSE.csv', index=False)
CVMic5CasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/CV/CVMic5CasMSE.csv', index=False)
CVMic3CasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/CV/CVMic3CasMSE.csv', index=False)
CVMicM10CasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/CV/CVMicM10CasMSE.csv', index=False)
CVMic10CasMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/CV/CVMic10CasMSE.csv', index=False)


# In[ ]:


def DiffControls(BCErrorsMatrix, BCErrorsMatrixControls):
    DiffErrorsMatrix=((BCErrorsMatrix.mean(axis=0)-BCErrorsMatrixControls.mean(axis=0))/BCErrorsMatrixControls.mean(axis=0))
    return DiffErrorsMatrix


# In[55]:


DiffMicConMSE = DiffControls(BCMicConMSE, BCMicConMSE)
DiffMicCasMSE = DiffControls(BCMicCasMSE, BCMicConMSE)
DiffMic5CasMSE = DiffControls(BCMic5CasMSE, BCMicConMSE)
DiffMic3CasMSE = DiffControls(BCMic3CasMSE, BCMicConMSE)
DiffMic10CasMSE = DiffControls(BCMic10CasMSE, BCMicConMSE)
DiffMicM10CasMSE = DiffControls(BCMicM10CasMSE, BCMicConMSE)

