#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import glob
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

def run(cfg):
    normalization_controls = 
    dataset_cases = np.load("/home/77462217B/lois/ADMeth/outcomes/ValidateDataset1K_float16/ValidateDataset1K_float16_mse_per_sample_per_position.npy")
    dataset_controls = np.load("/home/77462217B/lois/ADMeth/outcomes/Michaud_float16/Michaud_float16_mse_per_sample_per_position.npy")
    
    def BackgroundNormalization(ErrorsMatrix):
        BCErrorsMatrix={}
        for i in range(ErrorsMatrix.shape[0]):
            #BCErrorsMatrix[i] = ErrorsMatrix.loc[i,:]/(ErrorsMatrix.loc[i,:].mean())
            BCErrorsMatrix[i] = ErrorsMatrix.iloc[i,:]/(ErrorsMatrix.iloc[i,:].median())
        BCErrorsMatrix = pd.DataFrame(BCErrorsMatrix).T
        return BCErrorsMatrix
        
    BCTrainConMSE =  BackgroundNormalization(TrainConMSE)  
    EVMicMSE = BackgroundNormalization(EVMicMSE)
    
    def ScoreRegion(BCErrorsMatrix, BCErrorsMatrixControls):
        ScoreMatrix=((BCErrorsMatrix-BCErrorsMatrixControls.mean(axis=0))/(BCErrorsMatrixControls.std(axis=0)))
        return ScoreMatrix
    
    EVMicConMSE = ScoreRegion(EVMicConMSE, BCTrainConMSE)
    EVMicCasMSE = ScoreRegion(EVMicCasMSE, BCTrainConMSE)
    
    EVMicMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/EV/EVMicMSE.csv', index=False)
    EVMicConMSE.to_csv('/home/77462217B/lois/ADMeth/outcomes/MRS/EV/EVMicConMSE.csv', index=False)
