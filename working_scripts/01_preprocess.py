#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pathlib import Path

def run(cfg):
    print("01-Preprocess starting...")
    Anno = pd.read_csv('/home/77462217B/lois/ADMeth/data/annotations/Anno320K.csv')
    def filter_and_complete_dataframe(new_df, Anno):
        # Ensure CpG is the index in Anno
        Anno_index = Anno.set_index("CpG")
        # Filter new_df to keep only row names present in Anno["CpG"]
        new_df_filtered = new_df.loc[new_df.index.intersection(Anno_index.index)]
        # Create a dataframe with all CpGs from Anno and default values of -1
        new_df_complete = pd.DataFrame(index=Anno_index.index).join(new_df_filtered, how="left").fillna(0)
        print("Max value: ")    
        print(new_df_complete.max().max())
        print("Min value: ")    
        print(new_df_complete.min().min())
        return new_df_complete.astype(float)