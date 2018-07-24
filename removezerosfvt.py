import os
import pandas as pd
import numpy as np

dataset = pd.read_csv('maline_fvt2.csv')

print(len(dataset.columns))
df=dataset.astype(bool).sum(axis=0)
df.to_csv("venice.csv",index=False) 
dataset = dataset.loc[:, dataset.any()]

print(len(dataset.columns))
#dataset.to_csv("helloworld.csv",index=False)  