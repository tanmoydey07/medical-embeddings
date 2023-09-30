# Databricks notebook source
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


def read_data():
    df=pd.read_csv('https://medicalembeddings.blob.core.windows.net/testcontainer/data/input/dimension-covid.csv?sp=r&st=2023-09-30T13:19:36Z&se=2023-11-10T21:19:36Z&spr=https&sv=2022-11-02&sr=b&sig=8TsiBBDT%2Bm11flfg6R3L%2F%2BsrhewIwTTjwm1B%2BX2JMdo%3D')   #for preprocessing
    df1=pd.read_csv('https://medicalembeddings.blob.core.windows.net/testcontainer/data/input/dimension-covid.csv?sp=r&st=2023-09-30T13:19:36Z&se=2023-11-10T21:19:36Z&spr=https&sv=2022-11-02&sr=b&sig=8TsiBBDT%2Bm11flfg6R3L%2F%2BsrhewIwTTjwm1B%2BX2JMdo%3D')  #for returning results
    return df.iloc[:100,:]