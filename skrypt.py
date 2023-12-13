import numpy as np
import pandas as pd

df = pd.read_csv(r"uzupelnienie_glukozy.csv")
print(df.head())
print(df.info())

counter=0
for row in df.Glucose:
    if(row== 0):    
        counter+=1
idxs = df.loc[df['Glucose'] == 0] .index
print(counter)
