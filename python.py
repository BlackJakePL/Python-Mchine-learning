import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import neurolab as neuro
import numpy.random as rand
import matplotlib.pyplot as plt

print("Podaj nazwe lub sciezke do pliku: ")
file = input()
df = pd.read_csv(file+".csv")
print(df.head())
print(df.info())
counter=0
for row in df.Age:
        if(row== 0):
            counter+=1
idxs = df.loc[df['Age'] == 0] .index
print("Brakujacych danych w kolumnie Age: ",counter)