TPC 3
Crie um ficheiro em python para trabalhar o dataset

    datasets.california_housing
Nesse ficheiro, crie um script (função) por alínea que lhe permita gerar novos datasets a partir do dataset principal, onde tenha usado cada um dos seguintes métodos de pre-processamento:

1) Aggregation
2) Sampling
4) Dimensionality Reduction 
5) Feature Subset Selection 
6) Feature Creation 
7) Discretization and Binarization 
8) Attribute Transformation
O que é feito em cada caso, é da sua inteira liberdade.

[3]

import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
"done"
​
data = datasets.california_housing.fetch_california_housing()
​
#show disorganized data
data
[5]

HouseAge
​
​
# 1- Agregação do máximo, mínimo, média da Feature Poputação
​
dframe = pd.DataFrame(data = data.data, columns=data.feature_names)
dframe.head()
dframe.aggregate(['max', 'min', 'mean'])['HouseAge']
​
​
[7]

Population
def sample_stat(sample):
  print(sample)
  return sample.mean(['HouseAge'])
[8]

dframe.loc[0:1,["MedInc"]]
[10]

HouseAge
dframe['FamilySize'] = dframe['HouseAge'] + dframe['Population']
dframe.head()
[16]

HouseAge
from sklearn import preprocessing
​
binarizer = preprocessing.Binarizer().fit('HouseAge')
print(binarizer)
binarizer.threshold=3.50
[18]

def draw_missing_data_table(dframe):
    total = dframe.isnull().sum().sort_values(ascending=False)
    percent = (dframe.isnull().sum()/dframe.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
