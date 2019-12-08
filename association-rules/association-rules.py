import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

storeData = pd.read_csv('./store_data.csv', header=None)

def display(data, number=7):
  print(data.head(number))

def countNumberProductPurchased(data):
  nbProduct = 0
  for cols in data:
    nbProduct += data[cols].count()
  return nbProduct

def transformIntoList(data=None):
  transformedDataList = []
  for index in range(0, 7501):
    transformedDataList.append([
      str(storeData.values[index, feature])
      for feature in range(0, 20)
    ])
  return transformedDataList

if __name__ == '__main__':
  print('Question 2:', display(storeData))
  print('Number of product purchased: ', countNumberProductPurchased(storeData))
  print('Question 3: Reshaping...')
  transData = transformIntoList(storeData)
  print('Done.\n')

  print('Question 4, 5, 6:')
  assRules = apriori(transData, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
  assResult = list(assRules)
  print('Number of rules: ', len(assResult))
  print('Content: \n', assResult)