import math
import copy
import pprint
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from apyori import apriori
from tabulate import tabulate
from operator import itemgetter
from collections import Counter
import matplotlib.pyplot as plt

def display(data, number=7):
  print(data.head(number))

def countNumberProductPurchased(data):
  nbProduct = 0
  for cols in data:
    nbProduct += data[cols].count()
  return nbProduct

def countUniqueProduct(data):
  listUniqueItems = list()
  for index, _value in enumerate(data):
    uniqueProduct = data[index].unique()
    listUniqueItems = list(set().union(listUniqueItems, uniqueProduct))
  return listUniqueItems, len(listUniqueItems) - 1

def transformIntoList(data=None):
  transformedDataList = []
  for index in range(0, len(data)):
    transformedDataList.append([
      str(data.values[index, feature])
      for feature in range(0, 20)
    ])
  return transformedDataList

def printRulesAsTable(data=None, mode='simple', header=[]):
  if data is None:
    return False
  print(tabulate(data, headers=header, tablefmt=mode))

def filterNanRules(rules):
  filteredRules = list()

  for rule in rules:
    items = [x for x in rule[0]]
    if items[0] != 'nan' and items[1] != 'nan':
      filteredRules.append([items[0],
        items[1],
        str(rule[1]),
        str(rule[2][0][2]),
        str(rule[2][0][3])]
      )
  return filteredRules

def concatProduct(rules):
  transformed = list()

  for rule in rules:
    transformed.append([rule[0] + ' -> ' + rule[1],
      rule[2],
      rule[3],
      rule[4]]
    )
  return transformed

def deleteDuplicate(rules):
  rulesDict = dict()
  for rule in rules:
    if rule[0] not in rulesDict:
      rulesDict[rule[0]] = list(rule)
  return [value for value in rulesDict.values()]

def cleanRules(data):
  tmp = filterNanRules(data)
  tmp = concatProduct(tmp)
  return deleteDuplicate(tmp)

def displayInfoRules(rules):
  print('Number of rules: ', len(rules), '\n')
  pprint.pprint(rules)

def showHeatMap(data, products):
  heatmap = [[0 for xValue in range(len(products) - 1)] for yValue in range(len(products) - 1)]

  # fig, ax = plt.subplots()
  # im = ax.imshow(data)
  # ax.set_xticks(np.arange(len(labelX)))
  # ax.set_yticks(np.arange(len(labelY)))
  # ax.set_xticklabels(labelX)
  # ax.set_yticklabels(labelY)
  # plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
  
  # for i in range(len(vegetables)):
  #   for j in range(len(farmers)):
  #     text = ax.text(j, i, harvest[i, j], ha='center', va='center', color='w')

  # ax.set_title('Heatmap products')
  # fig.tight_layout()
  # plt.show()

def main():
  storeData = pd.read_csv('./store_data.csv', header=None)
  nbProdPurchased = countNumberProductPurchased(storeData)
  uniqueProducts, nbUniqueProducts = countUniqueProduct(storeData)

  print('Question 2:', display(storeData))
  print('Number of product purchased: ', nbProdPurchased)
  print('Unique product: ', nbUniqueProducts)
  print('Question 3: Reshaping...')
  transData = transformIntoList(storeData)
  print('Done.\n')

  print('Question 4, 5, 6:')
  rules = apriori(transData, min_support=0.006, min_confidence=0.3, min_lift=3, min_length=2)
  transformedRules = cleanRules(list(rules))
  # displayInfoRules(transformedRules)

  print('\nQuestion 7:')
  printRulesAsTable(transformedRules, mode='simple', header=['Rules', 'Support', 'Confidence', 'Lift'])

  print('Question 9: ')
  showHeatMap(storeData, uniqueProducts)
  # flights_long = sns.load_dataset("flights")
  # pprint.pprint(flights_long)
  # flights = flights_long.pivot("month", "year", "passengers")

  # pprint.pprint(flights)
  # _f, ax = plt.subplots(figsize=(9, 6))
  # sns.heatmap(flights, fmt="d", linewidths=.5, ax=ax)
  # plt.show()

if __name__ == '__main__':
  sns.set()
  main()