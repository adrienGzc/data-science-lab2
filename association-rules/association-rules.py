import math
import copy
import time
import pprint
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sb
from apyori import apriori as ap
from tabulate import tabulate
from operator import itemgetter
from collections import Counter
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Get all the product purchased in the dataset
# Basically, a sum of all the item in each row of the table
def countNumberProductPurchased(data):
  nbProduct = 0
  for instance in data:
    nbProduct += len(instance)
  return nbProduct

# Get all product purchased as an unique product.
def countUniqueProduct(data):
  listUniqueItems = list()
  for instance in data:
    listUniqueItems = list(set().union(listUniqueItems, instance))
  listUniqueItems.sort()
  return listUniqueItems, len(listUniqueItems)

def getAssociationRules(data):
  return ap(data, min_support=0.006, min_confidence=0.3, min_lift=3.0, min_length=2)

def runAprioryFromMlextend(dataset):
  rules = apriori(dataset, min_support=0.006, use_colnames=True)
  rules = association_rules(rules, metric="lift", min_threshold=3.0)
  return rules

def runFpFromMlextend(dataset):
  rules = fpgrowth(dataset, min_support=0.006, use_colnames=True)
  rules = association_rules(rules, min_threshold=3.0, metric='lift')
  return rules

# Transform the dataset into a list of list.
def transformIntoList(data=None):
  transformedDataList = list()

  for instance in data.values:
    transformedDataList.append([str(value) for value in instance if str(value) != 'nan' ])
  return transformedDataList

# Print a list of list as a table using tabulate as library.
def printRulesAsTable(data=None, mode='simple', header=[]):
  if data is None:
    return False
  print(tabulate(data, headers=header, tablefmt=mode))

# Get rid of the nan value in the table if exist.
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

# Concate the 2 products related for the table view.
def concatProduct(rules):
  transformed = list()

  for rule in rules:
    transformed.append([rule[0] + ' -> ' + rule[1],
      rule[2],
      rule[3],
      rule[4]]
    )
  return transformed

# Get rid of the duplicate rules but reversed like: beef & tomato || tomato & beef.
def deleteDuplicate(rules):
  rulesDict = dict()
  for rule in rules:
    if rule[0] not in rulesDict:
      rulesDict[rule[0]] = list(rule)
  return [value for value in rulesDict.values()]

# Just a wrapper to call all the function to shape the data.
def cleanRules(data):
  tmp = filterNanRules(data)
  tmp = concatProduct(tmp)
  return deleteDuplicate(tmp)

def displayInfoRules(rules):
  print('Number of rules: ', len(rules), '\n')
  pprint.pprint(rules)

# Return 2D array with element delete in X and Y.
# HAS TO BE AN NUMPY ARRAY
def deleteElem2dArray(array2d, index):
  array2d = np.delete(array2d, index, 0)
  return np.delete(array2d, index, 1)

# Delete all elem of the heatmap to be higher than the breakpoint.
def filterLowerOccurence(heatmap, products, breakPoint=250):
  index = 0
  for instance in heatmap:
    # Get the max value and check if she is lower than the breakpoint.
    if max(instance) < breakPoint:
      # Delete case in the heatmap (2D array).
      heatmap = deleteElem2dArray(heatmap, index)
      # Remove also the product, otherwise still present for the label in heatmap.
      products.remove(products[index])
      index -= 1
    index += 1
  return heatmap, products

# Return the heatmap. Count all the product related to each other in a list of list.
def getHeatMap(data, products):
  # Init the heatmap with a list of list of 0, based on the number of unique products.
  heatmap = np.zeros((len(products), len(products)), dtype=int)
  # heatmap = [[0 for xValue in range(len(products))] for yValue in range(len(products))]

  for product in products:
    for instance in data:
      if product in instance:
        for item in instance:
          if product != item:
            yIndex = products.index(product)
            xIndex = products.index(item)
            heatmap[yIndex][xIndex] += 1
  return filterLowerOccurence(heatmap, products)

def showHeatmap(data, labels):
  sb.set()
  plt.figure(figsize=(10,6))
  plt.xlabel("Unique products X")
  plt.ylabel("Unique products Y")
  sb.heatmap(data, cmap='YlGn', xticklabels=labels, yticklabels=labels)
  plt.show()

def compareAlgo(dataset):
  # Transform dataset to a readable dataset accepted for the algorithms.
  te = TransactionEncoder()
  te_ary = te.fit(dataset).transform(dataset)
  df = pd.DataFrame(te_ary, columns=te.columns_)

  # Collect the start and end point of the timer.
  start = time.time()
  runAprioryFromMlextend(df)
  end = time.time()
  print("Time apriori:", end - start)
  
  start = time.time()
  runFpFromMlextend(df)
  end = time.time()
  print("Time fpgrowth:", end - start)

def main():
  storeData = pd.read_csv('./store_data.csv', header=None)

  print('Question 2, 3: ', storeData[:7])
  transData = transformIntoList(storeData)
  nbProdPurchased = countNumberProductPurchased(transData)
  uniqueProducts, nbUniqueProducts = countUniqueProduct(transData)
  print('Number of product purchased: ', nbProdPurchased)
  print('Unique product: ', nbUniqueProducts)

  print('Question 4, 5, 6: ')
  rules = getAssociationRules(transData)
  transformedRules = cleanRules(list(rules))
  displayInfoRules(transformedRules)

  print('\nQuestion 7: ')
  printRulesAsTable(transformedRules, mode='simple', header=['Rules', 'Support', 'Confidence', 'Lift'])

  print('\nQuestion 9: ')
  heatmap, productsLabel = getHeatMap(transData, uniqueProducts)
  showHeatmap(heatmap, productsLabel)

  print('\nQuestion 10: ')
  compareAlgo(transData)

if __name__ == '__main__':
  main()