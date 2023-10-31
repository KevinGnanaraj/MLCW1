import numpy as np
import matplotlib as plt
import sys
import time

np.set_printoptions(threshold=sys.maxsize)

# Decision tree rough node structure --> {"value":None, "attribute":None, "l_branch":None, "r_branch":None}

# Example parent node --> {"value":-55.4, "attribute":2, "l_branch":{-- Child Node --}, "r_branch":{-- Child Node --}}
# Example leaf node   --> {"value":None, "attribute":3, "l_branch":None, "r_branch":None}

class tree_node:
  def __init__(self, value, attr, l_branch, r_branch):
    self.value = value
    self.attr = attr
    self.l_branch = l_branch
    self.r_branch = r_branch


class decision_tree:
  def __init__(self):
    self.data = np.loadtxt("wifi_db/clean_dataset.txt")    # This loads the file into a 2000x8 array
    # print(self.data[:10])
    rng = np.random.default_rng()
    rng.shuffle(self.data)
    # print(self.data[:10])
  
  def clean_classification(self, dataset):   # If the dataset contains only samples from the same room return the room numbers, else return -1
    first_room = dataset[0][-1]
    for row in dataset:
      if row[-1] != first_room:
        return -1

    return first_room


  def split_dataset(self, dataset, split_value, split_attribute):
    """l_dataset = []
    r_dataset = []
    for row in dataset:
      if row[split_attribute] < split_value:
        l_dataset.append(row)
      else:
        r_dataset.append(row)

    return (np.array(l_dataset), np.array(r_dataset))"""
    return dataset[dataset[:,split_attribute] <= split_value], dataset[dataset[:,split_attribute] > split_value]


  def find_split(self, dataset):
    return (split_value, split_attribute)

  def splitDataset(self, data, splitValue):
    
    orderedData = data[data[:,splitValue[0]].argsort()]
    leftDataset = orderedData[orderedData[:,splitValue[0]] <= splitValue[1]]
    rightDataset = orderedData[orderedData[:,splitValue[0]] > splitValue[1]]
  
    return leftDataset, rightDataset

  def decision_tree_learning(self, dataset, depth):
    room_num = self.clean_classification(dataset)
    if(room_num != -1):
      return tree_node(None, room_num, None, None), depth
    else:
      split = self.find_split(dataset) # Assuming here that I'm going to receive a list containing the split value and the attribute to be split
      l_dataset, r_dataset = self.split_dataset(dataset, split[0], split[1])
      l_branch, l_depth = self.decision_tree_learning(l_dataset, depth+1)
      r_branch, r_depth = self.decision_tree_learning(r_dataset, depth+1)
      new_node = tree_node(split[0],split[1], l_branch, r_branch)
      return (new_node, max(l_depth, r_depth))


  def predict(self):
    pass

  def evaluate(self):
    pass

  def plot_tree(self):
    pass


tree = decision_tree()
print(tree.split_dataset(tree.data[::50], -55, 2))