import numpy as np

def entropy(data):
  
  # Want to find the probability of each label, rather than relative frequency 
  # because it is a multivalued class
  entropy = 0
  uniqueValues, freq = np.unique(data[:,-1], return_counts=True)
  for i in range(uniqueValues.shape[0]):
    entropy += -1 * (freq[i]/np.sum(freq)) * np.log2(freq[i]/np.sum(freq))    
  return entropy


def infGain(S, S_Left, S_Right): 

  H_S = entropy(S)
  H_S_Left = entropy(S_Left)
  H_S_Right = entropy(S_Right)
  total_rows, left_rows, right_rows  = S.shape[0], S_Left.shape[0], S_Right.shape[0]

  #Left and Right Proportions of the dataset
  prop_left = (left_rows / total_rows) * H_S_Left
  prop_right = (right_rows/ total_rows) * H_S_Right

  remainder = prop_left + prop_right

  gain = H_S - remainder
  return gain


# We split on the attribue, value so split = [attr, val]

def splitDataset(data, splitValue):
  
  orderedData = data[data[:,splitValue[0]].argsort()]
  leftDataset = orderedData[orderedData[:,splitValue[0]] <= splitValue[1]]
  rightDataset = orderedData[orderedData[:,splitValue[0]] > splitValue[1]]  
      
  return leftDataset, rightDataset

def findSplit(data):  
  maxInfGain = 0
  bestSplit = [0, 0] # Class, Attribute
  # Sort against attributes for each class label
  for attr in range(data.shape[1] - 1):  # Number of columns excluding the class column 
      ordered = data[data[:,attr].argsort()]
      for row in range(1, ordered.shape[0]): # 1 to size of attributes column
          if ordered[row, attr] != ordered[row - 1, attr]:          
              value = ordered[row - 1, attr]     
              split = [attr, value]
              leftDataset, rightDataset = splitDataset(data, split) 

              gain = infGain(ordered, leftDataset, rightDataset)
              
              if gain > maxInfGain:
                  maxInfGain = gain
                  bestSplit = split

  return bestSplit

  

  
# This loads the file into a 2000x8 array
#data = np.loadtxt("wifi_db/clean_dataset.txt")  
#print(findSplit(data))







  

