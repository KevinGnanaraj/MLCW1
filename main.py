import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as matColours
import sys

class tree_node:

  def __init__(self, value=None, attr=None, l_branch=None, r_branch=None):
    self.value = value
    self.attr = attr
    self.l_branch = l_branch
    self.r_branch = r_branch

class decision_tree:

  def __init__(self):
    self.decision_tree = tree_node()

    if len(sys.argv) == 2:
      if sys.argv[1] == 'c':
        self.data = np.loadtxt("wifi_db/clean_dataset.txt")
      elif sys.argv[1] == 'n':
        self.data = np.loadtxt("wifi_db/noisy_dataset.txt")
      else:
        sys.exit("Please enter a valid dataset option")
    else:
      self.data = np.loadtxt("wifi_db/clean_dataset.txt")

    if len(self.data) == 0:
      print("Why have you fed us an empty dataset??  :(")
      return 0

    rng = np.random.default_rng()
    rng.shuffle(self.data)

  def train_test_split(self, index):
    train_split = 0.9
    test_split = 0.1
    num_splits = 10
    split_size = len(self.data) // num_splits
    split_start = index * split_size
    split_end = split_start + split_size
    self.train_set = np.concatenate(
        (self.data[:split_start], self.data[split_end:]), axis=0)
    self.test_set = self.data[split_start:split_end]

    return 0

  def clean_classification(self, dataset):  # If the dataset contains only samples from the same room return the room numbers, else return -1
    first_room = dataset[0][-1]
    for row in dataset:
      if row[-1] != first_room:
        return -1

    return first_room

  def entropy(self, data):
    # Want to find the probability of each label, rather than relative frequency
    # because it is a multivalued class
    entropy = 0
    uniqueValues, freq = np.unique(data[:, -1], return_counts=True)
    for i in range(uniqueValues.shape[0]):
      entropy += -1 * (freq[i] / np.sum(freq)) * np.log2(
          freq[i] / np.sum(freq))
    return entropy

  def infGain(self, S, S_Left, S_Right):

    H_S = self.entropy(S)
    H_S_Left = self.entropy(S_Left)
    H_S_Right = self.entropy(S_Right)
    total_rows, left_rows, right_rows = S.shape[0], S_Left.shape[
        0], S_Right.shape[0]

    #Left and Right Proportions of the dataset
    prop_left = (left_rows / total_rows) * H_S_Left
    prop_right = (right_rows / total_rows) * H_S_Right

    remainder = prop_left + prop_right

    gain = H_S - remainder
    return gain


  def splitDataset(self, data, splitValue):                   # We split on the attribue, value so split = [attr, val]
    orderedData = data[data[:, splitValue[0]].argsort()]
    leftDataset = orderedData[orderedData[:, splitValue[0]] <= splitValue[1]]
    rightDataset = orderedData[orderedData[:, splitValue[0]] > splitValue[1]]

    return leftDataset, rightDataset

  def findSplit(self, data):
    maxInfGain = 0
    bestSplit = [0, 0]  # Attr, Value
    # Sort against attributes for each class label
    for attr in range(data.shape[1] - 1):  # Number of columns excluding the class column
      ordered = data[data[:, attr].argsort()]
      for row in range(1, ordered.shape[0]):  # 1 to size of attributes column
        if ordered[row, attr] != ordered[row - 1, attr]:
          value = ordered[row - 1, attr]
          split = [attr, value]
          leftDataset, rightDataset = self.splitDataset(data, split)
          gain = self.infGain(ordered, leftDataset, rightDataset)

          if gain > maxInfGain:
            maxInfGain = gain
            bestSplit = split

    return bestSplit

  def decision_tree_learning(self, dataset, depth):

    room_num = self.clean_classification(dataset)
    if room_num != -1:
      return tree_node(None, room_num, None, None), depth
    else:
      split = self.findSplit(dataset)  # Assuming here that I'm going to receive a list containing the split value and the attribute to be split
      # findSplit() returns [Class, Attribute]
      l_dataset, r_dataset = self.splitDataset(dataset, split)
      l_branch, l_depth = self.decision_tree_learning(l_dataset, depth + 1)
      r_branch, r_depth = self.decision_tree_learning(r_dataset, depth + 1)
      new_node = tree_node(split[1], split[0], l_branch, r_branch)
      return new_node, max(l_depth, r_depth)

  def check_nodes(self, node, nodes_present, node_number, depth):
    if node.value != None:
      nodes_present.add(node_number)
      self.check_nodes(node.l_branch, nodes_present, (node_number<<1)+1, depth+1)
      self.check_nodes(node.r_branch, nodes_present, (node_number<<1)+2, depth+1)

  def plot_nodes(self, node, x, y, dx, dy, segments, line_colour_index, colour, depth, dimensions, nodes_present, node_number):

    dimensions[0] = max(dimensions[0], depth)
    dimensions[1] = min(dimensions[1], x)
    dimensions[2] = max(dimensions[2], x)

    if node.value != None:
      left_x, left_y = x - dx, y - dy
      right_x, right_y = x + dx, y - dy

      if node_number-1 not in nodes_present and node_number != (1<<depth)-1:
        left_x -= dx//2
        left_dx = dx
      else:
        left_dx = dx//2

      if node_number+1 not in nodes_present and node_number != (1<<depth+1)-2:
        right_x += dx//2
        right_dx = dx
      else:
        right_dx = dx//2

      segments.append([[x, y], [left_x, left_y]])
      segments.append([[x, y], [right_x, right_y]])

      if colour>9:
        colour = 0

      line_colour_index.append(
          plt.rcParams['axes.prop_cycle'].by_key()['color'][colour])
      line_colour_index.append(
          plt.rcParams['axes.prop_cycle'].by_key()['color'][colour])

      colour += 1

      self.plot_nodes(node.l_branch, left_x, left_y, max(left_dx, 2), dy, segments,
                      line_colour_index, colour, depth + 1, dimensions, 
                      nodes_present, (node_number<<1)+1)
      self.plot_nodes(node.r_branch, right_x, right_y, max(right_dx, 2), dy, segments,
                      line_colour_index, colour + 1, depth + 1, dimensions,
                      nodes_present, (node_number<<1)+2)
      plt.text(x,
               y,
               f"Signal {node.attr} \n <= {node.value}",
               ha='center',
               bbox=dict(facecolor='white', edgecolor="royalblue", 
                         boxstyle='round,pad=0.2'), 
               fontsize = 5, clip_on=True)

    else:
      plt.text(x,
               y,
               f"Room: {node.attr}\n depth: {str(depth)}",
               ha='center',
               bbox=dict(facecolor='white', edgecolor="royalblue", 
                         boxstyle='round,pad=0.2'), 
               fontsize = 5, clip_on=True)

  def plot_decision_tree(self, node):
    # dy is arbitrary
    dx, dy = 200, 2

    # dimensions required for canvas dimensions, depth, left, and right
    dimensions = [0, 0, 0] 

    segments = []
    line_colour = []
    nodes_present = set()
    self.check_nodes(node, nodes_present, 0, 0)
    self.plot_nodes(node, 0, 0, dx, dy, segments, line_colour, 0, 0, 
                    dimensions, nodes_present, 0)

    line_colours = [matColours.to_rgba(c) for c in line_colour]
    line_segments = LineCollection(segments,
                                   linewidths=1,
                                   colors=line_colours,
                                   linestyle='solid')
    print("Maximum Tree Depth is:", dimensions[0])
    plt.xlim(dimensions[1]-15, dimensions[2]+15)
    plt.ylim(-dimensions[0] * dy-2, 0+2)
    plt.axis('off')  # Hide axes

    # Add line segments to the plot
    plt.gca().add_collection(line_segments)

    # Display the plot
    plt.show()

  def predict(self, tree):
    # Go through the test dataset and run each row through the decision tree.
    # Return a numpy array of the predicted values.
    self.predictions = []
    for row in self.test_set:
      self.predictions.append(self.navigate_tree(row, tree))

    return 0

  def navigate_tree(self, row, node):
    if node.value != None:
      if row[node.attr] <= node.value:
        return self.navigate_tree(row, node.l_branch)
      else:
        return self.navigate_tree(row, node.r_branch)
    else:
      return node.attr

  def confusion_matrix(self):
    matrix = np.zeros((4, 4), dtype="int")

    for i in range(len(self.test_set)):
      num_true = int(self.test_set[i][-1] - 1)
      num_predicted = int(self.predictions[i] - 1)
      matrix[num_true][num_predicted] += 1
      
    return matrix

  def evaluation_metrics(self, matrix):
    classes = len(matrix[0])
    accuracy_scores = np.zeros(classes)
    precision_scores = np.zeros(classes)
    recall_scores = np.zeros(classes)
    f1_scores = np.zeros(classes)

    for i in range(len(matrix[0])):
      True_Positives = matrix[i][i]  # True positives are along the diagonal
      False_Positives = sum(matrix[:, i]) - True_Positives
      False_Negatives = sum(matrix[i, :]) - True_Positives
      True_Negatives = np.sum(matrix) - True_Positives - False_Positives - False_Negatives  # True Negatives are classes that are correctly predicted not to be i

      Accuracy = (True_Positives + True_Negatives) / (True_Negatives + True_Positives + False_Positives + False_Negatives)
      Precision = True_Positives / (True_Positives + False_Positives)
      Recall = True_Positives / (True_Positives + False_Negatives)
      F1 = 2 * ((Precision * Recall) / (Precision + Recall))

      accuracy_scores[i] = Accuracy
      precision_scores[i] = Precision
      recall_scores[i] = Recall
      f1_scores[i] = F1

    average_accuracy = round(np.mean(accuracy_scores), 3)
    average_precision = round(np.mean(precision_scores), 3)
    average_recall = round(np.mean(recall_scores), 3)
    average_f1 = round(np.mean(f1_scores), 3)

    return (average_accuracy, average_precision, average_recall, average_f1)

  def evaluate(self):
    conf_matrix = self.confusion_matrix()
    accuracy, precision, recall, f1 = self.evaluation_metrics(conf_matrix)
    return (accuracy, precision, recall, f1)


if __name__ == "__main__":
  if len(sys.argv) == 2:
    if sys.argv[1] == 'c':
      print("Using clean dataset")
    elif sys.argv[1] == 'n':
      print("Using noisy dataset")
    else:
      sys.exit("Please enter a valid dataset option")
  else:
    print("Defaulting to using the clean dataset")

  accuracy_scores = []
  precision_scores = []
  recall_scores = []
  f1_scores = []

  for i in range(10):
    print("Starting loop " + str(i) + ":")
    tree = decision_tree()
    tree.train_test_split(i)
    tree.decision_tree, depth = tree.decision_tree_learning(tree.train_set, depth=0)
    tree.predict(tree.decision_tree)
    accuracy, precision, recall, f1 = tree.evaluate()

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print("Accuracy score", accuracy)
    print("Precision score:", precision)
    print("Recall score:", recall)
    print("F1 score:", f1)

    if i == 9:
      plt.figure(figsize=(15, 5))
      tree.plot_decision_tree(tree.decision_tree)

  acc_average = round(np.mean(accuracy_scores), 3)
  prec_average = round(np.mean(precision_scores), 3)
  recall_average = round(np.mean(recall_scores), 3)
  f1_average = round(np.mean(f1_scores), 3)

  print("Macro-Averaged Accuracy:", acc_average)
  print("Macro-Averaged Precision:", prec_average)
  print("Macro-Averaged Recall:", recall_average)
  print("Macro-Averaged F1:", f1_average)
