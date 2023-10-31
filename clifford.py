import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as matColours

# A simple decision tree structure for testing
# tree_structure = {
#     'feature_name': 'Feature A',
#     'threshold': 5,
#     'left': {
#         'feature_name': 'Feature B',
#         'threshold': 3,
#         'left': {
#             'class': 'Room1'
#         },
#         'right': {
#             'class': 'Room3'
#         }
#     },
#     'right': {
#         'feature_name': 'Feature B',
#         'threshold': 3,
#         'left': {
#             'class': 'Room2'
#         },
#         'right': {
#             'feature_name': 'Feature B',
#             'threshold': 3,
#             'left': {
#                 'class': 'Room2'
#             },
#             'right': {
#                 'class': 'Room3'
#             }
#         }
#     }
# }

def plot_nodes(node, x, y, dx, dy, segments, line_colour_index, colour, depth, max_depth):
    if depth > max_depth[0]:
      max_depth[0] = depth

    if node.value != None:
        # Internal node
        left_x, left_y = x - dx, y - dy
        right_x, right_y = x + dx, y - dy

        segments.append([[x, y], [left_x, left_y]])
        segments.append([[x, y], [right_x, right_y]])

        line_colour_index.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][colour])
        line_colour_index.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][colour])
        colour+=1

        plot_nodes(node.l_branch, left_x, left_y, dx / 2, dy, segments, line_colour_index, colour, depth+1, max_depth)
        plot_nodes(node.r_branch, right_x, right_y, dx / 2, dy, segments,line_colour_index,colour+1, depth+1, max_depth)
        plt.text(x, y, f"{node.attr} <= {node.value}\n depth: {str(depth)}", ha='center', bbox=dict(facecolor='white', edgecolor="royalblue"))

    else:
        # Leaf node
        plt.text(x, y, f"Class: {node.attr}\n depth: {str(depth)}", ha='center', bbox=dict(facecolor='white', edgecolor="royalblue"))

def plot_decision_tree(node):
    # The following values are arbitrary
    dx, dy = 16, 2

    # max_depth required for canvas dimensions
    max_depth = [0]  

    # plt.figure(figsize=(width, height))

    segments = []
    line_colour = []
    plot_nodes(node, 0, 0, dx, dy, segments,line_colour, 0, 0, max_depth)

    line_colours = [matColours.to_rgba(c) for c in line_colour]
    line_segments = LineCollection(segments, linewidths=1, colors=line_colours, linestyle='solid')

    plt.xlim(-dx*(max_depth[0]-1), dx*(max_depth[0]-1))
    plt.ylim(-max_depth[0]*dy, 0)
    plt.axis('off')  # Hide axes

    # Add line segments to the plot
    plt.gca().add_collection(line_segments)

    # Display the plot
    plt.show()

# run the following and just give plot_decision_tree() the starting node
plt.figure(figsize=(15, 5))
plot_decision_tree(tree)

"""
TODO:
- check if it works on the sklearn datasets and final dataset, as adjustments to recursion (node['left'] and node['right']) and the conditional (if 'feature_name' in node:) may be required during final implementation
"""