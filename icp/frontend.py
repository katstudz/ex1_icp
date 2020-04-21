#!/usr/bin/env python3

import ex1_1 as ex
import numpy as np
import matplotlib.pyplot as plt

Data = np.load('icp_data.npz')
Line1 = Data['LineGroundTruth']
Line2 = Data['LineMovedCorresp']

# Show the initial positions of the lines
ex.show_figure(Line1, Line2)
#%%

# We assume that the there are 1 to 1 correspondences for this data
QInd = np.arange(len(Line1[0]))
PInd = np.arange(len(Line2[0]))

# Perform icp given the correspondences
[Line2, E] = ex.icp_known_corresp(Line1, Line2, QInd, PInd)

# Show the adjusted positions of the lines
ex.show_figure(Line1, Line2)

# print the error
print('Error value is: ', E)

Data = np.load('icp_data.npz')
Line1 = Data['LineGroundTruth']
Line2 = Data['LineMovedNoCorresp']

MaxIter = 10
Epsilon = 0.001
E = np.inf

# show figure
ex.show_figure(Line1, Line2)

# for i in range(MaxIter):
#     # TODO: find correspondences of points
#     # point with index QInd(1, k) from Line1 corresponds to
#     # point with index PInd(1, k) from Line2
#     QInd = ...
#     PInd = ...
#
#     # update Line2 and error
#     # Now that you know the correspondences, use your implementation
#     # of icp with known correspondences and perform an update
#     EOld = E
#     [Line2, E] = ex.icp_known_corresp(Line1, Line2, QInd, PInd)
#
#     print('Error value on ' + str(i) + ' iteration is: ', E)
#
#     # TODO: perform the check if we need to stop iterating
#     ...
