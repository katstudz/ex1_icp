#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

# icp_known_corresp: performs icp given that the input datasets
# are aligned so that Line1(:, QInd(k)) corresponds to Line2(:, PInd(k))


def icp_known_corresp(Line1, Line2, QInd, PInd):
    Q = Line1[:,QInd ]
    P = Line2[:,PInd ]
    print(Q[0])
    print(P[0])
    MuQ = compute_mean(Q)
    MuP = compute_mean(P)

    Q_prim = Q - MuQ
    P_prim = P - MuP

    W = compute_W(Q_prim, P_prim)

    [R, t] = compute_R_t(W, MuQ, MuP)
    print(R)
    print(np.linalg.det(R))
    # Compute the new positions of the points after
    # applying found rotation and translation to them
    NewLine = R @ (Line2 - MuP) + MuQ
    # NewLine = R @ P_prim + MuQ

    E = compute_error(Q, NewLine)

    return NewLine, E

# compute_W: compute matrix W to use in SVD


def compute_W(Q_prim, P_prim): #potencjalny problem
    return Q_prim @ P_prim.transpose()

# compute_R_t: compute rotation matrix and translation vector
# based on the SVD as presented in the lecture


def compute_R_t(W, MuQ, MuP):
    [u, _, vh] = np.linalg.svd(W, full_matrices=False)
    R = u @ vh # zakladam ze to vh tak naprawde jest vh transponsowane
    t = MuQ - R @ MuP
    return R, t


# compute_mean: compute mean value for a [M x N] matrix
def compute_mean(M):
    return np.mean(M, axis=1, keepdims=True)


# compute_error: compute the icp error
def compute_error(Q, OptimizedPoints):
    return np.sum(np.linalg.norm(Q - OptimizedPoints))


# simply show the two lines
def show_figure(Line1, Line2):
    plt.figure()
    plt.scatter(Line1[0], Line1[1], marker='o', s=2, label='Line 1')
    plt.scatter(Line2[0], Line2[1], s=1, label='Line 2')

    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()

    plt.show()


# initialize figure
def init_figure():
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    line1_fig = plt.scatter([], [], marker='o', s=2, label='Line 1')
    line2_fig = plt.scatter([], [], marker='o', s=1, label='Line 2')
    # plt.title(title)
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()

    return fig, line1_fig, line2_fig


# update_figure: show the current state of the lines
def update_figure(fig, line1_fig, line2_fig, Line1, Line2, hold=False):
    line1_fig.set_offsets(Line1.T)
    line2_fig.set_offsets(Line2.T)
    if hold:
        plt.show()
    else:
        fig.canvas.flush_events()
        fig.canvas.draw()
        plt.pause(0.5)


# def find_correspondences(src, dst):
#     knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
#     correspondence = (-1)*np.ones([dst.shape()[1], 1])
#     knn.fit(dst.transpose())
#     distances, indices = knn.kneighbors(src.transpose())
#     dict_unique = {}
#     for i, distance, index in enumerate(zip(distances, indices)):
#         if index in dict_unique:
#             if distance < dict_unique[index][1]:
#                 dict_unique[index] = [i, distance]
#         else:
#             dict_unique[index] = [i, distance]

#     for key in dict_unique:
#         correspondence[dict_unique[key]] = key


#     return None

def find_correspondences(reference, source):
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(reference.transpose())
    distances, indices = knn.kneighbors(source.transpose())

    return indices.ravel()


#CWICZENIE 1
# import numpy as np
# import matplotlib.pyplot as plt
#
# Data = np.load('icp_data.npz')
# Line1 = Data['LineGroundTruth']
# Line2 = Data['LineMovedCorresp']
#
# # Show the initial positions of the lines
# show_figure(Line1, Line2)
#
# # We assume that the there are 1 to 1 correspondences for this data
# QInd = np.arange(len(Line1[0]))
# PInd = np.arange(len(Line2[0]))
#
# # Perform icp given the correspondences
# [Line2, E] = icp_known_corresp(Line1, Line2, QInd, PInd)
#
# # Show the adjusted positions of the lines
# show_figure(Line1, Line2)
#
# # print the error
# print('Error value is: ', E)


#CWICZENIE 2
Data = np.load('icp_data.npz')
Line1 = Data['LineGroundTruth']
Line2 = Data['LineMovedNoCorresp']
#
#
# Line1 = np.array([[-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3]])
# Line2 = np.array([[-2, -1, 0, 1, 2, 3], [2, 1, 0, -1, -2, -3]])
#
# fi = np.pi/4
# rotation = np.array([[np.cos(fi), -np.sin(fi)], [np.sin(fi), np.cos(fi)]])
#
# for column in Line2.T:
#     column = rotation @ column
#
#
MaxIter = 1
Epsilon = 0.001
E = np.inf
#
# # show figure

QInd = np.arange(len(Line1[0]))
PInd = np.arange(len(Line2[0]))

fig, line1_fig, line2_fig = init_figure()

for i in range(MaxIter):

    # TODO: find correspondences of points
    # point with index QInd(1, k) from Line1 corresponds to
    # point with index PInd(1, k) from Line2
    update_figure(fig, line1_fig, line2_fig, Line1, Line2, hold=True)
    PInd = find_correspondences(Line1, Line2)
    update_figure(fig, line1_fig, line2_fig, Line1, Line2, hold=True)
    # update Line2 and error
    # Now that you know the correspondences, use your implementation
    # of icp with known correspondences and perform an update
    EOld = E
    [Line2, E] = icp_known_corresp(Line1, Line2, QInd, PInd)
    show_figure(Line1, Line2)
    print('Error value on ' + str(i) + ' iteration is: ', E)

    # TODO: perform the check if we need to stop iterating
    # if (E < Epsilon or np.abs(E-EOld) < Epsilon):
    #     break
