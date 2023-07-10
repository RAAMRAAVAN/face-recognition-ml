import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_curvature(x, y, z, landmark_1, landmark_2):
    # Select the range of points between the two landmarks
    start_index = min(landmark_1, landmark_2)
    end_index = max(landmark_1, landmark_2)
    x_range = x[start_index:end_index+1]
    y_range = y[start_index:end_index+1]
    z_range = z[start_index:end_index+1]

    # Compute the first and second derivatives
    dx = np.gradient(x_range)
    dy = np.gradient(y_range)
    dz = np.gradient(z_range)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)

    # Compute the curvature
    numerator = np.sqrt((ddy*dz - ddz*dy)**2 + (ddz*dx - ddx*dz)**2 + (ddx*dy - ddy*dx)**2)
    denominator = (dx**2 + dy**2 + dz**2)**(3/2)
    curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    # Return the local curvature at the midpoint between the landmarks
    midpoint_index = (landmark_1 + landmark_2) // 2 - start_index
    return curvature[midpoint_index]


def calc_local_curvature(new_x_axis, new_y_axis, new_z_axis, A, S):
    # print(A, S)
    localCurvature = []
    temp = []
    for i in range(len(S)):
        localCurvature.append(compute_curvature(new_x_axis, new_y_axis, new_z_axis, S[i], A))
    return(localCurvature)