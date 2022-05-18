from ClassificicationClass import Yolo5Load
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math
import sys
import os
import csv

pre_trained_yolo = Yolo5Load('best.pt')
K = np.identity(3)
K[0, 2] = 640 / 2.0
K[1, 2] = 480 / 2.0
K[0, 0] = K[1, 1] = 640 / (2.0 * np.tan(60 * np.pi / 360.0))
temparray = np.array([[0, 0, 0, 1]])
AXIS_SWAP = np.array(([0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]))
dist_coeffs = np.array([0., 0., 0., 0.])


index1 = 501
if len(sys.argv)-1 == 0:

    index2 = 999

else:

    index2 = int(sys.argv[1])
print("index1: ", index1, "index2: ", index2)
start = index2 + 1
path = os.getcwd()
base_dir = path+'\\merged\\images\\'

def get_R_matrix(yaw, pitch, roll):
    """
    Creates matrix from carla transform.
    """

    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    matrix = np.matrix(np.identity(3))
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix





def pair_based_on_labels(centers1, centers2):
    labels1 = []
    labels2 = []

    RetCenters1 = []
    RetCenters2 = []

    for center in centers1:
        if center[0] not in labels1:
            labels1.append(center[0])

    for center in centers2:
        if center[0] not in labels2:
            labels2.append(center[0])

    labels_temp = [] #avoid finding 2 of the same class
    for center in centers1:
        if center[0] in labels1 and center[0] in labels2 and center[0] not in labels_temp:
            labels_temp.append(center[0])
            RetCenters1.append(center[2])

    labels_temp = []
    for center in centers2:
        if center[0] in labels1 and center[0] in labels2 and center[0] not in labels_temp:
            labels_temp.append(center[0])
            RetCenters2.append(center[2])

    return RetCenters1, RetCenters2


def GetLandmarkCenters(pic):
    centers = []
    bbs = pre_trained_yolo.classify(pic)
    for bb in bbs:
        try:
            label = int(bb[0])
            [x, y, w, h] = bb[1]
            x = int(pic.shape[1] * x)
            y = int(pic.shape[0] * y)
            w = int(pic.shape[1] * w / 2)
            h = int(pic.shape[0] * h / 2)
            centers.append((label, bb[2].item(), (x, y)))
            #test_pic_ROI = pic[y - h:y + h, x - w:x + w]

        except IndexError:
            raise
    centers.sort(key=lambda a: (a[0], a[1]), reverse=True)
    #centers = del_duplicate(centers)
    return centers


def GetImgPoints(pic1, pic2):
    pic1 = cv.resize(pic1, (640, 480))
    pic2 = cv.resize(pic2, (640, 480))

    centers1 = GetLandmarkCenters(pic1)
    centers2 = GetLandmarkCenters(pic2)

    centers1, centers2 = pair_based_on_labels(centers1, centers2)
    return centers1, centers2

def CalcWorldPoints(centers1, centers2,  poz1, poz2):

    P_first = np.dot(np.dot(np.concatenate((K, np.zeros((3, 1))), axis=1), AXIS_SWAP),
                     np.linalg.inv(np.concatenate((np.concatenate((poz1['R'], poz1['t']), axis=1), temparray), axis=0)))
    P_second = np.dot(np.dot(np.concatenate((K, np.zeros((3, 1))), axis=1), AXIS_SWAP),
                      np.linalg.inv(np.concatenate((np.concatenate((poz2['R'], poz2['t']), axis=1), temparray), axis=0)))

    world_points = np.zeros((len(centers1), 1, 3))

    for i in range(len(centers1)):
        X = cv.triangulatePoints(P_first, P_second, centers1[i], centers2[i])
        X /= X[3]
        world_points[i, 0, 0] = X[0, 0]
        world_points[i, 0, 1] = X[1, 0]
        world_points[i, 0, 2] = X[2, 0]

    return world_points

#load images
imgs = []
grays = []
for i, filename in enumerate(os.listdir(base_dir)):
    if index1 <= i <= index2:
        if filename.endswith('.png'):
            frame = cv.imread(base_dir+filename)
            imgs.append(frame)
            grays.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))


# read x, y, z, yaw, pitch, roll for the two frames
csv_reader = csv.DictReader(open(base_dir+'result.csv', 'r'))
flight_data = list(csv_reader)

first_t = [[float(flight_data[index1]["x"])], [float(flight_data[index1]["y"])], [float(flight_data[index1]["z"])]]
first_yaw = (float(flight_data[index1]["yaw"])/180)*3.141
first_pitch = (float(flight_data[index1]["pitch"])/180)*3.141
first_roll = (float(flight_data[index1]["roll"])/180)*3.141

second_t = [[float(flight_data[index2]["x"])], [float(flight_data[index2]["y"])], [float(flight_data[index2]["z"])]]
second_yaw = (float(flight_data[index2]["yaw"])/180)*3.141
second_pitch = (float(flight_data[index2]["pitch"])/180)*3.141
second_roll = (float(flight_data[index2]["roll"])/180)*3.141

first_R = get_R_matrix(first_yaw, first_pitch, first_roll)
second_R = get_R_matrix(second_yaw, second_pitch, second_roll)
poz1 = {'R' : first_R,
        't' : first_t,}
poz2 = {'R' : second_R,
        't' : second_t,}

centers1, centers2 = GetImgPoints(imgs[0], imgs[-1])

world_points = CalcWorldPoints(centers1, centers2, poz1, poz2)

centers1 = np.array(centers1, dtype=np.float32).reshape(len(centers1), 1, 2)
centers2 = np.array(centers2, dtype=np.float32).reshape(len(centers2), 1, 2)


img1 = cv.resize(imgs[0], (640, 480))
for u, point in enumerate(centers1):
    a, b = point.ravel()
    img1 = cv.circle(img1, (int(a), int(b)), 3, [0, 0, 255], 1)
cv.imshow('1', img1)

img2 = cv.resize(imgs[-1], (640, 480))
for u, point in enumerate(centers2):
    a, b = point.ravel()
    img2 = cv.circle(img2, (int(a), int(b)), 3, [0, 0, 255], 1)
cv.imshow('2', img2)
cv.waitKey(1000)

_, rvecs, tvecs = cv.solvePnP(world_points, centers1, K, dist_coeffs)

rotation_matrix = np.zeros(shape=(3, 3))
cv.Rodrigues(rvecs, rotation_matrix)
temparray = np.array([[0, 0, 0, 1]])
Rt_mat = np.linalg.inv( np.dot(AXIS_SWAP.T, np.concatenate((np.concatenate((rotation_matrix, tvecs), axis=1), temparray), axis=0)))
rvecs = Rt_mat[0:3, 0:3]*-1
tvecs = Rt_mat[0:3, 3]


x, y, z = tvecs


# show points IN a 3D space
# feature points in red
Measure_limit = 500
X = world_points
xdata = X[:Measure_limit, 0, 0]
ydata = X[:Measure_limit, 0, 1]
zdata = X[:Measure_limit, 0, 2]

# ground truth drone position, green line
T_GT = []
for i in range(len(flight_data)):
    T_GT.append([])
    T_GT[i].append(float(flight_data[i]["x"]))
    T_GT[i].append(float(flight_data[i]["y"]))
    T_GT[i].append(float(flight_data[i]["z"]))
T_GT = np.array(T_GT)
x_gt = T_GT[:, 0]
y_gt = T_GT[:, 1]
z_gt = T_GT[:, 2]

# real world reference
# real reference point (building)
LM_Origin = (-40.54, -34.21, 29.54)
LM_Extent = (19.13, 17.86, 29.51)
cords = np.zeros((8, 4))
cords[0, :] = np.array([LM_Origin[0]+LM_Extent[0], LM_Origin[1]+LM_Extent[1], LM_Origin[2]-LM_Extent[2], 1])
cords[1, :] = np.array([LM_Origin[0]-LM_Extent[0], LM_Origin[1]+LM_Extent[1], LM_Origin[2]-LM_Extent[2], 1])
cords[2, :] = np.array([LM_Origin[0]-LM_Extent[0], LM_Origin[1]-LM_Extent[1], LM_Origin[2]-LM_Extent[2], 1])
cords[3, :] = np.array([LM_Origin[0]+LM_Extent[0], LM_Origin[1]-LM_Extent[1], LM_Origin[2]-LM_Extent[2], 1])
cords[4, :] = np.array([LM_Origin[0]+LM_Extent[0], LM_Origin[1]+LM_Extent[1], LM_Origin[2]+LM_Extent[2], 1])
cords[5, :] = np.array([LM_Origin[0]-LM_Extent[0], LM_Origin[1]+LM_Extent[1], LM_Origin[2]+LM_Extent[2], 1])
cords[6, :] = np.array([LM_Origin[0]-LM_Extent[0], LM_Origin[1]-LM_Extent[1], LM_Origin[2]+LM_Extent[2], 1])
cords[7, :] = np.array([LM_Origin[0]+LM_Extent[0], LM_Origin[1]-LM_Extent[1], LM_Origin[2]+LM_Extent[2], 1])

# building for reference, black points
xbuilding = cords[:, 0]
ybuilding = cords[:, 1]
zbuilding = cords[:, 2]

ax = plt.axes(projection='3d')
ax.set_xlim3d(-100, 0)
ax.set_ylim3d(-250, -150)
ax.set_zlim3d(-43, 56)
ax.scatter3D(x_gt[index1], y_gt[index1], z_gt[index1], c='g')
ax.scatter3D(x, y, z)
print((x-x_gt[index1], y-y_gt[index1], z-z_gt[index1]))
ax.plot(x_gt, y_gt, z_gt, c='g') #plot
ax.scatter3D(xdata, ydata, zdata, c='r')
ax.scatter3D(xbuilding, ybuilding, zbuilding, c='k')
plt.show()
cv.waitKey(312)

