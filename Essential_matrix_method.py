import csv
import sys


import numpy as np
import cv2 as cv
import os
import math

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# constants, color used to display points
path = os.getcwd()
color = np.random.randint(0, 255, (300, 3))

base_dir = path+'\\F01_no_auto\\'
#base_dir = path+'\\Flights2\\01_LM_1\\'

AXIS_ORDERS = np.array([[0, 1, 2],
               [0, 2, 1],
               [1, 0, 2],
               [1, 2, 0],
               [2, 0, 1],
               [2, 1, 0]])
if len(sys.argv)-1 == 0:
    start = 2
    #AXIS_SWAP = np.array(([0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]))
    #AXIS_SWAP = np.eye(4)
    AXIS_SWAP = np.array(([0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]))
    print(1)
else:
    AXIS_SWAP = np.eye(3)[:, AXIS_ORDERS[int(sys.argv[1])]]
    if int(sys.argv[2]) != 0:
        AXIS_SWAP[:, 0] = AXIS_SWAP[:, 0]*-1
    if int(sys.argv[3]) != 0:
        AXIS_SWAP[:, 1] = AXIS_SWAP[:, 1]*-1
    if int(sys.argv[4]) != 0:
        AXIS_SWAP[:, 2] = AXIS_SWAP[:, 2]*-1
    print('Orders: ', AXIS_ORDERS[int(sys.argv[1])], " ", int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    AXIS_SWAP = np.concatenate((AXIS_SWAP, np.array((0, 0, 0)).reshape(1,3)))
    AXIS_SWAP = np.concatenate((AXIS_SWAP, np.array((0, 0, 0, 1)).reshape(4,1)), axis=1)
    print(AXIS_SWAP)
    start = 2


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







Feature_Params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

Lk_Params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

K = np.identity(3)
K[0, 2] = 1280 / 2.0
K[1, 2] = 960 / 2.0
K[0, 0] = K[1, 1] = 1280 / (2.0 * np.tan(60 * np.pi / 360.0))
temparray = np.array([[0, 0, 0, 1]])
#AXIS_SWAP = np.array(([0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]))




#load images
imgs = []
grays = []
for i, filename in enumerate(os.listdir(base_dir)):
    if filename.endswith('.png'):
        frame = cv.imread(base_dir+filename)
        imgs.append(frame)
        grays.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))


img_points = first_points = cv.goodFeaturesToTrack(grays[start], mask=None, **Feature_Params)
prev_img = grays[start]

# read x, y, z, yaw, pitch, roll for the two frames
csv_reader = csv.DictReader(open(base_dir+'result.csv', 'r'))
flight_data = list(csv_reader)

first_t = np.array(([float(flight_data[start]["x"])], [float(flight_data[start]["y"])], [float(flight_data[start]["z"])]))
first_R = get_R_matrix((float(flight_data[start]["yaw"])/180)*3.141, (float(flight_data[start]["pitch"])/180)*3.141, (float(flight_data[start]["roll"])/180)*3.141)
# ground truth drone position, green line
length = 0
ts = []
Rs = []
R_gts = []
testt = []
testRtmat = []
for i, filename in enumerate(os.listdir(base_dir)):
    if filename.endswith(".png"):
            if i > start:
                next_frame = imgs[i]
                next_img = grays[i]
                img_points, st, err = cv.calcOpticalFlowPyrLK(prev_img, next_img, img_points, None, **Lk_Params)
                img_points = img_points[st == 1].reshape(-1, 1, 2)
                first_points = first_points[st == 1].reshape(-1, 1, 2)
                #for u, point in enumerate(img_points):
                #    a, b = point.ravel()
                #    img = cv.circle(imgs[i], (int(a), int(b)), 3, [0, 0, 255], 1)
                ## cv.imwrite(path+'/saved/' + f"{i:05d}" + '.png', img)
                #cv.imshow('before', img)
                #cv.waitKey(10)

                if i==200:
                    print("test")
                prev = np.array([[float(flight_data[i - 1]["x"])], [float(flight_data[i - 1]["y"])],[float(flight_data[i - 1]["z"])]])
                curr = np.array([[float(flight_data[i]["x"])], [float(flight_data[i]["y"])], [float(flight_data[i]["z"])]])
                length += math.sqrt((prev[0][0]-curr[0][0])**2 + (prev[1][0]-curr[1][0])**2 + (prev[2][0]-curr[2][0])**2)

                EssentialMat, mask = cv.findEssentialMat(first_points, img_points, cameraMatrix=K, prob=0.9)
                retval, R, t, mask = cv.recoverPose(EssentialMat, first_points, img_points, cameraMatrix=K, mask=mask)
                R1, R2, _ = cv.decomposeEssentialMat(EssentialMat)
                if np.abs(np.sum(np.eye(3)-np.abs(R1))) < np.abs(np.sum(np.eye(3)-np.abs(R2))):
                   R = R1
                else:
                   R = R2
                Rt_mat = np.linalg.inv(np.concatenate((np.concatenate((R, t), axis=1), temparray), axis=0))
                R = Rt_mat[0:3, 0:3]
                Rcopy = np.copy(R)
                R[1, 0] = Rcopy[0, 2]
                R[2, 0] = Rcopy[1, 2]
                R[0, 1] = Rcopy[2, 0]
                R[0, 2] = -1*Rcopy[2, 1]
                R[1, 2] = Rcopy[1, 0]
                R[2, 1] = Rcopy[0, 1]
                Rs.append(R)
                t = np.dot(AXIS_SWAP[0:3, 0:3], Rt_mat[0:3, 3])
                ts.append(first_t+np.dot(first_R, length*t.reshape(3,1)))

                R_gts.append(np.dot(np.linalg.inv(get_R_matrix((float(flight_data[start]["yaw"]) / 180) * 3.141, (float(flight_data[start]["pitch"]) / 180) * 3.141, (float(flight_data[start]["roll"]) / 180) * 3.141)),
                             get_R_matrix((float(flight_data[i]["yaw"]) / 180) * 3.141, (float(flight_data[i]["pitch"]) / 180) * 3.141, (float(flight_data[i]["roll"]) / 180) * 3.141)))

                R_gt = np.array(R_gts[-1])
                prev_img = next_img






# ground truth
T_GT = []
for i in range(start+1, len(flight_data)):
    T_GT.append([])
    T_GT[i-(start+1)].append(float(flight_data[i]["x"]))
    T_GT[i-(start+1)].append(float(flight_data[i]["y"]))
    T_GT[i-(start+1)].append(float(flight_data[i]["z"]))
T_GT = np.array(T_GT)
x_gt = T_GT[:, 0]
y_gt = T_GT[:, 1]
z_gt = T_GT[:, 2]


# measured position of the drone in blue
T_total = np.array(ts).reshape(len(ts), 3)





x = T_total[:, 0]
y = T_total[:, 1]
z = T_total[:, 2]





# calculating mean squared error in drone position
mse = (np.sqrt((x-x_gt)**2 + (y-y_gt)**2 + (z-z_gt)**2))

#plt.plot(R_err)
#plt.ylabel('R_err')
#plt.show()

print('avg mse: ', np.average(mse), '\n')
t_start = np.array(([float(flight_data[10]["x"])], [float(flight_data[10]["y"])], [float(flight_data[10]["z"])]))-\
          np.array(([float(flight_data[start]["x"])], [float(flight_data[start]["y"])], [float(flight_data[start]["z"])]))
if len(sys.argv)-1 == 0:
    #plt.plot(mse)
    #plt.ylabel('Mean Squared Error')
    #plt.show()

    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-120, -20)
    ax.set_ylim3d(-350, -250)
    ax.set_zlim3d(-44, 56)
    ax.scatter3D(x, y, z)
    ax.scatter3D(x_gt, y_gt, z_gt, c='g')
    length = 0
    for i in range(0, len(Rs), 10):
        #length+=0.5
        t = t_start
        t = np.dot(Rs[i], t)#*length


        a = Arrow3D([x[i], x[i]+t[0][0]], [y[i], y[i]+t[1][0]],
                    [z[i], z[i]+t[2][0]], mutation_scale=10,
                    lw=3, arrowstyle="-|>", color="b")
        ax.add_artist(a)

    for i in range(0, len(R_gts), 10):
        #length+=0.5
        t = t_start
        t = np.array(np.dot(R_gts[i], t))#*length


        a = Arrow3D([x_gt[i], x_gt[i]+t[0][0]], [y_gt[i], y_gt[i]+t[1][0]],
                    [z_gt[i], z_gt[i]+t[2][0]], mutation_scale=10,
                    lw=3, arrowstyle="-|>", color="g")
        ax.add_artist(a)

    plt.show()
