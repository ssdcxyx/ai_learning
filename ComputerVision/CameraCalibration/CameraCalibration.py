# -*- coding: utf-8 -*-
# @Time    : 2018/10/19 下午9:39
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : CameraCalibration.py

import cv2
import numpy as np
import glob

# Question 6 answer
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# checkerboard size
W, H = 9, 6
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
obj_point = np.zeros((W*H, 3), np.float32)
# Remove the Z coordinate and record it as a two-dimensional matrix.
obj_point[:, :2] = np.mgrid[0:W, 0:H].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
obj_points = []   # 3d point in real world space
img_points = []   # 2d points in image plane.

# get all image path
images = glob.glob('figure/example/left/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (W, H), None)
    # If found, add object points, image points (after refining them)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(obj_point)
        img_points.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (W, H), corners2, ret)
        cv2.imshow('findCorners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# camera calibration
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Question 7 answer
img2 = cv2.imread('figure/example/left/left01.jpg')
img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
h, w = img2.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))

# undistort
# method 1
# dst = cv2.undistort(img2, cameraMatrix, distCoeffs, None, new_camera_matrix)

# method 2
map_x, map_y = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, new_camera_matrix, (w, h), 5)
dst = cv2.remap(img2, map_x, map_y, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y: y+h, x: x+w]
cv2.imwrite('figure/example/left/left01_c.jpg', dst)
