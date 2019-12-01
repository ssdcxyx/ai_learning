# -*- coding: utf-8 -*-
# @Time    : 2018/10/28 下午9:55
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : StereoCalibration.py

import cv2
import numpy as np
import glob
import ruamel.yaml as yaml


# Arrays to store object points and image points from all the images.
obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane.


# Question 12 answer
def cameraCalibration(path, name):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # checkerboard size
    W, H = 9, 6
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_point = np.zeros((W * H, 3), np.float32)
    # Remove the Z coordinate and record it as a two-dimensional matrix.
    obj_point[:, :2] = np.mgrid[0:W, 0:H].T.reshape(-1, 2)

    # get all image path
    images = glob.glob(path)
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
            cv2.imshow(name, img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    # camera calibration
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None,
                                                                         None)
    return cameraMatrix, distCoeffs, obj_points, img_points, gray.shape[::-1]


if __name__ == '__main__':
    left_camera_matrix, left_distortion, obj_points, left_img_points, img_size =\
        cameraCalibration('figure/example/left/*.jpg', 'left')
    right_camera_matrix, right_distortion, obj_points, right_img_points, img_size =\
        cameraCalibration('figure/example/right/*.jpg', 'right')
    # stereo calibration
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F =\
        cv2.stereoCalibrate(obj_points, left_img_points, right_img_points, left_camera_matrix, left_distortion,
                            right_camera_matrix, right_distortion, img_size)
    # read parameters from parameters.yml
    with open('../config/parameters.yaml', 'r') as f:
        yaml_obj = yaml.load(f, Loader=yaml.Loader)
        f.close()
    # write parameters to parameters.yml
    with open('../config/parameters.yaml', 'w') as f:
        yaml_obj['LeftCameraMatrix'] = cameraMatrix1.tolist()
        yaml_obj['LeftDistortion'] = distCoeffs1.tolist()
        yaml_obj['RightCameraMatrix'] = cameraMatrix2.tolist()
        yaml_obj['RightDistortion'] = distCoeffs2.tolist()
        yaml_obj['RotationMatrix'] = R.tolist()
        yaml_obj['TranslationMatrix'] = T.tolist()
        yaml_obj['EssentialMatrix'] = E.tolist()
        yaml_obj['FundamentalMatrix'] = F.tolist()
        yaml.dump(yaml_obj, f, default_flow_style=False, allow_unicode=True)
        f.close()
