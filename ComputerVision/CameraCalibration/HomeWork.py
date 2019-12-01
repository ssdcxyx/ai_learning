# -*- coding: utf-8 -*-
# @time       : 2019-10-12 14:11
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : HomeWork.py
# @description: 

import cv2
import numpy as np
import glob


obj_points = []
img_points = []
# 棋盘标定版尺寸
W, H = 8, 6


def main(path):
    images = glob.glob(path)
    origin = glob.glob('figure/myphoto/1.jpg')
    for fname in images:
        _, gray,  _, _ = findCorners(fname)
        orb(fname)
        flann(fname, origin[0])
    # 相机标定
    # 返回相机内参矩阵、畸变矩阵 、旋转向量、位移向量
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print('相机内参:')
    print(camera_matrix)
    print('畸变矩阵:')
    print(dist_coeffs)
    print('外参:')
    print('旋转矩阵:')
    print(rvecs)
    print('位移矩阵')
    print(tvecs)
    # 畸变矫正
    img_path = 'figure/origin/1.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y: y + h, x: x + w]
    dst_path = 'figure/origin/2.png'
    cv2.imwrite(dst_path, dst)

    _corners, gray, _img_points, _obj_points = findCorners(dst_path)
    img = cv2.imread(dst_path)

    # 绘制这四个选点
    cv2.circle(img, (_img_points[0][0], _img_points[0][1]), 9, (255, 0, 0), 3)
    cv2.circle(img, (_img_points[1][0], _img_points[1][1]), 9, (255, 0, 0), 3)
    cv2.circle(img, (_img_points[2][0], _img_points[2][1]), 9, (255, 0, 0), 3)
    cv2.circle(img, (_img_points[3][0], _img_points[3][1]), 9, (255, 0, 0), 3)
    cv2.imshow(img_path, img)
    # 等待输入
    cv2.waitKey()
    cv2.destroyWindow(img_path)

    # 透视变换
    _H = cv2.getPerspectiveTransform(_img_points, _obj_points)
    _img = cv2.warpPerspective(img, _H, (1280, 960),
                               cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS)
    cv2.imshow(img_path, _img)
    # 等待输入
    cv2.waitKey()
    cv2.destroyWindow(img_path)


# 检测角点
def findCorners(fname):
    # 棋盘标定版角点的位置
    obj_point = np.zeros((W * H, 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:W, 0:H].T.reshape(-1, 2)
    # 设置求亚像素角点的参数，最大循环次数30，最大误差0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 5, 0.001)
    # 读取返回BGR格式的图像
    img = cv2.imread(fname)
    # 将BGR格式转换为灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测棋盘角点
    ret, corners = cv2.findChessboardCorners(gray, (W, H), None)
    if ret:
        # 检测棋盘亚像素角点
        _corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(obj_point)
        img_points.append(corners)
        # 在图像中绘制角点
        _img = cv2.drawChessboardCorners(img, (W, H), _corners, ret)
        # 选定四个点用于透视变换
        _obj_points = np.float32([[0, 0], [(H - 1) * 100, 0], [0, (W - 1) * 100], [(H - 1) * 100, (W - 1) * 100]])
        _corners = _corners.reshape(H, W, 2)
        _img_points = np.float32([_corners[0][0], _corners[H - 1][0], _corners[0][W - 1], _corners[H - 1][W - 1]])

        cv2.imshow(fname, _img)
        cv2.waitKey()
        cv2.destroyWindow(fname)
    return _corners, gray, _img_points, _obj_points


# 局部特征提取
def orb(fname):
    img = cv2.imread(fname)
    # 初始化orb特征检测器
    orb = cv2.ORB_create()
    # 检测关键点 计算描述符
    key_points, description = orb.detectAndCompute(img, None)
    im_with_key_points = cv2.drawKeypoints(img, key_points, np.array([]), color=(255, 0, 0), flags=0)
    cv2.imshow(fname, im_with_key_points)
    # 等待输入
    cv2.waitKey()
    cv2.destroyWindow(fname)


# 特征匹配
def flann(fname, origin):
    img1 = cv2.imread(origin, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()

    key_points1, description1 = orb.detectAndCompute(img1, None)
    key_points2, description2 = orb.detectAndCompute(img2, None)

    img1 = cv2.drawKeypoints(img1, key_points1, np.array([]), color=(255, 0, 0), flags=0)
    img2 = cv2.drawKeypoints(img2, key_points2, np.array([]), color=(255, 0, 0), flags=0)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=12,
                        key_size=20,
                        multi_probe_level=2)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 特征匹配
    matches = flann.knnMatch(description1, description2, k=2)

    good = []
    # 比率测试 筛选匹配点
    for matche in matches:
        if len(matche) == 2:
            if matche[0].distance < 0.7 * matche[1].distance:
                good.append(matche[0])

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=None,
                       flags=2)

    img3 = cv2.drawMatches(img1, key_points1, img2, key_points2, good, None, **draw_params)
    cv2.imshow(fname, img3)
    # 等待输入
    cv2.waitKey()
    cv2.destroyWindow(fname)


if __name__ == "__main__":
    main('figure/myphoto/*.jpg')
    print()
