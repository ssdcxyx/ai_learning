> #### Question 6:
Programming Question 6 answer in CameraCalibration.py.  
I have found the camera intrinsic and extrinsic parameters from left01.jpg to left13.jpg. I have used cv2.calibrateCamera().
***
> #### Question 7:
Programming Question 7 answer in CameraCalibration.py.  
I have finished it in two ways in CameraCalibration.py. I have used cv2.undistort() or cv2.initUndistortRectifyMap().I have got two figures that was calibrated image left14.jpg in two ways in ../figure/Test.
left14.jpg is original image, calibrate_left_14_1.jpg in cv2.undistort(), calibrate_left_14_2.jpg in cv2.initUndistortRectifyMap().
***
> #### Question 12:
Programming Question 12 answer in StereoCalibration.py.  
I have used method cv2.calibrateCamera() to get left camera parameters(Left Camera Matrix, Left Distortion) and right camera parameters(Right Camera Matrix, Right Distortion). I have used method cv2.stereoCalibrate()
to get other parameters(Rotation Matrix, Translation Matrix, Essential Matrix, Fundamental Matrix). I have stored the parameters of the two cameras in the file parameters.yml.

