import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

class CameraCalibration:
    """Camera calibration class, this class takes as input a folder with images and a folder with the corresponding Base2endeffector transforms
    and outputs the intrinsic matrix in a .npz file. It also performs hand-eye calibration and saves those results in a .npz file.
    The images with the corner detection are saved in a folder called 'DetectedCorners'

    This class has 4 optional parameters:
    pattern_size: the number of corners in the chessboard pattern, default is (4,7)
    square_size: the size of the squares in the chessboard pattern, default is 33/1000
    ShowProjectError: if True, it will show the reprojection error for each image in a bar plot, default is False
    ShowCorners: if True, it will show the chessboard corners for each image, default is False

    """
    def __init__(self, image_folder, Transforms_folder, pattern_size=(4, 7), square_size=33/1000, ShowProjectError=False, ShowCorners=False):

        #Initiate parameters
        self.pattern_size = pattern_size
        self.square_size = square_size

        #load images and joint positions
        self.image_files = sorted(glob.glob(f'{image_folder}/*.png'))
        self.transform_files = sorted(glob.glob(f'{Transforms_folder}/*.npz'))
        self.images = [cv2.imread(f) for f in self.image_files]
        self.images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in self.images]
        self.All_T_base2EE_list = [np.load(f)['arr_0'] for f in self.transform_files]

        #find chessboard corners and index of images with chessboard corners
        self.chessboard_corners, self.IndexWithImg = self.find_chessboard_corners(self.images, self.pattern_size, ShowCorners=ShowCorners)
        self.intrinsic_matrix = self.calculate_intrinsics(self.chessboard_corners, self.IndexWithImg,
                                                           self.pattern_size, self.square_size,
                                                           self.images[0].shape[:2], ShowProjectError = ShowProjectError)

        #Remove transforms were corners weren't detected
        self.T_base2EE_list = [self.All_T_base2EE_list[i] for i in self.IndexWithImg]

        #save intrinsic matrix
        np.savez("IntrinsicMatrix.npz", self.intrinsic_matrix)
        #Calculate camera extrinsics
        self.RTarget2Cam, self.TTarget2Cam = self.compute_camera_poses(self.chessboard_corners,
                                                                       self.pattern_size, self.square_size,
                                                                       self.intrinsic_matrix)

        #Convert to homogeneous transformation matrix
        self.T_target2cam = [np.concatenate((R, T), axis=1) for R, T in zip(self.RTarget2Cam, self.TTarget2Cam)]
        for i in range(len(self.T_target2cam)):
            self.T_target2cam[i] = np.concatenate((self.T_target2cam[i], np.array([[0, 0, 0, 1]])), axis=0)

        #Calculate T_cam2target
        self.T_cam2target = [np.linalg.inv(T) for T in self.T_target2cam]
        self.R_cam2target = [T[:3, :3] for T in self.T_cam2target]
        self.R_vec_cam2target = [cv2.Rodrigues(R)[0] for R in self.R_cam2target]
        self.T_cam2target = [T[:3, 3] for T in self.T_cam2target]   #4x4 transformation matrix

        #Calculate T_Base2EE

        self.TEE2Base = [np.linalg.inv(T) for T in self.T_base2EE_list]
        self.REE2Base = [T[:3, :3] for T in self.TEE2Base]
        self.R_vecEE2Base = [cv2.Rodrigues(R)[0] for R in self.REE2Base]
        self.tEE2Base = [T[:3, 3] for T in self.TEE2Base]

        #Create folder to save final transforms
        if not os.path.exists("FinalTransforms"):
            os.mkdir("FinalTransforms")
        #solve hand-eye calibration
        for i in range(0, 5):
            print("Method:", i)
            self.R_cam2gripper, self.t_cam2gripper = cv2.calibrateHandEye(
                self.R_cam2target,
                self.T_cam2target,
                self.R_vecEE2Base,
                self.tEE2Base,
                method=i
            )

            #print and save each results as .npz file
            print("The results for method", i, "are:")
            print("R_cam2gripper:", self.R_cam2gripper)
            print("t_cam2gripper:", self.t_cam2gripper)
            #Create 4x4 transfromation matrix
            self.T_cam2gripper = np.concatenate((self.R_cam2gripper, self.t_cam2gripper), axis=1)
            self.T_cam2gripper = np.concatenate((self.T_cam2gripper, np.array([[0, 0, 0, 1]])), axis=0)
            #Save results in folder FinalTransforms
            np.savez(f"FinalTransforms/T_cam2gripper_Method_{i}.npz", self.T_cam2gripper)
            #Save the inverse transfrom too
            self.T_gripper2cam = np.linalg.inv(self.T_cam2gripper)
            np.savez(f"FinalTransforms/T_gripper2cam_Method_{i}.npz", self.T_gripper2cam)

        #solve hand-eye calibration using calibrateRobotWorldHandEye
        for i in range(0,2):
            self.R_base2world, self.t_base2world, self.R_gripper2cam, self.t_gripper2cam= cv2.calibrateRobotWorldHandEye( self.RTarget2Cam, self.TTarget2Cam, self.REE2Base, self.tEE2Base, method=i)
            #print and save each results as .npz file
            print("The results for method using calibrateRobotWorldHandEye", i+4, "are:")
            print("R_cam2gripper:", self.R_gripper2cam)
            print("t_cam2gripper:", self.t_gripper2cam)
            #Create 4x4 transfromation matrix T_gripper2cam
            self.T_gripper2cam = np.concatenate((self.R_gripper2cam, self.t_gripper2cam), axis=1)
            self.T_gripper2cam = np.concatenate((self.T_gripper2cam, np.array([[0, 0, 0, 1]])), axis=0)
            #Save results in folder FinalTransforms
            np.savez(f"FinalTransforms/T_gripper2cam_Method_{i+4}.npz", self.T_gripper2cam)
            #save inverse too
            self.T_cam2gripper = np.linalg.inv(self.T_gripper2cam)
            np.savez(f"FinalTransforms/T_cam2gripper_Method_{i+4}.npz", self.T_cam2gripper)

    def find_chessboard_corners(self, images, pattern_size, ShowCorners=False):
        """Finds the chessboard patterns and, if ShowImage is True, shows the images with the corners"""
        chessboard_corners = []
        IndexWithImg = []
        i = 0
        print("Finding corners...")
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)
            if ret:
                chessboard_corners.append(corners)

                cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                if ShowCorners:
                    #plot image using maplotlib. The title should "Detected corner in image: " + i
                    plt.imshow(image)
                    plt.title("Detected corner in image: " + str(i))
                    plt.show()
                #Save the image in a folder Named "DetectedCorners"
                #make folder
                if not os.path.exists("DetectedCorners"):
                    os.makedirs("DetectedCorners")

                cv2.imwrite("DetectedCorners/DetectedCorners" + str(i) + ".png", image)

                IndexWithImg.append(i)
                i = i + 1
            else:
                print("No chessboard found in image: ", i)
                i = i + 1
        return chessboard_corners, IndexWithImg

    def compute_camera_poses(self, chessboard_corners, pattern_size, square_size, intrinsic_matrix, Testing=False):
        """Takes the chessboard corners and computes the camera poses"""
        # Create the object points.Object points are points in the real world that we want to find the pose of.
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Estimate the pose of the chessboard corners
        RTarget2Cam = []
        TTarget2Cam = []
        i = 1
        for corners in chessboard_corners:
            _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, None)
            # rvec is the rotation vector, tvec is the translation vector
            if Testing == True:
                print("Current iteration: ", i, " out of ", len(chessboard_corners[0]), " iterations.")

                # Convert the rotation vector to a rotation matrix
                print("rvec: ", rvec)
                print("rvec[0]: ", rvec[0])
                print("rvec[1]: ", rvec[1])
                print("rvec[2]: ", rvec[2])
                print("--------------------")
            i = 1 + i
            R, _ = cv2.Rodrigues(rvec)  # R is the rotation matrix from the target frame to the camera frame
            RTarget2Cam.append(R)
            TTarget2Cam.append(tvec)

        return RTarget2Cam, TTarget2Cam

    def calculate_intrinsics(self, chessboard_corners, IndexWithImg, pattern_size, square_size, ImgSize, ShowProjectError=False):
        """Calculates the intrinc camera parameters fx, fy, cx, cy from the images"""
        # Find the corners of the chessboard in the image
        imgpoints = chessboard_corners
        # Find the corners of the chessboard in the real world
        objpoints = []
        for i in range(len(IndexWithImg)):
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
            objpoints.append(objp)
        # Find the intrinsic matrix
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ImgSize, None, None)

        print("The projection error from the calibration is: ",
              self.calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist,ShowProjectError))
        return mtx

    def calculate_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist,ShowPlot=False):
        """Calculates the reprojection error of the camera for each image. The output is the mean reprojection error
        If ShowPlot is True, it will show the reprojection error for each image in a bar graph"""

        total_error = 0
        num_points = 0
        errors = []

        for i in range(len(objpoints)):
            imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints_projected = imgpoints_projected.reshape(-1, 1, 2)
            error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
            errors.append(error)
            total_error += error
            num_points += 1

        mean_error = total_error / num_points

        if ShowPlot:
            # Plotting the bar graph
            fig, ax = plt.subplots()
            img_indices = range(1, len(errors) + 1)
            ax.bar(img_indices, errors)
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Reprojection Error')
            ax.set_title('Reprojection Error for Each Image')
            plt.show()
            print(errors)

            #Save the bar plot as a .png
            fig.savefig('ReprojectionError.png')

        return mean_error

if __name__== "__main__":
    # Create an instance of the class
    image_folder = "Cal2/RGBImgs/"
    PoseFolder = "Cal2/T_base2ee/"
    calib = CameraCalibration(image_folder, PoseFolder,ShowProjectError=True)
