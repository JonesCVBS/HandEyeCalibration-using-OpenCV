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

    Convention: the files in Transforms_folder must hold the gripper's pose expressed
    in the robot base frame (i.e. plain forward kinematics from the joint angles -
    this is OpenCV's R_gripper2base/t_gripper2base). The legacy folder/file naming
    ("T_base2EE") suggests the opposite direction; the content is what matters, not
    the name.

    Requires OpenCV < 5.0 - cv2.calibrateHandEye and cv2.calibrateRobotWorldHandEye
    are not present in the OpenCV 5.x Python wheels.

    Of the five cv2.calibrateHandEye methods, method 1 (Park) is the recommended
    default. See https://github.com/opencv/opencv/issues/24871 - Tsai, Horaud,
    Andreff and Daniilidis are reported unreliable on some inputs; Tsai in
    particular is consistently the outlier on the bundled example dataset.
    """
    def __init__(self, image_folder, Transforms_folder, pattern_size=(4, 7), square_size=33/1000, ShowProjectError=False, ShowCorners=False):

        #Initiate parameters
        self.pattern_size = pattern_size
        self.square_size = square_size

        #load images and joint positions
        self.image_files = sorted(glob.glob(f'{image_folder}/*.png'))
        self.transform_files = sorted(glob.glob(f'{Transforms_folder}/*.npz'))
        self.images = [cv2.imread(f) for f in self.image_files]
        self.All_T_base2EE_list = [np.load(f)['arr_0'] for f in self.transform_files]

        #find chessboard corners and index of images with chessboard corners
        self.chessboard_corners, self.IndexWithImg = self.find_chessboard_corners(self.images, self.pattern_size, ShowCorners=ShowCorners)
        self.intrinsic_matrix, self.dist_coeffs = self.calculate_intrinsics(self.chessboard_corners, self.IndexWithImg,
                                                           self.pattern_size, self.square_size,
                                                           self.images[0].shape[1::-1], ShowProjectError = ShowProjectError)

        #Remove transforms were corners weren't detected
        self.T_base2EE_list = [self.All_T_base2EE_list[i] for i in self.IndexWithImg]

        #save intrinsic matrix
        np.savez("IntrinsicMatrix.npz", self.intrinsic_matrix)
        #Calculate camera extrinsics
        self.RTarget2Cam, self.TTarget2Cam = self.compute_camera_poses(self.chessboard_corners,
                                                                       self.pattern_size, self.square_size,
                                                                       self.intrinsic_matrix, self.dist_coeffs)

        #Gripper pose in base coordinates, straight from the loaded transforms (OpenCV's R_gripper2base/t_gripper2base)
        self.R_gripper2base = [T[:3, :3] for T in self.T_base2EE_list]
        self.t_gripper2base = [T[:3, 3] for T in self.T_base2EE_list]

        #Also keep the inverse (EE2Base, i.e. base pose in gripper coordinates) - calibrateRobotWorldHandEye needs it
        self.TEE2Base = [np.linalg.inv(T) for T in self.T_base2EE_list]
        self.REE2Base = [T[:3, :3] for T in self.TEE2Base]
        self.tEE2Base = [T[:3, 3] for T in self.TEE2Base]

        #Create folder to save final transforms
        if not os.path.exists("FinalTransforms"):
            os.mkdir("FinalTransforms")

        residual_report = []

        #solve hand-eye calibration
        for i in range(0, 5):
            print("Method:", i)
            self.R_cam2gripper, self.t_cam2gripper = cv2.calibrateHandEye(
                self.R_gripper2base,
                self.t_gripper2base,
                self.RTarget2Cam,
                self.TTarget2Cam,
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

            residual_report.append((f"Method {i}", self.T_cam2gripper))

        #solve hand-eye calibration using calibrateRobotWorldHandEye
        #Numbered 5, 6 (continuing on from the 5 calibrateHandEye methods, 0-4) -
        #using 4, 5 here would silently overwrite the Daniilidis (method 4) output above.
        for i in range(0,2):
            self.R_base2world, self.t_base2world, self.R_gripper2cam, self.t_gripper2cam= cv2.calibrateRobotWorldHandEye( self.RTarget2Cam, self.TTarget2Cam, self.REE2Base, self.tEE2Base, method=i)
            #print and save each results as .npz file
            print("The results for method using calibrateRobotWorldHandEye", i+5, "are:")
            print("R_gripper2cam:", self.R_gripper2cam)
            print("t_gripper2cam:", self.t_gripper2cam)
            #Create 4x4 transfromation matrix T_gripper2cam
            self.T_gripper2cam = np.concatenate((self.R_gripper2cam, self.t_gripper2cam), axis=1)
            self.T_gripper2cam = np.concatenate((self.T_gripper2cam, np.array([[0, 0, 0, 1]])), axis=0)
            #Save results in folder FinalTransforms
            np.savez(f"FinalTransforms/T_gripper2cam_Method_{i+5}.npz", self.T_gripper2cam)
            #save inverse too
            self.T_cam2gripper = np.linalg.inv(self.T_gripper2cam)
            np.savez(f"FinalTransforms/T_cam2gripper_Method_{i+5}.npz", self.T_cam2gripper)

            residual_report.append((f"Method {i+5}", self.T_cam2gripper))

        self.report_residuals(residual_report)

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
                #Refine the corner locations to sub-pixel accuracy
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                chessboard_corners.append(corners)

                cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                if ShowCorners:
                    #plot image using maplotlib. The title should "Detected corner in image: " + i
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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

    def compute_camera_poses(self, chessboard_corners, pattern_size, square_size, intrinsic_matrix, dist_coeffs, Testing=False):
        """Takes the chessboard corners and computes the camera poses"""
        # Create the object points.Object points are points in the real world that we want to find the pose of.
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Estimate the pose of the chessboard corners
        RTarget2Cam = []
        TTarget2Cam = []
        i = 1
        for corners in chessboard_corners:
            _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, dist_coeffs)
            # rvec is the rotation vector, tvec is the translation vector
            if Testing == True:
                print("Current iteration: ", i, " out of ", len(chessboard_corners), " iterations.")

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
        return mtx, dist

    def calculate_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist,ShowPlot=False):
        """Calculates the reprojection error of the camera for each image. The output is the mean RMS reprojection error
        If ShowPlot is True, it will show the reprojection error for each image in a bar graph"""

        total_sq_error = 0
        total_points = 0
        errors = []

        for i in range(len(objpoints)):
            imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints_projected = imgpoints_projected.reshape(-1, 1, 2)
            n = len(imgpoints_projected)
            sq_error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) ** 2
            errors.append((sq_error / n) ** 0.5)  # per-image RMS error
            total_sq_error += sq_error
            total_points += n

        mean_error = (total_sq_error / total_points) ** 0.5  # overall RMS error

        if ShowPlot:
            # Plotting the bar graph
            fig, ax = plt.subplots()
            img_indices = range(1, len(errors) + 1)
            ax.bar(img_indices, errors)
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Reprojection Error (RMS, px)')
            ax.set_title('Reprojection Error for Each Image')
            plt.show()
            print(errors)

            #Save the bar plot as a .png
            fig.savefig('ReprojectionError.png')

        return mean_error

    def report_residuals(self, transforms):
        """Ranks the T_cam2gripper results by how consistent they are across the input poses.
        For a correct X, T_gripper2base[k] . X . T_target2cam[k] should be nearly the same
        rigid transform for every pose k (it is the fixed pose of the calibration target in
        the robot base frame). Large spread across poses signals a bad method or a labelling
        error, without needing a robot to check against.
        Also written to FinalTransforms/residuals.txt."""
        lines = []
        header = f"{'Method':<12}{'rotation std (deg)':<22}{'translation std (mm)':<22}"
        lines.append(header)
        lines.append('-' * len(header))

        results = []
        for name, T_cam2gripper in transforms:
            target_in_base = []
            for Rg, tg, Rt, Tt in zip(self.R_gripper2base, self.t_gripper2base, self.RTarget2Cam, self.TTarget2Cam):
                T_gripper2base = np.concatenate((np.concatenate((Rg, np.asarray(tg).reshape(3, 1)), axis=1),
                                                  [[0, 0, 0, 1]]), axis=0)
                T_target2cam = np.concatenate((np.concatenate((Rt, np.asarray(Tt).reshape(3, 1)), axis=1),
                                                [[0, 0, 0, 1]]), axis=0)
                target_in_base.append(T_gripper2base @ T_cam2gripper @ T_target2cam)

            translations = np.array([T[:3, 3] for T in target_in_base])
            t_std_mm = float(np.linalg.norm(translations.std(axis=0)) * 1000)

            mean_R = np.mean([T[:3, :3] for T in target_in_base], axis=0)
            angles = []
            for T in target_in_base:
                R_diff = T[:3, :3] @ mean_R.T
                trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
                angles.append(np.degrees(np.arccos(trace)))
            r_std_deg = float(np.std(angles))

            results.append((name, r_std_deg, t_std_mm))
            lines.append(f"{name:<12}{r_std_deg:<22.4f}{t_std_mm:<22.4f}")

        results.sort(key=lambda r: r[2])
        lines.append("")
        lines.append(f"Most consistent across the input poses: {results[0][0]} "
                      f"(translation std {results[0][2]:.4f} mm)")

        report = "\n".join(lines)
        print(report)
        with open("FinalTransforms/residuals.txt", "w") as f:
            f.write(report + "\n")

if __name__== "__main__":
    # Create an instance of the class
    image_folder = "Cal2/RGBImgs/"
    PoseFolder = "Cal2/T_base2ee/"
    calib = CameraCalibration(image_folder, PoseFolder,ShowProjectError=True)
