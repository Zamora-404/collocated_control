import cv2
import numpy as np
import os

# === SETTINGS ===
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.005 # meters
NUM_IMAGES = 15
IMAGE_SIZE = (1280, 720)
CAM_IDS = [0, 2, 4] # Adjust to your actual camera IDs

# === OBJECT POINTS (same for all images) ===
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# === Storage ===
objpoints = []
imgpoints = [[] for _ in CAM_IDS]

# === Open cameras ===
caps = [cv2.VideoCapture(cam_id) for cam_id in CAM_IDS]
for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

print("Press 'c' to capture checkerboard. Press 'q' when done.")

captured = 0
while captured < NUM_IMAGES:
    frames = [cap.read()[1] for cap in caps]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    found_all, corners_all = [], []

    for i, gray in enumerate(grays):
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        found_all.append(found)
        corners_all.append(corners)
        disp = frames[i].copy()
        if found:
            cv2.drawChessboardCorners(disp, CHECKERBOARD, corners, found)
        cv2.imshow(f'Camera {i+1}', disp)

    key = cv2.waitKey(1)
    if key == ord('c') and all(found_all):
        print(f"Captured {captured + 1}")
        objpoints.append(objp)
        for i in range(len(CAM_IDS)):
            refined = cv2.cornerSubPix(grays[i], corners_all[i], (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints[i].append(refined)
        captured += 1

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
for cap in caps:
    cap.release()

# === Calibrate each camera ===
camera_params = []
for i in range(len(CAM_IDS)):
    print(f"Calibrating Camera {i+1}")
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints[i], IMAGE_SIZE, None, None)
    camera_params.append({'K': K, 'D': D, 'rvecs': rvecs, 'tvecs': tvecs})
    np.savez(f"cam{i+1}.npz", K=K, D=D)

# === Extrinsic calibration: cam1 as world origin ===
ref_idx = 0
K_ref = camera_params[ref_idx]['K']
D_ref = camera_params[ref_idx]['D']
rvecs_ref = camera_params[ref_idx]['rvecs']
tvecs_ref = camera_params[ref_idx]['tvecs']

for i in range(len(CAM_IDS)):
    if i == ref_idx:
        R = np.eye(3)
        T = np.zeros((3, 1))
    else:
        # Estimate relative pose by comparing average poses across all frames
        R_all, T_all = [], []
        for rvec1, tvec1, rveci, tveci in zip(rvecs_ref, tvecs_ref, camera_params[i]['rvecs'], camera_params[i]['tvecs']):
            R1, _ = cv2.Rodrigues(rvec1)
            Ri, _ = cv2.Rodrigues(rveci)
            R_rel = Ri @ R1.T
            T_rel = tveci - R_rel @ tvec1
            R_all.append(R_rel)
            T_all.append(T_rel)
        R = np.mean(R_all, axis=0)
        T = np.mean(T_all, axis=0)
    np.savez(f"cam{i+1}.npz", K=camera_params[i]['K'], D=camera_params[i]['D'], R=R, T=T)

print("Calibration complete. Saved cam1.npz, cam2.npz, cam3.npz")




# import cv2
# import numpy as np
# import glob
# import os

# # === Calibration Settings ===
# CHECKERBOARD = (9, 6)  # inner corners
# SQUARE_SIZE = 0.005  # in meters
# NUM_IMAGES = 15

# # Create object points for checkerboard (0,0,0), (1,0,0), ..., (8,5,0)
# objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# objp *= SQUARE_SIZE

# # Arrays to store object points and image points
# objpoints = []
# imgpointsL = []
# imgpointsR = []

# capL = cv2.VideoCapture(0, cv2.CAP_V4L2)
# capR = cv2.VideoCapture(2, cv2.CAP_V4L2)
# capT = cv2.VideoCapture(4, cv2.CAP_V4L2)

# # capL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# # capR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# # capL.set(cv2.CAP_PROP_FPS, 15)
# # capR.set(cv2.CAP_PROP_FPS, 15)

# width, height = 640, 480
# capL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# capL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# capR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# capR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# capT.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# capT.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# print("Press 'c' to capture image pairs. 'q' to finish.")
# captured = 0

# while captured < NUM_IMAGES:
#     retL, frameL = capL.read()
#     retR, frameR = capR.read()
#     retT, frameT = capT.read()

#     if not retL or not retR or not retT:
#             break

#     grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
#     grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
#     grayT = cv2.cvtColor(frameT, cv2.COLOR_BGR2GRAY)

#     foundL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD,
#                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
#     foundR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD,
#                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
#     foundT, cornersT = cv2.findChessboardCorners(grayT, CHECKERBOARD,
#                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

#     displayL = frameL.copy()
#     displayR = frameR.copy()
#     displayT = frameT.copy()
    

#     if foundL:
#         cv2.drawChessboardCorners(displayL, CHECKERBOARD, cornersL, foundL)
    
#     if foundR:
#         cv2.drawChessboardCorners(displayR, CHECKERBOARD, cornersR, foundR)
    
#     if foundT:
#         cv2.drawChessboardCorners(displayT, CHECKERBOARD, cornersT, foundT)

#     cv2.putText(displayL, f'Captured: {captured}/{NUM_IMAGES}', (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     cv2.putText(displayR, f'Captured: {captured}/{NUM_IMAGES}', (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     cv2.putText(displayT, f'Captured: {captured}/{NUM_IMAGES}', (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
#     cv2.imshow("CalibrationL", displayL)
#     cv2.imshow("CalibrationR", displayR)
#     cv2.imshow("CalibrationT", displayT)

#     key = cv2.waitKey(1)
#     if key == ord('c'):
#         retL_c, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
#         retR_c, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
#         retT_c, cornersT = cv2.findChessboardCorners(grayT, CHECKERBOARD, None)

#         if retL_c and retR_c and retR_c:
#             objpoints.append(objp)
#             cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1),
#                                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
#             cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1),
#                                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
#             imgpointsL.append(cornersL)
#             imgpointsR.append(cornersR)
#             captured += 1
#             print(f"Captured {captured}/{NUM_IMAGES}")
#         else:
#             print("Chessboard not found in both images.")

#     elif key == ord('q'):
#         break

# cv2.destroyAllWindows()
# capL.release()
# capR.release()

# # === Calibrate each camera ===
# retL, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
# retR, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# # === Stereo calibration ===
# flags = cv2.CALIB_FIX_INTRINSIC
# criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
#     objpoints, imgpointsL, imgpointsR,
#     K1, D1, K2, D2,
#     grayL.shape[::-1],
#     criteria=criteria_stereo,
#     flags=flags
# )

# # === Stereo rectification ===
# R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
#     K1, D1, K2, D2, grayL.shape[::-1],
#     R, T, alpha=0
# )

# # === Save calibration ===
# np.savez("stereo_calib.npz",
#          K1=K1, D1=D1, K2=K2, D2=D2,
#          R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

# print("Stereo calibration saved as stereo_calib.npz")
