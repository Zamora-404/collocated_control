import cv2
import numpy as np
import time
import os

class Cam_Tracker:
    def __init__(self):
        # ========================== Settings ========================
        CAM_IDS = [0, 2, 4] # Adjust as needed
        VID_PATH = ['output_cam0.avi','output_cam1.avi','output_cam2.avi',]
        IMAGE_SIZE = (1280, 720)
        isRos = False

        # ================ Marker Kinematics===============
        # Start Kinematics
        base_mark_height = 0.012 # meters
        # sba base frame wrt to global, eventually repplace with three marker reference
        self.T_base_0 = np.eye(4)
        R_b0 = np.array([[0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0]])
        # R_bc = np.array([[ 0.9899,  0.0849, -0.1136],
        #                 [-0.0778,  0.9948,  0.0657],
        #                 [ 0.1186, -0.0562,  0.9914]])

        # p_b0 = - np.array([0.14205, -0.0026393, -0.0263967 - 0])
        p_b0 = - np.array([0.14186, 0.00327, -0.00993 - base_mark_height])
        self.T_base_0[0:3,0:3] = R_b0
        self.T_base_0[0:3,3] = p_b0
        
        # mass_h = 6.75   # steel mass height (6.6g)
        ef_h = -6     # end effector height 
        # ef_h = 0
        scale = 1.3     # marker vector scale factor
        self.marker_ref_b = np.array([
            [ 0.0   * scale, -6.729 * scale,    ef_h],  # red
            [-5.827 * scale, -3.265 * scale,    ef_h],  # yellow
            [-5.827 * scale,  3.265 * scale,    ef_h],  # blue
            ]) / 1000                                   # convert to meters

        self.marker_ref_0 = self.marker_ref_b @ R_b0

        # print(marker_ref_b)
        # print(marker_ref_0)

        # ===================== Load camera params ========================
        self.params = [self.load_params(f'cam{i+1}.npz') for i in range(3)]

        # ======================= Setup cameras =========================
        # caps = [cv2.VideoCapture(cam_id) for cam_id in VID_PATH] # Use to run tracker on videos
        self.caps = [cv2.VideoCapture(cam_id) for cam_id in CAM_IDS]
        self.outs = []
        for i, cap in enumerate(self.caps):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # self.outs.append( cv2.VideoWriter( 'output_cam'+str(i)+'.avi', fourcc, 30, IMAGE_SIZE) ) # FIX by adding toggle for recording

        # ========================= Begin time and file recording =========================
        self.time_start = time.time()
        self.time_prev = self.time_start - 0.1


        self.R_tip_b = np.eye(3)
        self.t_tip_b = np.array([0,0,0])


    def load_params(self, filename):
        data = np.load(filename)
        return data['K'], data['D'], data['R'], data['T']

    def find_marker_center(self, frame, n=2, min_pixels=200):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Color thresholding
        if n == 0:  # Yellow
            lower = np.array([13, 40, 80])
            upper = np.array([56, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        elif n == 1:  # Blue
            lower = np.array([100, 120, 80])
            upper = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        else:  # Red (n==2)
            lower_red1 = np.array([0, 80, 80])
            upper_red1 = np.array([5, 255, 255])
            lower_red2 = np.array([170, 80, 80])
            upper_red2 = np.array([179, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        # Count matching pixels
        pixel_count = cv2.countNonZero(mask)

        # Debug visualization
        # cv2.imshow("mask debug", mask)
        # cv2.putText(frame, f"Pixels: {pixel_count}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # # Skip if too few pixels
        if pixel_count < min_pixels:
            # print(f"Warning: Only {pixel_count} pixels detected (minimum {min_pixels} required)")
            return None
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return np.array([cx, cy])
        return None


    def backproject_pixel(self, K, R, T, pt):
        uv1 = np.array([pt[0], pt[1], 1.0])
        K_inv = np.linalg.inv(K)
        ray_cam = K_inv @ uv1
        d = R.T @ ray_cam
        d /= np.linalg.norm(d)
        origin = -R.T @ T
        return origin, d

    def triangulate_rays(self, origins, directions):
        A, b = [], []
        for o, d in zip(origins, directions):
            I = np.eye(3)
            A.append(I - np.outer(d, d))
            b.append((I - np.outer(d, d)) @ o)
        A = np.sum(A, axis=0)
        b = np.sum(b, axis=0)
        return np.linalg.lstsq(A, b, rcond=None)[0]


    def estimate_pose(self, marker_world, marker_ref):
        """
        Estimate rigid transformation from marker_ref to marker_world.
        Returns rotation matrix R and translation vector t.
        """
        assert marker_world.shape == marker_ref.shape == (3, 3)

        # Center the points
        centroid_world = marker_world.mean(axis=0)
        centroid_ref = marker_ref.mean(axis=0)

        P = marker_ref - centroid_ref
        Q = marker_world - centroid_world

        # Compute optimal rotation using SVD
        H = P.T @ Q
        U, _, Vt = np.linalg.svd(H)
        R_opt = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R_opt) < 0:
            Vt[2, :] *= -1
            R_opt = Vt.T @ U.T

        t_opt = centroid_world - R_opt @ centroid_ref

        return R_opt, t_opt

    def draw_pose(self, params, frames, R_tip_0, t_tip_0, backbone):
        '''Draw pose on frames using camera parameters and rot/ trans'''

        # print(backbone)

        R_b0 = self.T_base_0[0:3,0:3]
        p_b0 = self.T_base_0[0:3,3]
        for (K, D, R, T), frame in zip(params, frames):
            imgpts, _ = cv2.projectPoints(t_tip_0, R, T, K, D)
            imgpts_int = tuple(map(int, imgpts.ravel())) # flatten to 1D array, then convert to int tupple

            # Define length and axes in the tip frame
            axis_length = 0.005  # 5 mm
            origin = np.array([[0, 0, 0]], dtype=np.float32)
            axes = np.array([
                [axis_length, 0, 0],   # X axis
                [0, axis_length, 0],   # Y axis
                [0, 0, axis_length]    # Z axis
            ], dtype=np.float32)

            # Transform to global frame using T_tip_0 or T_tip_base
            R_tip = R_tip_0 @ R_b0.T
            t_tip = t_tip_0 
            axes_world = (R_tip @ axes.T).T + t_tip  # shape (3, 3)
            points_to_project = np.vstack((t_tip, axes_world))  # (4, 3)

            # Convert to meters, then project to world frame (camera frame)
            backbone_world = (backbone.T/1000 - p_b0.T) @ R_b0

            # Project to image and draw
            imgpts, _ = cv2.projectPoints(points_to_project, R, T, K, D)
            imgpts = imgpts.reshape(-1, 2).astype(int)
            origin_2d = tuple(imgpts[0])

            # project backbone in meters
            img_backbone, _ = cv2.projectPoints(backbone_world, R, T, K, D) 
            img_backbone = img_backbone.reshape(-1, 2).astype(int)

            # print(imgpts, img_backbone)
            # print(img_backbone, np.shape(img_backbone))
            # print(imgpts, np.shape(imgpts))


            cv2.circle(frame, imgpts_int, 8, (0, 255, 255), -1)             # Tip position in yellow
            cv2.line(frame, origin_2d, tuple(imgpts[1]), (0, 0, 255), 2)    # X axis in red
            cv2.line(frame, origin_2d, tuple(imgpts[2]), (0, 255, 0), 2)    # Y axis in green
            cv2.line(frame, origin_2d, tuple(imgpts[3]), (255, 0, 0), 2)    # Z axis in blue

            cv2.polylines(frame, [img_backbone], False, (255, 255, 0), 5)   # Backbone in cyan
        return

    def update(self, rigid_coords = np.zeros((3, 4))):
        time_now = time.time()
        t_elapsed = time_now - self.time_start
        dt = time_now - self.time_prev
        self.time_prev = time_now
        
        frames = [cap.read()[1] for cap in self.caps]
        if any(f is None for f in frames):
            print("Frame capture failed")
            return
        
        pts_r = [self.find_marker_center(f, 2) for f in frames]
        pts_y = [self.find_marker_center(f, 0) for f in frames]
        pts_b = [self.find_marker_center(f, 1) for f in frames]
        pts = [pts_r, pts_y, pts_b]

        # for i, f in enumerate(frames):
        #     self.outs[i].write(f)

        i_views = [[],[],[]]
        for j, pts_c in enumerate(pts):
            for i, pt in enumerate(pts_c):
                if pt is not None:
                    cv2.circle(frames[i], tuple(pt.astype(int)), 5, (255, 255, 0), -1)
                    i_views[j].append(i)

        # i_views = [0, 1]
        pts3d_base = np.zeros((3,3))
        pts3d_global = np.zeros((3,3))
        for j, pts_c in enumerate(pts):
            if len(i_views[j]) >= 2: # triangualte pt if seen by at least two cams
                params_seen, pts_seen = [self.params[i] for i in i_views[j] ], [pts_c[i] for i in i_views[j]]
                origins, directions = [], []
                for (K, D, R, T), pt in zip(params_seen, pts_seen):
                    o, d = self.backproject_pixel(K, R, T, pt)
                    origins.append(o)
                    directions.append(d)

                pt3d_global = self.triangulate_rays(origins, directions).ravel() 
                pts3d_global[j,:] = pt3d_global

                # the following transforms the color 3d pts to the base frame
                pt3d_base = self.T_base_0 @ np.append(pt3d_global, 1) # transform to base frame
                pts3d_base[j,:] = pt3d_base[0:3] # update all colored pts
                

        T_tip_0 = np.eye(4)  
        if pts3d_global is not None: 
            #  extract tip pose using 3 markers in base frame
            self.R_tip_b, self.t_tip_b = self.estimate_pose(pts3d_base, self.marker_ref_b)
            # print(R_tip_b)
            # print(T_tip_base[0:3,0:3])

            # print('Tip position (base):', t_tip_b*1000)
            # print('Tip position (global):', T_tip_base[0:3,3]*1000)


            # extract tip pose using 3 markers in camera frame
            R_tip_0, t_tip_0 = self.estimate_pose(pts3d_global, self.marker_ref_0)

            # project tip pose and backbone to camera frame 
            self.draw_pose(self.params, frames, R_tip_0, t_tip_0, rigid_coords)
            # print(self.R_tip_b)
            # print(self.t_tip_b)

            # print(R_tip_0)
            # print(t_tip_0)

            # project base pose and backbone to camera frame
            # self.draw_pose(self.params, frames, np.eye(3), np.array([0.0009,0.0178,.1421]), rigid_coords)
            # print(np.eye(3))
            # print(np.array([0,0,0]))




                
        # cv2.imshow(f"Camera 1", frames[0])
        for i, f in enumerate(frames):
            cv2.imshow(f"Camera {i+1}", f)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

        return
    
    def close(self):
        for i, cap in enumerate(self.caps):
            cap.release()
            # self.outs[i].release()
        cv2.destroyAllWindows()

    def get_pose(self):
        return (self.t_tip_b, self.R_tip_b )


def main():
    np.set_printoptions(precision=5, suppress=True)
    tracker = Cam_Tracker()

    # try:
    while True:
        tracker.update()
        pose = tracker.get_pose()
        print("tip position:", pose[0]*1000)
    # except:
    #     print("error, closing tracker")
    #     tracker.close()


if __name__ == "__main__":
    main()
