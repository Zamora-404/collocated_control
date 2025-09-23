import cv2
import numpy as np
import time
import os

class Cam_Tracker:
    def __init__(self, captureFrame = False):
        # ========================== Settings ========================
        CAM_IDS = [0, 2, 4] # Adjust as needed
        VID_PATH = ['output_cam0.avi','output_cam1.avi','output_cam2.avi',]
        IMAGE_SIZE = (1280, 720)
        self.captureFrame = captureFrame

        # ================ Marker Kinematics===============
        # Start Kinematics
        base_mark_height = 0.012 
        # sba base frame wrt to global, eventually repplace with three marker reference
        self.T_base_0 = np.eye(4)
        R_b0 = np.array([[0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0]])

        # p_b0 = - np.array([0.14205, -0.0026393, -0.0263967 - 0])
        p_b0 = - np.array([0.14186, 0.00327, -0.00993 - base_mark_height])
        self.T_base_0[0:3,0:3] = R_b0
        self.T_base_0[0:3,3] = p_b0
        
        # mass_h = 6.75   # steel mass height (6.6g)
        ef_h = -9     # end effector height (w/o ball bearing 6mm)
        scale = 1.3     # marker vector scale factor
        self.marker_ryb_ref_b = np.array([
            [ 0.0   * scale, -6.729 * scale,    ef_h],  # red
            [-5.827 * scale, -3.265 * scale,    ef_h],  # yellow
            [-5.827 * scale,  3.265 * scale,    ef_h],  # blue
            ]) / 1000                                   # convert to meters

        self.marker_pov_ref_b = np.array([
            [ 0.0   * scale, -6.729 * scale,    0],  # pink
            [-5.827 * scale, -3.265 * scale,    0],  # orange
            [-5.827 * scale,  3.265 * scale,    0],  # violet
            ]) / 1000   
        

        self.marker_ryb_ref_0 = self.marker_ryb_ref_b @ R_b0
        self.marker_pov_ref_0 = self.marker_pov_ref_b @ R_b0

        # print(marker_ref_b)
        # print(marker_ref_0)

        # ===================== Load camera params ========================
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.params = [self.load_params(f'{self.path}/cam{i+1}.npz') for i in range(3)]

        # ======================= Setup cameras =========================
        # caps = [cv2.VideoCapture(cam_id) for cam_id in VID_PATH] # Use to run tracker on videos
        self.caps = [cv2.VideoCapture(cam_id) for cam_id in CAM_IDS]
        self.outs = []
        for i, cap in enumerate(self.caps):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
            if self.captureFrame:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.outs.append( cv2.VideoWriter( 'output_cam'+str(i)+'.avi', 30, IMAGE_SIZE) )

        # ========================= Begin time and file recording =========================
        self.time_start = time.time()
        self.time_prev = self.time_start - 0.1

        # pose for SBA tip marker
        self.R_tip_b = np.eye(3)
        self.t_tip_b = np.array([0,0,0])

        # pose for wall marker
        self.R_wall_b = np.eye(3)
        self.t_wall_b = np.array([0,0,0])
        




    def load_params(self, filename):
        data = np.load(filename)
        return data['K'], data['D'], data['R'], data['T']

    def find_marker_center(self, frame, c, min_pixels=200):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # mask Color thresholding
        if c.lower() == 'y':  # Yellow
            lower = np.array([16, 40, 80])
            upper = np.array([56, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        elif c.lower() == 'b':  # Blue
            lower = np.array([100, 120, 80])
            upper = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        elif c.lower() == 'r':  # Red
            lower_red1 = np.array([0, 80, 80])
            upper_red1 = np.array([5, 255, 255])
            lower_red2 = np.array([176, 80, 80])
            upper_red2 = np.array([179, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif c.lower() == 'p': # Pink
            lower = np.array([159, 75, 160])
            upper = np.array([176, 200, 255])
            mask = cv2.inRange(hsv, lower, upper)
        elif c.lower() == 'o': # Orange
            lower = np.array([6, 88, 175])
            upper = np.array([18, 217, 255])
            mask = cv2.inRange(hsv, lower, upper)
        elif c.lower() == 'v': # Violet
            lower = np.array([132, 81, 36])
            upper = np.array([151, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        else:
            print("Invalid marker color string")

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
    
    def get_pts_2D(self, frames, colors):
        pts_2D_color = []
        
        # iterate through each color in colors string
        for c in colors:
            pt = []
            # find the color point in each frame and add to list
            for f in frames:
                pt.append( self.find_marker_center(f, c) )
            
            # add the list of points to master list 
            pts_2D_color.append(pt)

        return pts_2D_color
    
    def get_pts_3D(self, frames, pts2D_color):

        # check which cameras can view each marker by detecting None inside pts_color
        i_views = [[],[],[]]
        for j, pts_c in enumerate(pts2D_color):
            for i, pt in enumerate(pts_c):
                if pt is not None:
                    cv2.circle(frames[i], tuple(pt.astype(int)), 5, (255, 255, 0), -1)
                    i_views[j].append(i)

        # i_views = [0, 1]
        pts3d_base = np.zeros((3,3))
        pts3d_global = np.zeros((3,3))
        for j, pts_c in enumerate(pts2D_color):
            if len(i_views[j]) >= 2: # triangualte pt if seen by at least two cams
                params_seen, pts_seen = [self.params[i] for i in i_views[j] ], [pts_c[i] for i in i_views[j]]
                origins, directions = [], []
                for (K, D, R, T), pt in zip(params_seen, pts_seen):
                    o, d = self.backproject_pixel(K, R, T, pt)
                    origins.append(o)
                    directions.append(d)

                pt3d_global = self.triangulate_rays(origins, directions).ravel() 
                pts3d_global[j,:] = pt3d_global

                # transforms the color 3d pts to the base frame
                pt3d_base = self.T_base_0 @ np.append(pt3d_global, 1) # transform to base frame
                pts3d_base[j,:] = pt3d_base[0:3] # update all colored pts
                
        return pts3d_base, pts3d_global
    

        

    def update(self, rigid_coords = np.zeros((3, 4))):
        time_now = time.time()
        t_elapsed = time_now - self.time_start
        dt = time_now - self.time_prev
        self.time_prev = time_now
        
        # capture frames from cameras and check if successful
        frames = [cap.read()[1] for cap in self.caps]
        if any(f is None for f in frames):
            print("Frame capture failed")
            return
        
        # record capture to video file
        if self.captureFrame:
            for i, f in enumerate(frames):
                self.outs[i].write(f)
        
        # set markers using colors
        color_markers = ['ryb', 'pov']
        # color_markers = ['ryb']
        
        for cm in color_markers:
            if cm == 'ryb':
                marker_ref_b, marker_ref_0 = self.marker_ryb_ref_b, self.marker_ryb_ref_0
            elif cm == 'pov':
                marker_ref_b, marker_ref_0 = self.marker_pov_ref_b, self.marker_pov_ref_0

            # gets color points in 2D (pixel coordinates from camera frames)
            pts_2D = self.get_pts_2D(frames, cm)

            # gets color points in 3D (cartesian coordinates in base and global/camera1 frame)
            pts_3D_base, pts_3D_global = self.get_pts_3D(frames, pts_2D)
            

            
            # Find pose for marker if all colors are dected 
            if pts_3D_global is not None: 
                #  extract tip pose using 3 markers in base frame
                if cm == 'ryb':
                    self.R_tip_b, self.t_tip_b = self.estimate_pose(pts_3D_base, marker_ref_b)
                elif cm == 'pov':
                    self.R_wall_b, self.t_wall_b = self.estimate_pose(pts_3D_base, marker_ref_b)
                

                # extract tip pose using 3 markers in camera frame
                R_tip_0, t_tip_0 = self.estimate_pose(pts_3D_global, marker_ref_0)

                # project tip pose and backbone to camera frame 
                self.draw_pose(self.params, frames, R_tip_0, t_tip_0, rigid_coords)



                
        cv2.imshow(f"Camera 1", frames[0])
        # for i, f in enumerate(frames):
        #     cv2.imshow(f"Camera {i+1}", f)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

        return
    
    def close(self):
        for i, cap in enumerate(self.caps):
            cap.release()
            if self.captureFrame:
                self.outs[i].release()
        cv2.destroyAllWindows()

    def get_pose(self):
        return (self.t_tip_b, self.R_tip_b, self.t_wall_b, self.R_wall_b)


def main():
    np.set_printoptions(precision=5, suppress=True)
    tracker = Cam_Tracker()

    # try:
    while True:
        tracker.update()
        p1, p2, p3, p4 = tracker.get_pose()
        print(p1*1000, p2 )
        # print("tip position:", pose[0]*1000)
    # except:
    #     print("error, closing tracker")
    #     tracker.close()


if __name__ == "__main__":
    main()
