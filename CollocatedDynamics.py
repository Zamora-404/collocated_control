import numpy as np
import matplotlib.pyplot as plt
import json


class RobotKinematics():
    def __init__(self, config_file, q_0=np.zeros(6), isFortimedix=False, isEmbeddedKinematics=False):
        
        # check robot architecture
        self.isEmbeddedKinematics = isEmbeddedKinematics

        # Define the robot's kinematic parameters from json config
        self.LINK_LENGTHS = None
        self.LINK_DIAMETERS = None
        self.ARTICULATION_LIMITS = None
        self.JOINT_RANGES = None
        self.JOINT_LABELS = ["outer_rot", "outer_advance", "outer_flex", "inner_rot",
                            "inner_advance", "inner_flex", "depth_advance", "deploy_advance"]
        self.q_aug_labels = ["r", "p",      "r", "p", "r",      "r", "p",       "r", "p", "r",     "p"] # augmented joint labels for better interpretability of the jacobian and kinematics calculations, update based on the number of catheters and joints in the system
        self.device_config = self.load_config(config_file)
        self.joint_weights = np.array([1, 1, 1, 1, 1, 1]) # uniform weights for testing

        # define robot's initial configuration (home position)
        self.q_0 = q_0
        self.q = q_0
        self.n_dof = len(q_0)
        self.q_aug = self.compute_q_aug(self.q_0)
        self.n_aug = len(self.q_aug)
        self.a_aug = self.compute_a_aug()


        self.T = None # end effector transformation matrix
        self.J = None # end effector jacobian matrix
        self.T_back = None # vector of catheter joint positions for visualization and debugging
        self.compute_jacobian(self.q_0) # compute initial jacobian and transformation matrix for home position



    def load_config(self, config_file): 
        #TODO: Request a diagram of configuration parameters and how they relate 
        # to the robot's geometry and kinematics. This will help ensure that we
        #  are interpreting the parameters correctly and using them appropriately 
        # in our kinematics calculations.

        with open(config_file, 'r') as f:
            device_config = json.load(f)

             # print(device_config)
        component_lengths = device_config["component_lengths"]
        component_diameters = device_config["component_diameters"]
        articulation_limits = device_config["articulation_limits"]
        
        
        # link lengths
        inner_flex_len = float(component_lengths["INNER_FLEX"])
        inner_flex_cap = float(component_lengths["INNER_FLEX_CAP"])
        outer_flex_len = float(component_lengths["OUTER_FLEX"])
        outer_flex_cap = float(component_lengths["OUTER_FLEX_CAP"])
        inner_tetris_len = float(component_lengths["INNER_TETRIS"])
        capsule_len = float(component_lengths["CAPSULE"])
        guide_shaft =  float(component_lengths["GUIDE_SHAFT"])

        # link diameters
        OUTER_DIAMETER_MM = component_diameters["OUTER_FLEX"]
        INNER_DIAMETER_MM = component_diameters["INNER_FLEX"]
        CAPSULE_DIAMETER_MM = component_diameters["CAPSULE"]

        #hard joint limits constants
        CAPSULE_SLIDE_LIMIT_MM = articulation_limits["CAPSULE_SLIDE"]
        DEPTH_SLIDE_LIMIT_MM = articulation_limits["DEPTH_SLIDE"]
        OUTER_FLEX_LIMIT_DEG = articulation_limits["OUTER_FLEX"] 
        INNER_FLEX_LIMIT_DEG = articulation_limits["INNER_FLEX"]
        INNER_ADVANCE_LIMIT_MM = inner_flex_len + inner_flex_cap + inner_tetris_len
        OUTER_ADVANCE_LIMIT_MM = outer_flex_len + outer_flex_cap + guide_shaft


        
        # TODO: Review if system_advance should be the same as outer flex limit
        self.ARTICULATION_LIMITS = [(-2*np.pi,  2*np.pi),                       # outer_rot,
                                    (0,         OUTER_ADVANCE_LIMIT_MM),        # system_advance 
                                    (0,         np.deg2rad(INNER_FLEX_LIMIT_DEG)),    # outer_flex
                                    (-2*np.pi,  2*np.pi),                       # inner_rot
                                    (0,         INNER_ADVANCE_LIMIT_MM),        # inner_advance
                                    (0,         np.deg2rad(OUTER_FLEX_LIMIT_DEG)*1.1),    # inner_flex # TODO: remove temporary override
                                    # (0,         DEPTH_SLIDE_MM),                # depth_advance
                                    # (0,         CAPSULE_SLIDE_MM),              # deploy_advance
                                    ]
        
        self.LINK_LENGTHS = [0, outer_flex_len, 0, inner_flex_len, capsule_len]
        self.LINK_DIAMETERS = [OUTER_DIAMETER_MM, OUTER_DIAMETER_MM, INNER_DIAMETER_MM, INNER_DIAMETER_MM, CAPSULE_DIAMETER_MM] 
        self.JOINT_RANGES = np.array(self.ARTICULATION_LIMITS)[:,1] - np.array(self.ARTICULATION_LIMITS)[:,0]

    def compute_q_aug(self, q):
        '''
        Compute augmented joint positions from standard joint positions.

        param q: numpy.ndarray, shape (n_dof,) 
            Standard joint positions. {th1, d1, th2, th3, d2, th4} 
            for the 6-DOF catheter robot where th is a rotational joint and d is a prismatic joint.
        '''
        q_aug = np.zeros(11) #TODO: Update to the use the number of catheters

        # for partial extension of flex joints, consider alternative mappings where flex links can per partially embeded into the previous joints shortening the flex region
        if self.isEmbeddedKinematics:
            inner_length_extended = q[4] # current translation from base for outer

            if inner_length_extended > self.LINK_LENGTHS[3]:
                inner_rigid_len = self.LINK_LENGTHS[3]
                inner_flex_len = inner_length_extended - self.LINK_LENGTHS[3] # length of the flex segment
                
            else:
                inner_rigid_len = 0
                inner_flex_len = inner_length_extended # length of the flex segment is equal to the current translation from base for outer since the inner catheter is still embedded within the outer catheter


        else:
            if abs(q[2]) < 0.0001: # avoid division by zero for small angles, use limit value instead
                q_3 = self.LINK_LENGTHS[1]
            else:
                q_3 = 2*self.LINK_LENGTHS[1] * np.sin(q[2]/2) / q[2]

            if abs(q[5]) < 0.0001:
                q_8 = self.LINK_LENGTHS[3]
            else:
                q_8 = 2*self.LINK_LENGTHS[3] * np.sin(q[5]/2) / q[5] 

        q_aug[0] = q[0]
        q_aug[1] = q[1]
        q_aug[2] = q[2]/2
        q_aug[3] = q_3
        q_aug[4] = q[2]/2

        q_aug[5] = q[3]
        q_aug[6] = q[4]
        q_aug[7] = q[5]/2
        q_aug[8] = q_8
        q_aug[9] = q[5]/2

        q_aug[10] = 0 # depth advance not currently represented in augmented joints, treated as a rigid offset

       
        return q_aug
    
    def compute_a_aug(self):
        a_aug = np.zeros(self.n_aug) #TODO: Update to the use the number of catheters

        #TODO: Update base lengths depending on starting configuration. 
        # For example, if we start with the inner catheter fully extended, then 
        # we should set a_aug[2] and a_aug[7] to the length of the inner catheter,
        # and adjust a_aug[3] and a_aug[8] accordingly.

        a_aug[0] = 0                    
        a_aug[1] = 0
        a_aug[2] = 0                    
        a_aug[3] = 0                    
        a_aug[4] = 0
        a_aug[5] = 0
        a_aug[6] = 0
        a_aug[7] = 0
        a_aug[8] = 0
        a_aug[9] = 0
        a_aug[10] = self.LINK_LENGTHS[4]

        return a_aug

    def q_to_q_aug_mapping(self, q):
        J_a_q = np.zeros((self.n_aug,self.n_dof))

        if abs(q[2]) < 0.0001: # avoid division by zero for small angles, use taylor expansion limit value instead
            J_3 = -self.LINK_LENGTHS[1]/12 * q[2]
        else:
            J_3 = self.LINK_LENGTHS[1]/ q[2] * (np.cos(q[2]/2) - 2 * np.sin(q[2]/2) / q[2])

        if abs(q[5]) < 0.0001:
            J_8 = -self.LINK_LENGTHS[3]/12 * q[5]
        else:
            J_8 = self.LINK_LENGTHS[3]/ q[5] * (np.cos(q[5]/2) - 2 * np.sin(q[5]/2) / q[5])

        J_a_q[0][0] = 1
        J_a_q[1][1] = 1
        J_a_q[2][2] = 1/2
        J_a_q[3][2] = J_3
        J_a_q[4][2] = 1/2

        J_a_q[5][3] = 1
        J_a_q[6][4] = 1
        J_a_q[7][5] = 1/2
        J_a_q[8][5] = J_8
        J_a_q[9][5] = 1/2

        J_a_q[10][0] = 0 # depth advance does not affect the augmented joint variables, set to 1 to enable

        # print("J_a_q", J_a_q)
        return J_a_q
    
    def compute_dh_table(self, q_a):
        '''
        Docstring for compute_dh_table
        
        :param self: Description
        :param q_a: Description

        DH parameters Table is organized as follows:
        | Joint | theta (rotation) | alpha (twist) | r (link length) | d (link offset) |
        '''
        # print("q_a", q_a)
        DH = np.zeros( (self.n_aug, 4) )

        # Outer catheter DH parameters
        DH[0, :] = [q_a[0],  0,        0,   self.a_aug[0] ]
        DH[1, :] = [0,       np.pi/2,  0,   self.a_aug[1] + q_a[1]]
        DH[2, :] = [q_a[2],  -np.pi/2, 0,   self.a_aug[2]]
        DH[3, :] = [0,       np.pi/2,  0,   self.a_aug[3] + q_a[3]]
        DH[4, :] = [q_a[4],  -np.pi/2, 0,   self.a_aug[4]]

        # Inner catheter DH parameters
        DH[5, :] = [q_a[5],  0,        0,   self.a_aug[5] ]
        DH[6, :] = [0,       np.pi/2,  0,   self.a_aug[6] + q_a[6]]
        DH[7, :] = [q_a[7],  -np.pi/2, 0,   self.a_aug[7]]
        DH[8, :] = [0,       np.pi/2,        0,   self.a_aug[8] + q_a[8]]
        DH[9, :] = [q_a[9],  -np.pi/2,        0,   self.a_aug[9] ]

        # Depth advance treated as a rigid offset for now, not currently represented in augmented joint variables
        DH[10, :] = [0,      0,        0,   self.a_aug[10] + q_a[10] ]

        return DH
    
    def dh_transform(self, theta, alpha, r, d):
        """
        Compute the homogeneous transformation matrix using
        Denavit-Hartenberg (DH) parameters.

        Parameters
        ----------        
        theta: float
            Joint angle (radians)
        alpha: float
            Link Twist (radians)
        r: float
            Link length along x_{n-1}
        d: float
            Link offset along z_{n}

        Returns
        -------
        T_n_1_n: numpy.ndarray, shape (4,4)
            Homogeneous transformation matrix from frame {n-1} to {n}
        
        """
        T_n_1_n = np.eye(4)

        T_n_1_n = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),    np.sin(theta)*np.sin(alpha),    r*np.cos(theta)],
                            [np.sin(theta), np.cos(theta)*np.cos(alpha),     -np.cos(theta)*np.sin(alpha),   r*np.sin(theta)],
                            [0,             np.sin(alpha),                   np.cos(alpha),                  d],
                            [0,             0,                               0,                              1]])


        return T_n_1_n
    
    def compute_jacobian(self, q, updateState = True): 
        # TODO: Consider writing a version with an expanded jacobian to better
        # represent the decoupled inner and outer catheter kinematics. 
        self.q = q
        self.q_aug = self.compute_q_aug(self.q)
        dh_table = self.compute_dh_table(self.q_aug)
        
        J_x_a = np.zeros((6, self.n_aug))
        T = np.eye(4)
        T_back = [T] # store transformations for visualization and debugging

        p_vec = np.zeros((3, self.n_aug+1)) # vector of joint positions wrt global frame
        z_ax = np.tile( np.array([[0], [0], [1]]), (1, self.n_aug+1)) # vector of joint z axes wrt global frame


        # compute p_0_n (end effector position wrt global origin)
        for i in range(self.n_aug):
            theta, alpha, r, d = dh_table[i, :]
            T_n_1_n =  self.dh_transform(theta, alpha, r, d)

            T = T @ T_n_1_n
            T_back.append(T) # store transformations for visualization and debugging

            p_vec[:, i+1] = T[0:3, 3]
            z_ax[:,i+1] = T[0:3, 2]

        p_0_n = p_vec[:,-1] # end effector position vector
        
        # compute J_x_q (end effector jacobian wrt global origin)
        for i, link_type in enumerate(self.q_aug_labels):
            if link_type == "p": # check for prismatic joint based on index
                # print("prism at", i)
                J_x_a[0:3, i] = z_ax[:,i]
                J_x_a[3:6, i] = np.array([[0], [0], [0]]).ravel() # zero angular velocity contribution for prismatic joints
            else:
                # print("revol at", i)
                J_x_a[0:3, i] = np.cross(z_ax[:,i], (p_0_n - p_vec[:,i]) )
                J_x_a[3:6, i] = z_ax[:,i]
        
        J_a_q = self.q_to_q_aug_mapping(q)
        J_x_q = J_x_a @ J_a_q
        
        if updateState: # convert jacobian from augmented joint space to original joint space for control
            self.T = T
            self.T_back = T_back
            self.J = J_x_q

        return J_x_q, T, T_back
    
    def transform_along_curve(self, theta, phi,l):
        T  = np.eye(4)

        # TODO: Formally derive this rotation constant to align flex
        #  segments with DH paramter joints
        H1 = np.eye(4)
        H1[:3,:3] = self.Rz(np.pi)
        H2 = np.eye(4)
        H2[:3,:3] = self.Rx(np.pi/2)
        H1_inv = np.linalg.inv(H1 @ H2)

        if abs(theta) < 1e-4:
            T[:3, :3] = self.Ry(theta)
            T[:3,  3] = [0.0, 0.0, l]

            return H1_inv @ T
        
        r = l/theta
        T[:3, :3] = self.Ry(theta)
        T[:3,  3] = [r*(1 - np.cos(theta)), 0, r*np.sin(theta) ]

        return  H1_inv @ T  # apply final rotation to align with catheter frame
    
    def continous_curve(self, T_base, catheter_index, num=10):

        # TODO: Implement continuous curve function to compute position along the curve of the catheter based on the flex angle and length of the flex segment
        theta = self.q_aug[2]*2 if catheter_index==2 else self.q_aug[7]*2 # flex angle for the specified catheter
        length = self.LINK_LENGTHS[1] if catheter_index==2 else self.LINK_LENGTHS[3]
        phi = self.q_aug[0] if catheter_index==2 else self.q_aug[5] # rotation angle for the specified catheter

        # print("theta", theta)
        # print("length", length)
        theta_linspace = np.linspace(0, theta, num=num) # 10 points along the curve segment
        len_linspace = np.linspace(0, length, num=num) # corresponding lengths along the curve segment

        T_trans = np.eye(4)
        T_trans[:3, 3] = T_base[:3, 3] # set the translation component of T_trans to the translation component of T_base
        curve_vect = []
        for theta_i, len_i in zip(theta_linspace, len_linspace):  #TODO: review if we should skip the first points since the rot angle is zero.
            T_i = self.transform_along_curve(theta_i, phi, len_i)
            T_result = T_base @ T_i
            # T_result = T_i
            curve_vect.append(T_result[0:3, 3]) # extract position vector from transformation matrix
        
        # print("curve_vect", curve_vect)
        return curve_vect # return the list of transformations along the curve for visualization and debugging and the corresponding position vectors
    
    def compute_catheter_positions(self):
        # TODO: UPdate to segmetize segments by 1mm increments for smoother visualization of the catheter shape, currently we are just visualizing the joint positions and approximating the curve segments with 10 points which may not be sufficient for larger flex angles. We can use the transform_along_curve function to compute the position along the curve of the catheter based on the flex angle and length of the flex segment, and generate a list of transformations along the curve for visualization and debugging.
        # TODO: polish iterable look to avoid repeating points and improve bending joint detection, currently we are checking the catheter index to determine which joint angles and link lengths to use for the curve transformation, we can instead check the type of joint (revolute or prismatic) and apply the appropriate transformation based on the joint type for better scalability to different robot configurations and to avoid hardcoding joint indices which may lead to errors if the configuration changes.
        pos_vect = []
        skel_vect = []

        T_back = self.T_back[1:] # skip the first transformation which is the base frame
        
        num_straight = 3 # number of points to represent the curve segments, adjust for visualization density
        num_curve = 4 # number of points to represent the straight segments, adjust for visualization density

        for i in range(1, len(T_back)):
            if ((i==2) or (i==7)):                            # for the flex joints, we can compute the position along the curve of the catheter using the length of the flex segment and the angle of flexion
                # print("flex section at", i)
                pos_vect.extend( self.continous_curve(T_back[i-1], i, num_curve) )# TODO: Implement continuous curve function to compute position along the curve of the catheter based on the flex angle and length of the flex segment
            elif i != 2+1 and i != 7+1:                                   # for the prismatic joints, we can directly take the position from the transformation matrix
                linear_pts = [T_back[i-1][0:3, 3] + t * (T_back[i][0:3, 3] - T_back[i-1][0:3, 3]) for t in np.linspace(0, 1, num=num_straight)] # generate linearly spaced points between the previous joint position and the current joint position for visualization of the straight segments
                # print(linear_pts)
                pos_vect.extend(linear_pts) # add linearly spaced points between the previous joint position and the current joint position for visualization of the straight segments
                
                # print("straight section at", i)

            # print("position from T", T_back[i][0:3, 3])
            skel_vect.append(T_back[i][0:3, 3])

        return np.array(pos_vect), np.array(skel_vect) # return as 2D array with shape (3, num_points) for visualization and debugging
    
    def Rz(self, phi):
        T = np.array([[np.cos(phi), -np.sin(phi), 0],
                      [np.sin(phi),  np.cos(phi), 0],
                      [          0,            0, 1]])
        return T

    def Ry(self, theta):
        T = np.array([[ np.cos(theta), 0, np.sin(theta)],
                      [             0, 1,             0],
                      [-np.sin(theta), 0, np.cos(theta)]])
        return T

    def Rx(self, theta):
        T = np.array([[1,             0,              0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta),  np.cos(theta)]])
        return T


def main():
    np.set_printoptions(precision=3, suppress=True)
    CONFIG_FILE = './tutorial/catheter/configs/intrepid_35fr.json'
    kin = RobotKinematics(CONFIG_FILE)
    # print(kin.q_aug)

    q_test = np.array([0, 20 , np.pi/2, 0, 20, np.pi/2])
    q_test = np.array([1, 20 , -np.pi*1/2, 0, 20, np.pi*1/2])

    q_test =  [
        -0.8,                # outer_rot = 0
        120.0,                # system_advance = 0
        1.50,                # outer_flex = 0
        -1.5,                # inner_rot = 0
        0.0,               # inner_advance = set to 23 mm adjusted for better start
        1.8,                # inner_flex = 0
        ]

    q_test2 =  [
        0,                # outer_rot = 0
        20.0,                # system_advance = 0
        np.pi/4,                # outer_flex = 0

        0,                # inner_rot = 0
        0,                # inner_advance = 0
        np.pi/4,                # inner_flex = 0
        ]
    
    q_test0 = np.array([0, 0 , 0, 0, 0, 0])
    J, T, T_back = kin.compute_jacobian(q=q_test2) 
    print(J)

    pos_vect, pos_vect_skel = kin.compute_catheter_positions()

    # print(pos_vect)

    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pos_vect = np.array(pos_vect)
        ax.plot(pos_vect_skel[:,0], pos_vect_skel[:,1], pos_vect_skel[:,2], '-x', label='Skeleton')
        ax.plot(pos_vect[:,0], pos_vect[:,1], pos_vect[:,2], '-o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Robot Kinematics Visualization')
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(0, 200)
        ax.set_box_aspect([1,1,1])  # Equal aspect ratio for all axes
        plt.show()


if __name__ == "__main__":
    main()

    
