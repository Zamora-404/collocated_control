import numpy as np
import matplotlib.pyplot as plt
import json


class RobotKinematics():
    def __init__(self, config_file, q_0=np.zeros(6 )):
        
        # Define the robot's kinematic parameters from json config
        self.LINK_LENGTHS = None
        self.LINK_DIAMETERS = None
        self.ARTICULATION_LIMITS = None
        self.JOINT_LABELS = ["outer_rot", "outer_advance", "outer_flex", "inner_rot",
                            "inner_advance", "inner_flex", "depth_advance", "deploy_advance"]
        self.device_config = self.load_config(config_file)
        

        # define robot's initial configuration (home position)
        self.q_0 = q_0
        self.q_aug = self.compute_q_aug(self.q_0)
        self.a_aug = self.compute_a_aug()
        self.n_aug = len(self.q_aug)

        self.T = None # end effector transformation matrix
        self.J = None # end effector jacobian matrix
        self.T_back = None # vector of catheter joint positions for visualization and debugging


    def load_config(self, config_file): 
        #TODO: Request a diagram of configuration parameters and how they relate 
        # to the robot's geometry and kinematics. This will help ensure that we
        #  are interpreting the parameters correctly and using them appropriately 
        # in our kinematics calculations.

        with open(config_file, 'r') as f:
            device_config = json.load(f)

        component_lengths = device_config["component_lengths"]
        component_diameters = device_config["component_diameters"]
        articulation_limits = device_config["articulation_limits"]

        # Extract constants
        CAPSULE_SLIDE_MM = articulation_limits["CAPSULE_SLIDE"]
        DEPTH_SLIDE_MM = articulation_limits["DEPTH_SLIDE"]
        OUTER_FLEX_DEG = articulation_limits["OUTER_FLEX"]
        INNER_FLEX_DEG = articulation_limits["INNER_FLEX"]

        OUTER_DIAMETER_MM = component_diameters["OUTER_FLEX"]
        INNER_DIAMETER_MM = component_diameters["INNER_FLEX"]
        CAPSULE_DIAMETER_MM = component_diameters["CAPSULE"]

        inner_flex_len = float(component_lengths["INNER_FLEX"])
        inner_flex_cap = float(component_lengths["INNER_FLEX_CAP"])
        outer_flex_len = float(component_lengths["OUTER_FLEX"])
        outer_flex_cap = float(component_lengths["OUTER_FLEX_CAP"])
        inner_tetris_len = float(component_lengths["INNER_TETRIS"])
        capsule_len = float(component_lengths["CAPSULE"])
        guide_shaft = float(component_lengths["GUIDE_SHAFT"])

        INNER_ADVANCE_LIMIT_MM = inner_flex_len + inner_flex_cap + inner_tetris_len
        OUTER_ADVANCE_LIMIT_MM = guide_shaft

        # TODO: Review if system_advance should be the same as outer flex limit
        self.ARTICULATION_LIMITS = [(-2*np.pi,  2*np.pi),                            # outer_rot,
                                    (0,         OUTER_ADVANCE_LIMIT_MM),              # system_advance 
                                    (0,         np.deg2rad(INNER_FLEX_DEG)),           # outer_flex
                                    (-2*        np.pi, 2*np.pi),                    # inner_rot
                                    (2*inner_flex_cap, INNER_ADVANCE_LIMIT_MM),     # inner_advance
                                    (0,         np.deg2rad(OUTER_FLEX_DEG)),           # inner_flex
                                    (0,         DEPTH_SLIDE_MM),                      # depth_advance
                                    (0,         CAPSULE_SLIDE_MM),                     # deploy_advance
                                    ]
        
        self.LINK_LENGTHS = [0, outer_flex_len, 0, inner_flex_len, capsule_len]
        self.LINK_DIAMETERS = [OUTER_DIAMETER_MM, OUTER_DIAMETER_MM, INNER_DIAMETER_MM, INNER_DIAMETER_MM, CAPSULE_DIAMETER_MM] 

    def compute_q_aug(self, q):
        q_aug = np.zeros(10) #TODO: Update to the use the number of catheters

        if q[2] < 0.0001: # avoid division by zero for small angles, use limit value instead
            d_3 = self.LINK_LENGTHS[1]
        else:
            d_3 = 2*self.LINK_LENGTHS[1] * np.sin(q[2]/2) / q[2]

        if q[5] < 0.0001:
            d_8 = self.LINK_LENGTHS[3]
        else:
            d_8 = 2*self.LINK_LENGTHS[3] * np.sin(q[5]/2) / q[5] 

        q_aug[0] = q[0]
        q_aug[1] = q[1]
        q_aug[2] = q[2]/2
        q_aug[3] = d_3
        q_aug[4] = q[2]/2

        q_aug[5] = q[3]
        q_aug[6] = q[4]
        q_aug[7] = q[5]/2
        q_aug[8] = d_8
        q_aug[9] = q[5]/2

       
        
        return q_aug
    

    def compute_a_aug(self):
        a_aug = np.zeros(10) #TODO: Update to the use the number of catheters

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

        return a_aug

    def q_to_q_aug_mapping(self, q):
        J_a_q = np.zeros((6,5))

        J_a_q[0][0] = q[0]
        J_a_q[1][1] = q[1]
        J_a_q[2][2] = q[2]/2
        J_a_q[3][2] = (self.a_aug[1]) * np.sin(q[2]/2) / q[2]
        J_a_q[4][2] = q[2]/2

        J_a_q[5][0] = q[3]
        J_a_q[6][1] = q[4]
        J_a_q[7][2] = q[5]/2
        J_a_q[8][2] = (self.a_aug[3]) * np.sin(q[2]/2) / q[5]
        J_a_q[9][2] = q[5]/2

        return J_a_q
    

    def compute_dh_table(self, q_a):
        '''
        Docstring for compute_dh_table
        
        :param self: Description
        :param q_a: Description

        DH parameters Table is organized as follows:
        | Joint | theta (rotation) | alpha (twist) | r (link length) | d (link offset) |
        '''
        print("q_a", q_a)
        DH = np.zeros( (self.n_aug, 4) )

        DH[0, :] = [q_a[0],  0,        0,   self.a_aug[0] ]
        DH[1, :] = [0,       np.pi/2,  0,   self.a_aug[1] + q_a[1]]
        DH[2, :] = [q_a[2],  -np.pi/2, 0,   self.a_aug[2]]
        DH[3, :] = [0,       np.pi/2,        0,   self.a_aug[3] + q_a[3]]
        DH[4, :] = [q_a[4],  -np.pi/2,        0,   self.a_aug[4]]

        DH[5, :] = [q_a[5],  0,        0,   self.a_aug[5] ]
        DH[6, :] = [0,       np.pi/2,  0,   self.a_aug[6] + q_a[6]]
        DH[7, :] = [q_a[7],  -np.pi/2, 0,   self.a_aug[7]]
        DH[8, :] = [0,       np.pi/2,        0,   self.a_aug[8] + q_a[8]]
        DH[9, :] = [q_a[9],  0,        0,   self.a_aug[9]]

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
    
    
    def compute_jacobian(self, q): 
        # TODO: Consider writing a version with an expanded jacobian to better
        # represent the decoupled inner and outer catheter kinematics. 

        self.q_aug = self.compute_q_aug(q)
        dh_table = self.compute_dh_table(self.q_aug)
        
        J_x_q_a = np.zeros((6, self.n_aug))
        T = np.eye(4)
        T_back = [T] # store transformations for visualization and debugging

        p_vec = np.zeros((3, self.n_aug+1)) # vector of joint positions wrt global frame
        z_ax = np.tile( np.array([[0], [0], [1]]), (1, self.n_aug+1)) # vector of joint z axes wrt global frame


        # compute p_0_n (end effector position wrt global origin)
        for i in range(self.n_aug):
            theta, alpha, r, d = dh_table[i, :]
            T_n_1_n =  self.dh_transform(theta, alpha, r, d)
            # print("T",T)
            # print(T_n_1_n)

            T = T @ T_n_1_n
            T_back.append(T) # store transformations for visualization and debugging

            # print(T)
            # print(T[0:3, 2])
            # print(z_ax[:,i+1])

            p_vec[:, i+1] = T[0:3, 3]
            z_ax[:,i+1] = T[0:3, 2]

        p_0_n = p_vec[:,-1] # end effector position vector
        
        # compute J_x_q (end effector jacobian wrt global origin)
        for i in range(self.n_aug):
            if (i % 5) % 2 == 1: # check for prismatic joint based on index
                # print("prism at", i)
                J_x_q_a[0:3, i] = z_ax[:,i]
                J_x_q_a[3:6, i] = np.array([[0], [0], [0]]).ravel() # zero angular velocity contribution for prismatic joints
            else:
                # print("revol at", i)
                J_x_q_a[0:3, i] = np.cross(z_ax[:,i], (p_0_n - p_vec[:,i]) )
                J_x_q_a[3:6, i] = z_ax[:,i]

        self.T = T
        self.J = J_x_q_a
        self.T_back = T_back

        return J_x_q_a, T, T_back
    

    def transform_along_curve(self, theta, phi,l):
        T  = np.eye(4)

        H1 = np.eye(4)
        H1[:3, :3] = self.Rz(phi)

        H3 = np.eye(4)
        H3[:3, :3] = self.Rz(-phi)

        H4 = np.eye(4)
        H4[:3, :3] = self.Rz(np.pi/2) @ self.Rx(np.pi/2) # rotate to align with catheter frame

        
        if abs(theta) < 1e-4:
            H2 = np.eye(4)
            H2[:3, :3] = self.Ry(theta)
            H2[:3,  3] = [0.0, 0.0, l]
        else:
            # print(theta, phi, l)
            r = l/theta

            H2 = np.eye(4)
            H2[:3, :3] = self.Ry(theta)
            H2[:3,  3] = [r*(1 - np.cos(theta)),
                                             0,
                              r*np.sin(theta) ]
            print(H2[:3,  3])
            
        
        T = np.linalg.inv(H4) @ H1 @ H2 @ H3

        return T  #@ H4 # apply final rotation to align with catheter frame
    
    def continous_curve(self, T_base, catheter_index):

        # TODO: Implement continuous curve function to compute position along the curve of the catheter based on the flex angle and length of the flex segment
        theta = self.q_aug[2]*2 if catheter_index==1 else self.q_aug[7]*2 # flex angle for the specified catheter
        length = self.LINK_LENGTHS[1] if catheter_index==1 else self.LINK_LENGTHS[3]
        phi = self.q_aug[0] if catheter_index==1 else self.q_aug[5] # rotation angle for the specified catheter

        # print("theta", theta)
        # print("length", length)
        theta_linspace = np.linspace(0, theta, num=5) # 10 points along the curve segment
        len_linspace = np.linspace(0, length, num=5) # corresponding lengths along the curve segment

        T_trans = np.eye(4)
        T_trans[:3, 3] = T_base[:3, 3] # set the translation component of T_trans to the translation component of T_base
        curve_vect = []
        for theta_i, len_i in zip(theta_linspace, len_linspace):
            T_i = self.transform_along_curve(theta_i, phi, len_i)
            T_result = T_base @ T_i
            # T_result = T_i
            curve_vect.append(T_result[0:3, 3]) # extract position vector from transformation matrix
        
        # print("curve_vect", curve_vect)
        return curve_vect # return the list of transformations along the curve for visualization and debugging and the corresponding position vectors
    
    
    def compute_catheter_positions(self):
        pos_vect = []

        T_back = self.T_back[1:] # skip the first transformation which is the base frame

        for i in range(0, len(T_back)):        
            if ((i==1) or (i==6)) and i>0:                            # for the flex joints, we can compute the position along the curve of the catheter using the length of the flex segment and the angle of flexion
                print("flex section at", i)
                pos_vect.extend( self.continous_curve(T_back[i], i) )#TODO: Implement continuous curve function to compute position along the curve of the catheter based on the flex angle and length of the flex segment
            else:                                   # for the prismatic joints, we can directly take the position from the transformation matrix
                pos_vect.append(T_back[i][0:3, 3])
                print("straight section at", i)
            print("position from T", T_back[i][0:3, 3])

        return np.array(pos_vect).T # return as 2D array with shape (3, num_points) for visualization and debugging
    

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
    print(kin.q_aug)

    q_test = np.array([0, 20 , np.pi/2, 0, 20, np.pi/2])
    q_test = np.array([0, 20 , np.pi/4, 0, 20, np.pi/4])
    # q_test = np.array([0, 20 , 0, 0, 20, 0])
    J, T, T_back = kin.compute_jacobian(q=q_test) 
    print(T)

    pos_vect_skel = np.zeros((3, len(T_back)-1))
    for i, T in enumerate(T_back[1:]): # skip the first transformation which is the base frame
        pos_vect_skel[:, i] = T[0:3, 3]
    
    print(pos_vect_skel.T)

    pos_vect = kin.compute_catheter_positions()

    print(pos_vect.T)

    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pos_vect = np.array(pos_vect)
        ax.plot(pos_vect_skel[0,:], pos_vect_skel[1,:], pos_vect_skel[2,:], '-x', label='Skeleton')
        ax.plot(pos_vect[0,:], pos_vect[1,:], pos_vect[2,:], '-o')
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