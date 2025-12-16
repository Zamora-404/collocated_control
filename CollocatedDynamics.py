import math
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axess3D
import matplotlib.animation as animation
import time
from scipy.integrate import solve_ivp
from q_optimization import Q_Optimize

''' Notes on units used:
    mass:       kg
    volume:     mL
    length:     mm
    Force:      N
    Pressure:   kPa???
    
    q_u = [phi, theta, s]'''


#Initializing variables

n = 3           # number of segments
i_goal = 0
arduino = []
sensor = []
base_length = 5 # mm
tip_coords = np.array([0,0,base_length])
tip_prev = tip_coords.copy()
q_a_prev = np.array([0,0,0])
d_q_a_prev = np.array([0,0,0])
q_u_prev = np.array([base_length, 0, 0]*n)
d_q_u_prev = np.array([0,0,0]*n)
q_f_prev = np.array([0,0,0]*n)
time_now = 0
time_prev = 0
u_volumes = np.array([0,0,0])




class CollocatedDynamics:
    def __init__(self, u_init):

        self.n    = 3                   #Number of segments for the SBA
        self.d    = 4.75                #Geometric parameter chamber radial distance
        self.gain = 1                   #Gain for the U_ell
        self.r    = 9.5                  # rigid offset from sensor (7mm w/o ball bearing)
        self.K = np.diag([15,15,.5]*self.n)
        self.D = np.diag([5,5,5]*self.n)
        self.I = np.diag((self.K.diagonal()**2)/(4*self.D.diagonal()))

        self.i_goal = 0
        self.base_length = 5

        self.tip_prev = np.array([0,0,self.base_length])
        self.goal_prev = np.array([0,0,0])
        self.time_now = time.time()
        self.time_prev = self.time_now - 0.05

        self.q_a_prev = u_init/1000 * 50 # convert from uL to mL teh convert to mm via linear fit
        self.d_q_a_prev = np.array([0,0,0])

        self.q_u_prev = self.q_no_load(self.q_a_prev + self.base_length) 
        self.q_u_0_prev = self.q_no_load(self.q_a_prev + self.base_length)
        self.q_f_prev = np.array([0,0,0]*self.n)

        self.opt = Q_Optimize()

        self.b = 0
        self.alpha = 0.0003
        self.k = 5

        self.sigma = 0


        self.Efi = 0
        self.kf = 80
        self.kv = 0.1
        self.ki = 0
        self.kp = 2

        self.v_f_prev = 0

        self.error_f_integral = 0

        self.hasReachedGoal = False
        self.hasRechedStart = False
        self.T_0 = 0 # start time for the path

        self.f_des_prev = .1
        
        # print(self.q_a_prev,  self.q_u_prev )

        self.K_u = np.eye(3*self.n )
        self.D_u = np.eye(3*self.n )
        


    def volume_to_length(self, v1, v2, v3):
        l1, l2, l3 = 45.67*v1,  45.67*v2,  45.67*v3

        return l1, l2, l3

    def lengths_to_volumes(self, ells, base_height):
        volumes = 100/4667 * (ells - base_height)
        return volumes*1000

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


    def Link_function(self, qi): #Method to calculate the length of each prismatic links of each segment
        theta_i = qi[1,0]        #based on current configuration variables
        S_i     = qi[2,0]        #also evaluating the variation of these link lengths w.r.t change in theta, and arc length
                                 #The link length does not depend upon the out of the plane angle (phi)
        if theta_i < 1e-3:
            li = S_i
            dl_theta = 0
            dl_S = 1
        else:
            li = 2*S_i*np.sin(theta_i/2)/theta_i
            dl_theta = 2*S_i*(np.cos(theta_i/2)/(2*theta_i) - np.sin(theta_i/2)/(theta_i**2))
            dl_S = 2*np.sin(theta_i/2)/theta_i

        return li, dl_theta, dl_S

    def single_map(self, qi):
        #Method helps to evaluate the Xi vector along with the Jacobian_xi_w.r.t q for each segment
        phi   = qi[0,0]
        theta = qi[1,0]

        li, dl_theta, dl_S = self.Link_function(qi)

        xi = np.array([[phi], [theta/2], [li], [theta/2], [-phi]])

        Jmi = np.array([[1,0,0], [0, 0.5, 0], [0,dl_theta,dl_S], [0,0.5,0],[-1,0,0]])

        return xi, Jmi

    def Build_XI(self, q):
        #Method helps to build the final Xi which is a (3*n,1) vector and J_xi_q which is a (5*n, 3*n)
        xi = np.zeros((5*self.n,1))
        Jm = np.zeros((5*self.n, 3*self.n))

        for i in range(self.n):
            rows = slice(i*5, i*5 + 5)
            qi = q[3*i : 3*i + 3].reshape(3,1)
            xi_block, Jmi = self.single_map(qi)

            xi[rows] = xi_block
            Jm[rows, 3*i : 3*i + 3] = Jmi
        return xi, Jm

    def Augmented_DH(self, q):
        #Function to derive the DH Table based on our revolute and prismatic joints
        DH_aug = np.zeros((5*self.n, 4))
        Xi, _ = self.Build_XI(q)

        for i in range(self.n):
            # if i == (self.n - 1):
            #     base = i*5
            #     xi = Xi[base: base + 5, 0]
            #     DH_aug[base + 0, :] = [xi[0],    0,    0, -np.pi/2]
            #     DH_aug[base + 1, :] = [xi[1],    0,    0,  np.pi/2]
            #     DH_aug[base + 2, :] = [0,       xi[2], 0, -np.pi/2]
            #     DH_aug[base + 3, :] = [xi[3],    0,    0,  np.pi/2]
            #     DH_aug[base + 4, :] = [xi[4],    r,    0,        0]
            # else:
            base = i*5
            xi = Xi[base: base + 5, 0]
            DH_aug[base + 0, :] = [xi[0],    0,    0, -np.pi/2]
            DH_aug[base + 1, :] = [xi[1],    0,    0,  np.pi/2]
            DH_aug[base + 2, :] = [0,       xi[2], 0, -np.pi/2]
            DH_aug[base + 3, :] = [xi[3],    0,    0,  np.pi/2]
            DH_aug[base + 4, :] = [xi[4],    0,    0,        0]

        return DH_aug

    def DH_transform(self, theta, d, a, alpha):
        #Basic 4x4 Tranformation matrix
        T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [0,                            np.sin(alpha),                np.cos(alpha),               d],
                      [0,                                     0,                               0,               1]])

        return T

    def Compute_Augmented_Jacobian(self, dh_aug):
        # This function allows us to evaluate the Jacobian for the RRPRR model. The jacobian size is
        # 3 x 5*n, (here n is number of segments and 5 basically is the number of joints per segment
        # in our case we have 5 joints per segment)

        N = 5*self.n
        T = np.eye(4)
        p = np.zeros((3, N+1))
        z = np.tile(np.array([[0.],[0.],[1.]]), (1 , N+1))

        new_dh_aug = dh_aug
        new_dh_aug[-1,1] = self.r

        for k in range(N):
            theta, d, a, alpha = new_dh_aug[k]
            T     = T@self.DH_transform(theta,d,a,alpha)
            p[:, k+1] = T[0:3, 3]
            z[:, k+1] = T[0:3, 2]

        p_tip = p[:, -1]
        Jxi   = np.zeros((3, N))

        for k in range(N):
            idx = k%5
            if idx == 2:
                Jxi[:,k] = z[0:3, k]
            else:
                Jxi[:, k] = np.cross(z[0:3,k], (p_tip - p[:,k]))

        return Jxi
    
    def Compute_Full__Augmented_Jacobian(self, dh_aug):
        # This function allows us to evaluate the Jacobian for the RRPRR model. The jacobian size is
        # 3 x 5*n, (here n is number of segments and 5 basically is the number of joints per segment
        # in our case we have 5 joints per segment)
        N = 5*self.n
        T = np.eye(4)
        p = np.zeros((3, N+1))
        z = np.tile(np.array([[0.],[0.],[1.]]), (1 , N+1))
        new_dh_aug = dh_aug
        new_dh_aug[-1,1] = self.r
        for k in range(N):
            theta, d, a, alpha = new_dh_aug[k]
            T     = T@self.DH_transform(theta,d,a,alpha)
            p[:, k+1] = T[0:3, 3]
            z[:, k+1] = T[0:3, 2]
        p_tip = p[:, -1]
        Jxi   = np.zeros((6, N))
        for k in range(N):
            idx = k%5
            if idx == 2:
                Jxi[:3,k] = z[0:3, k]
                Jxi[3:,k] = np.array([[0.],[0.],[0.]]).ravel()
            else:
                Jxi[:3, k] = np.cross(z[0:3,k], (p_tip - p[:,k]))
                Jxi[3:, k] = z[0:3,k]
        return Jxi

    def ComputeJacobian(self, q):
        global r
        # Evaluating the final jacobian for the soft robot J(q). Its size is 3 x 3*n
        dh_aug = self.Augmented_DH(q)
        Jxi    = self.Compute_Augmented_Jacobian(dh_aug)
        _, Jm  = self.Build_XI(q)

        J_q    = Jxi @ Jm
    

        return J_q
    
    def ComputeFullJacobian(self, q):
        dh_aug = self.Augmented_DH(q)
        Jxi = self.Compute_Full__Augmented_Jacobian(dh_aug)
        _, Jm = self.Build_XI(q)
        J_q = Jxi @ Jm
        return J_q

    def q_diff(self, q1, q2):
        def angular_difference(phi1,phi2):
            diff = (phi1 - phi2 + math.pi) %(2*math.pi) - math.pi
            return diff

        diff_q = np.zeros(3*self.n)
        for i in range(self.n):
            idx = 3*i
            diff_q[idx] = angular_difference(q1[idx], q2[idx])
            diff_q[idx + 1] = q1[idx + 1] - q2[idx + 1]
            diff_q[idx + 2] = q1[idx + 2] - q2[idx + 2]

        return diff_q


    def q_no_load(self, ell):
        # Function to evaluate the q under no load from the given Ells

        q_0 = np.zeros(3*self.n)


        ell_1, ell_2, ell_3 = ell

        S = (ell_1 + ell_2 + ell_3)/3


        if np.isclose(ell_1,ell_2, atol = 0.00095) and np.isclose(ell_2, ell_3, atol  = 0.00095) and np.isclose(ell_1, ell_3, atol = 0.00095):
            theta = 0.0
            kappa = 0.0
            phi = 0
        else:
            kappa = 2*math.sqrt(ell_1**2 + ell_2**2 + ell_3**2 - ell_1 * ell_2 - ell_1*ell_3 - ell_2*ell_3)/(self.d*(ell_1 + ell_2 + ell_3))
            theta = kappa * S
            phi = math.atan2(math.sqrt(3) * (ell_2 + ell_3 - 2*ell_1), (3*(ell_2 - ell_3)))
            phi = phi % (2 * math.pi)



        for i in range(self.n):
            idx = i*3
            q_0[idx] = phi
            q_0[idx+1] = theta/self.n
            q_0[idx+2] = S/self.n

        return q_0

    def Compute_Jacobian_Ell_Q(self, q):
        # Function to evaluate the jacobian relating change in Ell (q_a) w.r.t change in q_u
        q = np.asarray(q).reshape(-1)
        J = np.zeros((3,3*self.n))
        def Compute_d_Ell(q,d,i,j):
            if i ==0:
                dphi   = -d*q[j+1]*np.cos(q[j])
                dtheta = -d*np.sin(q[j-1])
            elif i ==1:
                dphi   = d*q[j+1]*np.sin(math.pi/6 - q[j])
                dtheta = d*np.cos(math.pi/6 - q[j-1])
            else:
                dphi   = d*q[j+1]*np.sin(math.pi/6 + q[j])
                dtheta = -d*np.cos(math.pi/6 + q[j-1])
            return dphi, dtheta

        for i in range(3):
            for j in range(0,3*self.n,3):
                J[i,j],_ = Compute_d_Ell(q,self.d,i,j)
            for j in range(1,3*self.n,3):
                _,J[i,j] = Compute_d_Ell(q,self.d,i,j)
            for j in range(2,3*self.n,3):
                J[i,j] = 1
        return J

    def Actuated_Jacobian(self, q):
        # Final actuated Jacobian equating change in tip_coordinates based on changes in Ell
        J_rq = self.ComputeJacobian(q) # J_q = del_x/del_qu
        J_Eq = self.Compute_Jacobian_Ell_Q(q) # J_au = del_q_a/del_q_u

        J_Eq_inv = np.linalg.pinv(J_Eq)

        J_a = J_rq @ J_Eq_inv

        return J_a, J_rq



    # Next three functions basically help to evaluate the plotting points based on the q, in other words
    # is the forward_kinematics of the SBA

    def Transform(self, theta, phi,l):
        T  = np.eye(4)

        H1 = np.eye(4)
        H1[:3, :3] = self.Rz(phi)

        H3 = np.eye(4)
        H3[:3, :3] = self.Rz(-phi)

        if abs(theta) < 1e-4:
            H2 = np.eye(4)
            H2[:3, :3] = self.Ry(theta)
            H2[:3,  3] = [0.0, 0.0, l]

            T = H1 @ H2 @ H3
        else:
            r = l/theta

            H2 = np.eye(4)
            H2[:3, :3] = self.Ry(theta)
            H2[:3,  3] = [r*(1 - np.cos(theta)),
                                            0.0,
                                r*np.sin(theta)]
            T = H1 @ H2 @ H3

        return T

    def Position_Vector(self, theta, samples, phi, l, T):
        t = np.linspace(0, theta, samples)
        X = np.zeros((3, samples))
        L = np.linspace(0,l,samples)

        for i in range(samples):
            Mat = self.Transform(t[i], phi, L[i])
            Transform_matrix = T @ Mat
            X[:, i] = Transform_matrix[:3, 3]

        return X

    def Compute_Soft_Curve (self, q):
        q = np.asarray(q).reshape(-1)
        samples = 5
        T = np.eye(4)
        softPts = np.zeros((3, samples*self.n))
        skelPts = np.zeros((3, self.n + 1))
        v = np.zeros((3,1))
        phi_s   = q[0::3]
        theta_s = q[1::3]
        L       = q[2::3]

        Tr = np.eye(4)
        Tr[0:3,3] = np.array([0,0,self.r])

        for i in range(self.n):
            Xi = self.Position_Vector(theta_s[i], samples, phi_s[i], L[i], T)
            start = i*samples
            end   = start + samples
            softPts[:, start:end] = Xi
            T = T @ self.Transform(theta_s[i], phi_s[i], L[i])
            skelPts[:, i+1] = softPts[:, end - 1]
        
        T_ef = T @ Tr
        # print(skelPts)
        # print( np.array([ T_ef[0:3,3] ]).T )
        skelPts = np.hstack( (skelPts, np.array([ T_ef[0:3,3] ]).T ) )
        # skelPts = np.hstack([skelPts, T_ef[0:3,3]])
        # print(skelPts)

        v = skelPts[:, -1]

        return softPts, skelPts, v

    def Compute_actual_tip(self,q):
        dh_aug = self.Augmented_DH(q)
        new_dh_aug = dh_aug
        new_dh_aug[-1,1] = self.r

        T = np.eye(4)

        for k in range(len(new_dh_aug[:,0])):
            theta, d, a, alpha = new_dh_aug[k]
            T = T @ self.DH_transform(theta, d, a, alpha)

        actual_tip = T[:3,3]

        return actual_tip

    def Compute_tip_pose(self, q):
        dh_aug = self.Augmented_DH(q)
        new_dh_aug = dh_aug
        new_dh_aug[-1,1] = self.r

        T = np.eye(4)

        for k in range(len(new_dh_aug[:,0])):
            theta, d, a, alpha = new_dh_aug[k]
            T = T @ self.DH_transform(theta, d, a, alpha)

        tip_pose = T


        return tip_pose
    
    def ell_q(self,q):
        ell_1 = q.copy()
        ell_2 = q.copy()
        ell_3 = q.copy()
        for i in range(0,3*self.n,3):
            ell_1[2+i] = q[2+i] - self.d*q[i+1]*np.sin(q[i])
            ell_2[2+i] = q[2+i] + self.d*q[i+1]*np.cos((math.pi/6) - q[i])
            ell_3[2+i] = q[2+i] - self.d*q[i+1]*np.cos((math.pi/6) + q[i])

        return ell_1, ell_2, ell_3
    
    def compute_q_a_dynamics(self, volumes, q_a_prev, d_q_a_prev, dt):
        """
        Solves the ODE for q_a using the equation:
        M_a * ddq_a + D_a * dq_a + K_a * q_a = A_a * u
        """
        M_a, C_a, D_a, K_a, A_a = self.actuated_dynamics_matrices(q_a_prev, d_q_a_prev, volumes)
    

        # Define dynamics as first-order system
        def ode(t, y):
            q_a = y[:3]
            dq_a = y[3:]
            ddq_a = np.linalg.inv(M_a) @ (A_a @ volumes - D_a @ dq_a - K_a @ q_a)
            return np.concatenate((dq_a, ddq_a))
        
        #Verify the ode function

        # Initial state TODO:
        y0 = np.concatenate((q_a_prev, d_q_a_prev))

        # Integrate over one time step
        sol = solve_ivp(ode, [0, dt], y0, method='RK45', t_eval=[dt])

        
        q_a = sol.y[:3, -1]
        d_q_a = sol.y[3:, -1]
        
        return q_a, d_q_a
    

    def Matrix_inverse(self, J):
        U, S, Vh = np.linalg.svd(J, full_matrices = False)
        V = Vh.T
        sigma_0 = 0.01*max(S)
        nu = 50
        h = (S**3 +nu*S**2 + 2*S + 2*sigma_0)/(S**2 + nu*S + 2)
        H_inv = np.diag(1.0/h)
        J_inv = V @ H_inv @ U.T

        return J_inv

    def task_stiffness(self, J, K_u):
        K_u_inv = self.Matrix_inverse(K_u)
        K_c_inv = J @ K_u_inv @ J.T

        # K_c = self.Matrix_inverse(K_c_inv)

        return K_c_inv

    
    
    def compute_q_u_dynamics_2(self, q_a, F_ext, q_f_prev, dt, T_tip = np.eye(4)): 
        # print(self.q_u_0_prev)
        # q_u_0 = self.q_u_0_prev
        q_u_0 = self.q_no_load(q_a + self.base_length)         # Compute no load underacttuated vars
        J_u  = self.ComputeJacobian(q_u_0)  # compute underact jacobian (shape) as function of actuated coordinates (no load condition, CC jacobian from feedforward shape estimation)
        # J_u = self.ComputeFullJacobian(q_u_0)
        M_u, C_u, D_u, K_u = self.underactuated_dynamics_matrices(q_a) # compute the dynamics matrices for the system inflate to state q_a
        
        # print('Joint_forces: {}' .format(J_u.T @ F_ext))
        
        q_f = np.linalg.inv(K_u + D_u/dt) @ (J_u.T @ F_ext + D_u @ q_f_prev / dt) 
        q_u = q_u_0 + q_f

        q_u = self.opt.solver(q_u, self.Compute_tip_pose, T_tip)
        
        return q_u, q_f, q_u_0

    def underactuated_dynamics_matrices(self, q_a):
        # Set to be diagonal approximations or fit from data
        M_u = np.array([0,0,0]*self.n)                  # mass/inertia matrix - assume neglible for now, compute intensive
        C_u = np.zeros([0,0,0]*self.n)                 # Coriolis- assumed negligible
        # D_u = np.diag([.1, .3, .05]*self.n) * 0.1       # Damping - set as fucntion of q_a (should make sys slightly underdamped), but constant for now
        # K_u = np.diag([0.03, 1, 1, 0.03, 1, 1 , 0.03, 1, 1])          # stiffness - set as a function of q_a (linearly scales with chamber length for now)
        # K_u = np.diag([.02, .04, 1, .02, .3, 1 ,.02, .3, 1]) * np.mean(q_a + self.base_length)/40

        # K_u = np.diag([.25*self.n, 0.8333*self.n, 0.3333*self.n] * self.n)
        K_u = np.diag([.25*self.n, .8*self.n, 0.33*self.n] * self.n)
        # K_u = np.diag([.75, 2.5, 1]*self.n)  
        D_u = K_u / 20

        self.K_u = K_u
        self.D_u = D_u
        
        return M_u, C_u, D_u, K_u,

    
    #TODO: investigate effect of mass matrix, prove correctness of mass matrix
    def compute_mass_matrix(self, q_a, u_vol):
        rho_water = 1  # g/mL
        chamber_mass = 0.16 # g

        M_i = ( chamber_mass + rho_water * u_vol)/1000 # kg mass from chambers material and water volume
        M = np.diag(M_i) # diagonal mass matrix

        return M


    def actuated_dynamics_matrices(self, q_a, d_q_a, u_vol):
        # Set to be diagonal approximations or fit from data
        M_a = self.compute_mass_matrix(q_a, u_vol)  # mass/inertia matrix - set dependent to volume in chamber
        C_a = np.zeros((3, 3))                 # Coriolis- assumed negligible
        # D_a = np.diag([0.0008, 0.0008, 0.0008])         # Damping - found experimetally by best fit
        # D_a = np.diag([0.006]*3)
        # D_a = np.diag([0.00012]*3)
        # D_a = np.diag([0.0012]*3)
        D_a = np.diag([0.0014]*3)
        # D_a = np.diag([ 0.002 if d_q >= 0 else 0.003 for d_q in d_q_a])


        K_a = np.diag([0.022, 0.022, 0.022])      # stiffness - set from linear fit between u_vol (mL) and ell (mm)
        A_a = np.eye(3)                        # input matrix - set to eye as defined by collocated control
        
        return M_a, C_a, D_a, K_a, A_a


    def compute_control_dynamics(self, q_a, d_q_a, q_a_des, d_q_a_des, d_d_q_a_des, u_vol):
        # Get dynamics matrices from current configuartion 
        M_a, C_a, D_a, K_a, A_a = self.actuated_dynamics_matrices(q_a, d_q_a, u_vol)
        # print(M_a)
        # print(d_d_q_a_des)

        M_a = np.zeros((3,3))

        # Compute commanded volume input from desired configuration
        v_cmd = np.linalg.inv(A_a) @ (
            M_a @ d_d_q_a_des + C_a @ d_q_a_des + D_a @ d_q_a_des + K_a @ q_a_des
        )
        return v_cmd
    

    def inverse_kinematics_force_2(self, error, tip_vel, goal_vel, q_a, q_u, R_wall, F_tip, F_des,dt):
        ''' Compute q_a desired to reach goal_coords from current tip_coords using
            the current robot shape for jacobian estimation'''
        
        normal_vec = -R_wall[:,2]
        F_tip = -F_tip
        f_des_vel = (np.linalg.norm( F_des ) - self.f_des_prev)/dt
        self.f_des_prev = np.linalg.norm( F_des )

    
        normal_basis     = np.outer(normal_vec, normal_vec)
        tangential_basis = np.eye(3) - normal_basis 

        e_t = tangential_basis@error       #tangential error
        e_n = float(normal_vec@ error)     #normal_error (scalar)


        J_a, J_u = self.Actuated_Jacobian(q_u)       #|  Actuated Jacobian and its inverse
        J_ac_inv = self.Matrix_inverse(J_a)  #|

    

        #normal direction control
        # f_ext = float(normal_vec @ F_tip)
        # if self.sigma > 0.7:
        #     self.Efi = self.Efi + error_f*dt

        f_ext = np.linalg.norm(F_tip)
        error_f  = F_des - f_ext
        

        speed_tip = float(normal_vec @ tip_vel)
        speed_wall_n = normal_vec @ goal_vel
        speed_wall_t = tangential_basis @ goal_vel


        # using global vars
        # v_f = self.kf * error_f 
        # v_n = self.kp*e_n     

        # testing local vars for dynamic case
        # v_f = 400 * error_f + 2.5*speed_wall_n
        # v_n = 2 * e_n       + 2.5*speed_wall_n
        # for static sinusoidal force
        # v_f = 150 * error_f + 200*f_des_vel
        # v_n = 2 * e_n       + 0*speed_wall_n

        # testing dynanic finger
        # v_f = 200 * error_f + .5*speed_wall_n
        # v_n = 1 * e_n       + .5*speed_wall_n

        # if f_ext <= 0.02 and abs(normal_vec.T @ tip_vel) < 0.1:
        #     self.b = (1 - self.alpha)*self.b + self.alpha * f_ext
         
        

        # print('F_tip: {}' .format(F_tip))
        # print('f_ext: {}' .format(f_ext))
        # print('error_f: {}' .format(error_f))
        # print('e_n: {}' .format(e_n))
        # print('e_t: {}' .format(e_t))
        # print('v_n: {}' .format(v_n))
        # print('v_f: {}' .format(v_f))
        # print('normal: {}' .format(normal_vec))
        # print('sigma: {}' .format(self.sigma))

        M = .0006
        B =  0.0033

        #Admittance Control
        v_f = self.v_f_prev + (1/M)*dt*(error_f - B*self.v_f_prev)
        self.v_f_prev = v_f

        print(self.v_f_prev)

        #normal position control
        v_n = 2 * e_n


        # print('R_wall: {}' .format(R_wall))
        F  = e_n
        Fc = .7
        
        self.sigma = 1 / (1 + math.exp(-10 * (Fc - F)))


        # calc normal force vec using sigmoid to transition between admittance and pos error
        V_n = normal_vec*((1 - self.sigma)*v_n + self.sigma*v_f)      #final normal direction control


        #tangential direction control
        V_t = 1*e_t + 0*speed_wall_t

        #desired control
        V_des = (1 - self.sigma)*V_t + V_n + 2*goal_vel

        V_des = V_des.flatten()

        #control inputs
        q_a_dot = J_ac_inv @ V_des
        del_q_a = dt*q_a_dot
        q_a_des = q_a + del_q_a

        # print("vdes:", V_des)


        # return q_a_des, del_q_a, np.array([self.sigma, 0, e_n])
        return q_a_des, del_q_a, error
    
    


    def inverse_kinematics_force_on_path(self, error, tip_vel, goal_vel, q_a, q_u, R_wall, F_tip, F_des, dt):
        ''' Compute q_a desired to reach goal_coords from current tip_coords using
            the current robot shape for jacobian estimation'''
        

        """Notice that I am overriding F_des here"""
        F_des = 0.1


        normal_vec = -R_wall[:,2]
        F_tip = -F_tip
    
        normal_basis     = np.outer(normal_vec, normal_vec)
        tangential_basis = np.eye(3) - normal_basis 

        e_t = tangential_basis@error       #tangential error
        e_n = float(normal_vec@ error)     #normal_error (scalar)


        J_a, J_u = self.Actuated_Jacobian(q_u)       #|  Actuated Jacobian and its inverse
        J_ac_inv = self.Matrix_inverse(J_a)  #|


        #normal direction control
        # f_ext = float(normal_vec @ F_tip)
        f_ext = np.linalg.norm(F_tip)
        error_f  = F_des - f_ext


        # speed_tip = float(normal_vec @ tip_vel)
        goal_vel_n = normal_vec @ goal_vel
        goal_vel_t = tangential_basis @ goal_vel

        # testing local vars
        # v_f = 200 * error_f + 0.*speed_wall_n
        # v_n = 2 * e_n       + 0.*speed_wall_n
        
        v_f = 10 * error_f 
        v_n = 2 * e_n       


        # calc sigma value from error
        e_now  = e_n
        e_set = 0.5
        self.sigma = 1 / (1 + math.exp(-10 * (e_set - e_now)))
        # self.sigma = 0


        V_n = normal_vec*((1 - self.sigma)*v_n + self.sigma*v_f)      #final normal direction control


        #tangential direction control
        # V_t = 3*e_t + 1*speed_wall_t
        V_t = 7*e_t 
        #desired control
        V_des = V_t + V_n + 2* goal_vel
        # V_des = (1 - self.sigma)*V_t + V_n


        #control inputs
        q_a_dot = J_ac_inv @ V_des
        del_q_a = dt*q_a_dot
        q_a_des = q_a + del_q_a

        # print("vdes:", V_des)


        return q_a_des, del_q_a, np.array([self.sigma, 0, 0])
        # return q_a_des, del_q_a, error

    # This is Leo's function
    # def inverse_kinematics_force(self, tip_coords, tip_vel, goal_coords, goal_vel, q_a, q_u, F_mes, F_des, t_wall, R_wall):
    #     ''' Compute q_a desired to reach goal_coords from current tip_coords using
    #         the current robot shape for jacobian estimation'''
    #     J_a = self.Actuated_Jacobian(q_u)
    #     J_a_dag = np.linalg.pinv(J_a)
    #     J_u = self.ComputeJacobian(q_u)
    #     _,_,_,K_u = self.underactuated_dynamics_matrices(q_a)
    #     K_c_inv = self.task_stiffness(J_u, K_u)

    #     # 1) Plane normal unit vector
    #     n = R_wall[:,2]

    #     # 2) Tangential unit vector
    #     P_t = np.eye(3) - np.outer(n, n)
        
    #     # 2) normal force measured and desired
    #     f_n_mes = n @ F_mes
    #     f_n_des = F_des

    #     # 3) Normal position
    #     x_n = n @ tip_coords
    #     v_n = n @ tip_vel
    #     # x_n_des = x_n + (f_n_mes - f_n_des) / (n @ K_c_inv)
    #     v_n_des = n @ goal_vel

    #     # 4) Tangent pos and vel
    #     x_t = P_t @ tip_coords
    #     v_t = P_t @ tip_vel
    #     x_t_des = P_t @ goal_coords
    #     v_t_des = P_t @ goal_vel

    #     # 5) normal velocity command
    #     v_n_cmd = 1 * (f_n_mes - f_n_des)
    #     # v_n = 0.1 * (x_n_des - x_n) - 0.1 * v_n

    #     # 6) tangent velocity command
    #     e_t = (x_t_des - x_t)
    #     v_t_cmd = 1 * e_t

    #     # print("n:", n)
    #     print("P_t:", P_t)
    #     # print("v_n:", v_n)
    #     print("v_t:", v_t)
    #     # print("f_n_des:", f_n_des)
    #     # print("f_n_mes:", f_n_mes, flush=True)

    #     # 7) joint velocities del_q_a
    #     # del_q_a =  4*J_a_dag @ (n * v_n_cmd) + 0.1*J_a_dag @ (n * v_n_des) + 0.2*J_a_dag @ v_t_cmd + 0.1*J_a_dag @ (v_t_des)
    #     del_q_a =  4*J_a_dag @ (n * v_n_cmd) + 0.2*J_a_dag @ v_t_cmd 


    #     # del_q_a = .065* J_a_dag @ K_c_inv @(- F_des + F_tip) 
    #     # del_q_a = .2* J_a_dag @(- F_des + F_mes) + .01* J_a_dag @ error 
        
    #     q_a_des = q_a + del_q_a

    #     return q_a_des, del_q_a, e_t
    
    # def inverse_kinematics_force(self, error, tip_vel, q_a, q_u, F_tip, F_des):
    #     ''' Compute q_a desired to reach goal_coords from current tip_coords using
    #         the current robot shape for jacobian estimation'''
    #     J_a = self.Actuated_Jacobian(q_u)
    #     J_a_dag = np.linalg.pinv(J_a)

    #     J_u = self.ComputeJacobian(q_u)

    #     _,_,_,K_u = self.underactuated_dynamics_matrices(q_a)

    #     K_c_inv = self.task_stiffness(J_u, K_u)


    #     # del_q_a = .065* J_a_dag @ K_c_inv @(- F_des + F_tip) 
    #     del_q_a = .3* J_a_dag @(- F_des + F_tip) + .01* J_a_dag @ error 
    #     q_a_des = q_a + del_q_a

    #     return q_a_des, del_q_a
    

    def generate_circle(self, radius, z_height, num_points=100):
        """ Generate points for a circle parallel to xy plane"""
        # Generate angle values between 0 and 2pi using numpy linspace
        thetha_v = np.linspace(0, 2 * np.pi, num_points)
        # Parametric equations for the circle in 3D space
        circleCoords = np.zeros((3,num_points))

        circleCoords[0,:] = radius * np.cos(thetha_v)
        circleCoords[1,:] = radius * np.sin(thetha_v)

        # circleCoords = self.rotate_axis(circleCoords, -30)

        circleCoords[2,:] += z_height  # Z is constant
        
        return circleCoords
    
    def inverse_kinematics_force(self, error, tip_vel, goal_vel, q_a, q_u, R_wall, F_tip, F_des, f_des_vel, dt):
        ''' Compute q_a desired to reach goal_coords from current tip_coords using
            the current robot shape for jacobian estimation'''
        
        # dt = 1/33
        
        normal_vec = -R_wall[:,2]
        F_tip = -F_tip
        # f_des_vel = (np.linalg.norm( F_des ) - self.f_des_prev)/dt
        # self.f_des_prev = np.linalg.norm( F_des )

    
        normal_basis     = np.outer(normal_vec, normal_vec)
        tangential_basis = np.eye(3) - normal_basis 

        e_t = tangential_basis@error       #tangential error
        e_n = float(normal_vec@ error)     #normal_error (scalar)


        J_a, J_u = self.Actuated_Jacobian(q_u)       #|  Actuated Jacobian and its inverse
        J_ac_inv = self.Matrix_inverse(J_a)  #|
        # J_ac_inv = np.linalg.pinv(J_a) # test if this more efficient than above eqtn
        J_u_inv = np.linalg.pinv(J_u)

        # K_c = J_u @ self.K_u @ J_u.T
        # K_n = normal_vec @ K_c

        K_c = J_u_inv.T @ self.K_u @ J_u_inv
        K_n = np.linalg.norm( normal_vec @ K_c )

        print("K_n:", K_n)
        # print(K_c, flush=True)
    

        # print("shape J_a:", np.shape(J_a))
        # print("shape J_u:", np.shape(J_u))
        # print("shape K_c:", np.shape(K_c))
        print("",flush=True)
    

        #normal direction control
        # f_ext = float(normal_vec @ F_tip)
        # if self.sigma > 0.7:
        #     self.Efi = self.Efi + error_f*dt

        f_ext = np.linalg.norm(F_tip)
        error_f  = F_des - f_ext
        
        self.error_f_integral = self.error_f_integral + error_f*dt


        

        speed_tip = float(normal_vec @ tip_vel)
        speed_wall_n = normal_vec @ goal_vel
        speed_wall_t = tangential_basis @ goal_vel


        tip_vel_n = normal_vec @ tip_vel

        # testing local vars for dynamic case
        # v_f = .6 * 1/K_n * error_f/dt  + 4 * speed_wall_n 
        # v_f =   2 * 1/K_n * (error_f) /dt  + 3 * speed_wall_n 
        # v_n =   0.3* e_n / dt               + 3 * speed_wall_n


        # v_f =   0.98 * 1/K_n * (error_f) /dt   + 5.0* speed_wall_n 
        # v_n =   0.2 * e_n / dt               + 5* speed_wall_n 


        # # for static force
        v_f = .3 * 1/K_n*error_f/dt + 0.05 * 1/K_n * f_des_vel  #+ 0*0.098*self.error_f_integral
        v_n = 0.2 * e_n / dt       + 0 * 0.1* 1/K_n * f_des_vel
        V_t = 0.2 *e_t / dt + 0*speed_wall_t

        # # for dynamic force
        # v_f = .3 * 1/K_n*error_f/dt + 0.05 * 1/K_n * f_des_vel  #+ 0*0.098*self.error_f_integral
        # v_n = 0.2 * e_n / dt       + 0 * 0.1* 1/K_n * f_des_vel
        # V_t = 0.2 *e_t / dt + 0*speed_wall_t


        # testing dynanic finger
        # v_f = 200 * error_f + .5*speed_wall_n
        # v_n = 1 * e_n       + .5*speed_wall_n
        


        #tangential direction control
        # V_t =   0.2 *e_t / dt + 5*speed_wall_t


        # print('F_tip: {}' .format(F_tip))
        # print('f_ext: {}' .format(f_ext))
        # print('error_f: {}' .format(error_f))
        # print('e_n: {}' .format(e_n))
        # print('e_t: {}' .format(e_t))
        # print('v_n: {}' .format(v_n))
        # print('v_f: {}' .format(v_f))
        # print('normal: {}' .format(normal_vec))
        # print('sigma: {}' .format(self.sigma))

        # print('R_wall: {}' .format(R_wall))
        F  = e_n
        Fc = 1
        
        self.sigma = 1 / (1 + math.exp(-13*(Fc - F)))

        print('sigma: {}' .format(self.sigma))


        # calc normal force vec using sigmoid to transition between admittance and pos error
        V_n = normal_vec*((1 - self.sigma)*v_n + self.sigma*v_f -(1 - 0.5*self.sigma)*0.12*tip_vel_n)       #final normal direction control


        #desired control
        V_des = (1 - self.sigma)*V_t + V_n 
        # V_des = (1 - self.sigma)*V_t + V_n - (1-self.sigma)*0.1*tip_vel

        # V_des = V_t + V_n

        V_des = V_des.flatten()

        #control inputs
        q_a_dot = 1.0*J_ac_inv @ V_des
        del_q_a = dt*q_a_dot
        q_a_des = q_a + del_q_a

        # print("vdes:", V_des)


        # return q_a_des, del_q_a, np.array([self.sigma, 0, e_n])
        return q_a_des, del_q_a, np.array([error_f,0,0])
    
    def inverse_kinematics(self, error, goal_vel, q_a, q_u):
        ''' Compute q_a desired to reach goal_coords from current tip_coords using
            the current robot shape for jacobian estimation'''
        J_a, J_u = self.Actuated_Jacobian(q_u)
        J_a_dag = np.linalg.pinv(J_a)

        # error :   2.88    2.51    
        # ff    :   .1      .15     
        if self.hasRechedStart:
            del_q_a = 0.6* J_a_dag @ error + 1*.1*J_a_dag @ (goal_vel)
            
        else:
            del_q_a = .1* J_a_dag @ error + 0*.1*J_a_dag @ (goal_vel)

        q_a_des = q_a + del_q_a
        return q_a_des, del_q_a
    
    def CLIK(self, robot_data, tip_coords, R_tip, t_wall, R_wall, force, path_coords, period):
        ''' M_a @ d_d_q_a + D_a @ d_q_a + K_a @ q_a = A @ u
        
            where 
            q_a: actuated config vars, CHANGE chamber lengths from base length (mm) (maybe consider as change in chamber lengths ell_i: ell = ell_0 + ell_i)
            u: control inputs, volume (mL)
            A: Input mapping (volume -> chamber length) use I (eye) for collocated control?
            
            M_u @ d_d_q_u(q_a) + D_u @ d_q_u(q_a) + K_u @ q_u(q_a) = F_ext'''
        
        num_points = np.shape(path_coords)[1]

        T_tip = np.eye(4)
        T_tip[:3,:3] = R_tip
        T_tip[:3,3] = tip_coords

        time_now = time.time()
        dt = time_now - self.time_prev
        self.time_prev = time_now
        
        # get system vars in order
        # goal_coords = t_wall
        
        # goal_vel = (goal_coords - self.goal_prev) / dt
        # self.goal_prev = goal_coords

        # goal_coords = path_coords[:, self.i_goal]

        if np.linalg.norm(tip_coords - np.array([0,0,29])) < 0.6 and not self.hasRechedStart:
            self.hasRechedStart = True
            print("==========STARTED PATH==========")
            self.T_0 = time_now
            

        if self.hasRechedStart:
            goal_coords = np.array([5*np.sin(2*np.pi*(time_now - self.T_0) / (period/2) ),
                                    10*np.sin(2*np.pi*(time_now - self.T_0) / period),
                                    25 + 4*np.cos(2*np.pi*(time_now - self.T_0) / (period/2) )])
            
            goal_vel = np.array([2*np.pi/ (period/2)    *  5*np.cos(2*np.pi*(time_now - self.T_0) / (period/2) ),
                                2*np.pi/ period         *  10*np.cos(2*np.pi*(time_now - self.T_0) / period),
                                -2*np.pi/ (period/2)    *  4*np.sin(2*np.pi*(time_now - self.T_0) / (period/2) )])
        
        else:
            goal_coords = np.array([0,0,29])
            goal_vel = np.array([0,0,0])
        
        

        

        # 0. Collect force data and system data
        # F_tip = -R_tip @ force  # transform to get force acting on tip in global frame
        F_tip = force
        f_n_des = 0.12
        f_des_vel = 0

        P = 10
        # f_n_des = 0.08 + 0.04*np.sin(2*np.pi*time_now/ P )
        # f_des_vel = 2*np.pi/ P  * 0.04*np.cos(2*np.pi*time_now/ P )
        


        # F_des = R_wall @ np.array( [0,0,f_n_des])
        F_des = np.array( [0,0,f_n_des] )

        

        pressures = np.array(robot_data[:3])
        volumes = np.array(robot_data[3:6])/1000 # convert from
        M_u = np.zeros((9,9))                  # mass/inertia matrix - assume neglible for now, compute intensive uL to mL
        # print(pressures, volumes)
        commands = [0,0,0,0]

        # 1.a Compute actuated config vars, q_a(volumes), best prediction using dynamic equations
        q_a, d_q_a = self.compute_q_a_dynamics(volumes, self.q_a_prev, self.d_q_a_prev, dt) # computes current actuated config var: chamber lenghts, ell
        self.q_a_prev = q_a
        self.d_q_a_prev = d_q_a
        M_u = np.zeros((9,9))                  # mass/inertia matrix - assume neglible for now, compute intensive
        
        # 1.b Compute under-actuated config vars, q_u(q(a), F_ext), prediction based on dynamics
        q_u, q_f, q_u_0 = self.compute_q_u_dynamics_2(q_a, F_tip, self.q_f_prev, dt, T_tip) # compute under-actuated config vars using 
        self.q_f_prev = q_f
        self.q_u_0_prev = q_u_0

        # TODO: compute IK for q_a
        # 2.a Compute Inverse kinematics to solve for desired actuated config vars q_a_des
        error = goal_coords - tip_coords
        tip_vel = (tip_coords - self.tip_prev) / dt
        self.tip_prev = tip_coords
        u_volumes = volumes.copy()
        
        # e_t = np.zeros(3)
        # e_t = error
        q_a_des, d_q_a_des, d_d_q_a_des = np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0])

        # 2.b given current shape, def by q_a and q_u, find the q_a_des to minimize tip error

        # This is shresth function
        q_a_des, d_q_a_des = self.inverse_kinematics(error, goal_vel, q_a, q_u)
        # q_a_des, d_q_a_des, e_t = self.inverse_kinematics_force(error, tip_vel, goal_vel, q_a, q_u, R_wall, F_tip, f_n_des, f_des_vel, dt)
        # q_a_des, d_q_a_des, e_t = self.inverse_kinematics_force_2(error, tip_vel, goal_vel, q_a, q_u, R_wall, F_tip, f_n_des, dt)
        # q_a_des, d_q_a_des, e_t = self.inverse_kinematics_force_on_path(error, tip_vel, goal_vel, q_a, q_u, R_wall, F_tip, f_n_des, dt)


        # 3.a Compute inverse dynamics
        u_volumes = self.compute_control_dynamics(q_a, d_q_a, q_a_des, d_q_a_des, d_d_q_a_des, volumes)
        u_volumes = np.clip(u_volumes, 0, .55)*1000 # convert from mL to uL

        backbone_pred, rigid_pred, tip_pred = self.Compute_Soft_Curve(q_u)
        


        # if np.linalg.norm(error) < 1:
        #     self.i_goal = (self.i_goal + 1) % num_points
        #     print("goal reached at:", goal_coords)
        #     print("Tip position", tip_coords)

        
        # return u_volumes, commands, backbone_pred, rigid_pred, tip_pred, dt, q_a, q_u_0, F_des, e_t, goal_coords
        return u_volumes, commands, backbone_pred, rigid_pred, tip_pred, dt, q_a, q_u_0, F_des, error, goal_coords





def CLIK(frame, path_line, back_mark, force_mark, kin, path_coords):
    ''' M_a @ d_d_q_a + C_a @ d_q_a + K_a @ q_a = A @ u
        where 
        q_a: actuated config vars, CHANGE chamber lengths from base length (mm) (maybe consider as change in chamber lengths ell_i: ell = ell_0 + ell_i)
        u: control inputs, volume (mL)
        A: Input mapping (volume -> chamber length) use I (eye) for collocated control?
        
        M_u @ d_d_q_u(q_a) + C_u @ d_q_u(q_a) + K_u @ q_u(q_a) = F_ext'''
    
    global i_goal, base_length, u_volumes, tip_coords, tip_prev, time_now, time_prev
    global q_a_prev, d_q_a_prev, q_u_prev, d_q_u_prev, q_f_prev

    num_points = np.shape(path_coords)[1]

    time_now = time.time()
    dt = time_now - time_prev
    time_prev = time_now
    
    # get system vars in order
    goal_coords = path_coords[:, i_goal]

    # 0. Collect force data and system data
    # force = sensor.receive_data() # get force data in local frame
    # F_tip = force @ R_tip # transform to get force acting on tip in global frame
    Fz = 6.6/1000*9.81
    F_tip = np.array([0,0,-Fz])

    # F_tip = np.array([0,0, 0.1 * np.sin(time_now * 2*np.pi * 1/4)])


    # data = arduino.receive_data()
    # volumes = data[0:3]
    # pressures = data[3:]
    volumes = u_volumes

    # 1.a Compute actuated config vars, q_a(volumes), best prediction using dynamic equations
    q_a, d_q_a = kin.compute_q_a_dynamics(volumes, q_a_prev, d_q_a_prev, dt) # computes current actuated config var: chamber lenghts, ell
    q_a_prev = q_a
    d_q_a_prev = d_q_a
    
    # 1.b Compute under-actuated config vars, q_u(q(a), F_ext), prediction based on dynamics
    q_u, q_f, q_u_0 = kin.compute_q_u_dynamics_2(q_a, F_tip, q_f_prev, dt) # compute under-actuated config vars using 
    q_f_prev = q_f

    # TODO: compute IK for q_a
    # 2.a Compute Inverse kinematics to solve for desired actuated confid vars q_a_des
    error = goal_coords - tip_coords
    tip_vel = (tip_coords - tip_prev) / dt
    q_a_des, d_q_a_des, d_d_q_a_des = np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0])

    if np.linalg.norm(error) < 1:
        i_goal = (i_goal + 1) % num_points
        print("goal reached at:", goal_coords)

    else:
        # given current shape, def by q_a and q_u, find the q_a_des to minimize tip error
        q_a_des, d_q_a_des = kin.inverse_kinematics(error, tip_vel, q_a, q_u) 

        # 3.a Compute inverse dynamics
        u_volumes = kin.compute_control_dynamics(q_a, d_q_a, q_a_des, d_q_a_des, d_d_q_a_des, u_volumes)
        u_volumes = np.clip(u_volumes, 0, .55)
        # print(u_volumes)

    backbone_pred, rigid_pred, tip_pred = kin.Compute_Soft_Curve(q_u)
    tip_prev = tip_coords
    tip_coords = tip_pred

    # backbone_pred = rigid_pred

    # print(backbone_pred)
    # print(tip_coords)
    # print()


    # 4. Plot backbone estimation
    path_line.set_data(path_coords[0,:], path_coords[1,:])
    path_line.set_3d_properties(path_coords[2,:])

    back_mark.set_data(rigid_pred[0,:], rigid_pred[1,:])
    back_mark.set_3d_properties(rigid_pred[2,:])

    f_scale = 100
    force_mark.set_data([tip_coords[0], tip_coords[0] + f_scale*F_tip[0]], [tip_coords[1], tip_coords[1] + f_scale*F_tip[1]])
    force_mark.set_3d_properties([tip_coords[2], tip_coords[2] + f_scale*F_tip[2]])


    return path_line, back_mark, force_mark 



def main():
    # global i_goal, base_length, u_volumes, tip_coords, time_now, time_prev
    # global q_a_prev, d_q_a_prev, q_u_prev, d_q_u_prev, q_f_prev



    kin = CollocatedDynamics(np.array([0,0,0]))

    # generate circle path offset in z, with radius r and paramtrized into n pts
    path_coords = kin.generate_circle(10, 20, 5)
    # PathCoords = kin.rotate_axis('y', PathCoords, 30)
    # print(path_coords)


    

    # Plotting and figure handles
    fig = plt.figure(figsize = (6,5))
    ax1 = fig.add_subplot(1,1,1, projection ='3d')


    path_line, = ax1.plot([], [], [], '--', color = 'tab:blue', label = 'path_coords', markersize = 5)
    back_mark,     = ax1.plot([], [], [], '-', color = 'tab:orange', label = 'PCC', linewidth = 3, )
    force_mark, = ax1.plot([], [], [], '-', color='cyan', linewidth=3, label='Force Vector')


    ax1.set_xlim([-25, 25])
    ax1.set_ylim([-25, 25])
    ax1.set_zlim([0,35])
    ax1.legend()
    ax1.set_xlabel('X-axis (mm)')
    ax1.set_ylabel('Y-axis (mm)')
    ax1.set_zlabel('Z-axis (mm)')
    ax1.grid(True)

    time_now = time.time()
    time_prev= time_now - 0.05



    anim = animation.FuncAnimation(
        fig, CLIK, fargs = (path_line, back_mark, force_mark, kin, path_coords),
                                     cache_frame_data= False, interval = 5, blit = True
    )

    # Function to stop/play the simulation using spacebar
    paused = {'flag': False}
    def on_press(event):
        if event.key == ' ' or event.key == 'space':
            if paused['flag']:
                anim.event_source.start()
                paused['flag'] = False
            else:
                anim.event_source.stop()
                paused['flag'] = True


    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()