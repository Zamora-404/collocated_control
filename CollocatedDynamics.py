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

n = 1           # number of segments
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

        self.n    = 1                   #Number of segments for the SBA
        self.d    = 4.75                #Geometric parameter chamber radial distance
        self.gain = 1                   #Gain for the U_ell
        self.r    = 7
        self.K = np.diag([15,15,.5]*self.n)
        self.D = np.diag([5,5,5]*self.n)
        self.I = np.diag((self.K.diagonal()**2)/(4*self.D.diagonal()))

        self.i_goal = 0
        self.base_length = 5

        self.tip_prev = np.array([0,0,self.base_length])
        self.time_now = time.time()
        self.time_prev = self.time_now - 0.05

        self.q_a_prev = u_init/1000 * 50 # convert from uL to mL teh convert to mm via linear fit
        self.d_q_a_prev = np.array([0,0,0])

        self.q_u_prev = self.q_no_load(self.q_a_prev + self.base_length) 
        self.q_u_0_prev = self.q_no_load(self.q_a_prev + self.base_length)
        self.q_f_prev = np.array([0,0,0]*self.n)

        self.opt = Q_Optimize()
        
        # print(self.q_a_prev,  self.q_u_prev )


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

    def ComputeJacobian(self, q):
        global r
        # Evaluating the final jacobian for the soft robot J(q). Its size is 3 x 3*n
        dh_aug = self.Augmented_DH(q)
        Jxi    = self.Compute_Augmented_Jacobian(dh_aug)
        _, Jm  = self.Build_XI(q)

        J_q    = Jxi @ Jm

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

    def Model(self, t, q, f_ext_fun, q_ref, Ei_prev):
        # Our basic dynamic model K(q - q_ref) + Dq_dot = J^T.F_ext

        K = self.K
        D = self.D
        I = self.I
        J  = self.ComputeJacobian(q)
        f  = f_ext_fun(t)
        diff_q_q_ref = self.q_diff(q, q_ref)

        Residual = J.T.dot(f) - K.dot(diff_q_q_ref) - I.dot(Ei_prev)
        dq = np.linalg.solve(D, Residual)
        return dq

    def new_Model(self, t, q, f_ext_fun,q_ref, q_ref_dot,Ei_prev):
        # Our basic dynamic model Kq + D(q_dot - q_ref_dot) = J^T.F_ext
        K = self.K
        D = self.D
        I = self.I
        J  = self.ComputeJacobian(q)
        f  = f_ext_fun(t)
        diff_q_q_ref = self.q_diff(q, q_ref)

        Residual = J.T.dot(f) - K.dot(diff_q_q_ref) - I.dot(Ei_prev)
        dq = q_ref_dot + np.linalg.solve(D, Residual)
        return dq

    def q_dynamics(self, q0, Ei_prev, f_ext_fun, t0, dt):
        # Function to evaluate the configuration variable (q) at t = t + dt under the action of
        # external force acting at the tip
        q0 = np.asarray(q0).reshape(-1)
        t_span = (t0, t0 + dt)

        sol = solve_ivp(fun = lambda t, q: self.Model(t, q, f_ext_fun, q0, Ei_prev),
                        t_span = t_span,
                        y0 = q0,
                        method= 'RK45',
                        t_eval = [t0 + dt])

        q_next = sol.y[:,-1]

        return q_next.reshape(-1,1)

    def q_dynamics_new(self, q_init, q_ref, q_ref_dot, Ei_prev, f_ext_fun, t0, dt):
        # Function to evaluate the configuration variable (q) at t = t + dt under the action of
        # external force acting at the tip
        q_init = np.asarray(q_init).reshape(-1)
        t_span = (t0, t0 + dt)

        sol = solve_ivp(fun = lambda t, q: self.new_Model(t, q, f_ext_fun, q_ref, q_ref_dot, Ei_prev),
                        t_span = t_span,
                        y0 = q_init,
                        method= 'RK45',
                        t_eval = [t0 + dt])

        q_next = sol.y[:,-1]

        return q_next.reshape(-1,1)

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



        # if np.isclose(ell_1, ell_2, atol = 0.05) and np.isclose(ell_2, ell_3, tol = 0.05) and np.isclose(ell_1, ell_3, atol = 0.05):
        #     theta = 0.1
        #     phi      = phi % (2*math.pi)
        #     prev_phi = prev_phi % (2*math.pi)

        #     eps = 0.01  # small tolerance for “near 0” or “near π”
        #     if abs(prev_phi) < eps and abs(phi - math.pi) < eps:
        #         hi = prev_phi
        #     elif abs(prev_phi - math.pi) < eps and abs(phi) < eps:
        #         phi = prev_phi
        #     else:
        #         # 3) compute minimal signed difference in (−π, π]
        #         diff = ((phi - prev_phi + math.pi) % (2*math.pi)) - math.pi

        #         # 4) if jump > 45° (0.785 rad), nudge by ±0.08 rad
        #         if abs(diff) > 0.785:
        #             phi = prev_phi + (0.08 if diff > 0 else -0.08)
        #             # re‐normalize if needed
        #             phi = phi % (2*math.pi)



        for i in range(self.n):
            idx = i*3
            q_0[idx] = phi
            q_0[idx+1] = theta/self.n
            q_0[idx+2] = S/self.n

        return q_0

    def Compute_Jacobian_Ell_Q(self, q):
        # Function to evaluate the jacobian relating change in Ell w.r.t change in q
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
        J_rq = self.ComputeJacobian(q)
        J_Eq = self.Compute_Jacobian_Ell_Q(q)

        J_Eq_inv = np.linalg.pinv(J_Eq)

        J_a = J_rq @ J_Eq_inv

        return J_a



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
        samples = 100
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
        M_a, C_a, D_a, K_a, A_a = self.actuated_dynamics_matrices(q_a_prev, volumes)
    

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
    
    def compute_q_a_dynamics_no_mass(self, volumes, q_a_prev, d_q_a_prev, dt):
        """
        Solves the ODE for q_a using the equation:
        M_a * ddq_a + D_a * dq_a + K_a * q_a = A_a * u
        """
        M_a, C_a, D_a, K_a, A_a = self.actuated_dynamics_matrices(q_a_prev, volumes)
        

        d_q_a = np.linalg.inv( D_a ) @ (A_a @ volumes - K_a @ q_a_prev) # solve kiento-satics problem with damp

        
        q_a = q_a_prev + d_q_a
        
        return q_a, d_q_a

    # def compute_q_u_dynamics(self, q_a, F_ext, q_f_prev, dt, T_tip = np.eye(4)): 
    #     q_u_0 = self.q_no_load(q_a + self.base_length)         # Compute no load underacttuated vars
    #     J_u  = self.ComputeJacobian(q_u_0)  # compute underact jacobian (shape) as function of actuated coordinates (no load condition, CC jacobian from feedforward shape estimation)
        
        
    #     M_u, C_u, D_u, K_u = self.underactuated_dynamics_matrices(q_a) # compute the dynamics matrices for the system inflate to state q_a
        
    #     # print(q_f_prev, K_u)
    #     joint_torques = J_u.T @ F_ext # compute joint torques using underact jacobian 
    #     elastic_torques = K_u @ q_f_prev 

    #     q_f = np.linalg.inv(K_u)@ (joint_torques)
    #     q_u = q_u_0 + q_f

    #     # print("joint torques:", joint_torques)

    #     # q_u_star = self.opt.solver(q_u, self.Compute_tip_pose, T_tip)
    #     # print(q_u)
    #     # print(q_u_star)
    
    #     return q_u, q_f, q_u_0
    
    def compute_q_u_dynamics_2(self, q_a, F_ext, q_f_prev, dt, T_tip = np.eye(4)): 
        # print(self.q_u_0_prev)
        # q_u_0 = self.q_u_0_prev
        q_u_0 = self.q_no_load(q_a + self.base_length)         # Compute no load underacttuated vars
        J_u  = self.ComputeJacobian(q_u_0)  # compute underact jacobian (shape) as function of actuated coordinates (no load condition, CC jacobian from feedforward shape estimation)
        M_u, C_u, D_u, K_u = self.underactuated_dynamics_matrices(q_a) # compute the dynamics matrices for the system inflate to state q_a
        
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
        K_u = np.diag([.2*self.n, 0.8333*self.n, 0.3333*self.n] * self.n)
        # K_u = np.diag([.75, 2.5, 1]*self.n)  
        D_u = K_u / 20
        
        return M_u, C_u, D_u, K_u,

    
    #TODO: investigate effect of mass matrix, prove correctness of mass matrix
    def compute_mass_matrix(self, q_a, u_vol):
        rho_water = 1  # g/mL
        chamber_mass = 0.16 # g

        M_i = ( chamber_mass + rho_water * u_vol)/1000 # kg mass from chambers material and water volume
        M = np.diag(M_i) # diagonal mass matrix

        return M


    def actuated_dynamics_matrices(self, q_a, u_vol):
        # Set to be diagonal approximations or fit from data
        M_a = self.compute_mass_matrix(q_a, u_vol)  # mass/inertia matrix - set dependent to volume in chamber
        C_a = np.zeros((3, 3))                 # Coriolis- assumed negligible
        D_a = np.diag([0.001, 0.001, 0.001])         # Damping - found experimetally by best fit
        K_a = np.diag([0.02, 0.02, 0.02])      # stiffness - set from linear fit between u_vol (mL) and ell (mm)
        A_a = np.eye(3)                        # input matrix - set to eye as defined by collocated control
        
        return M_a, C_a, D_a, K_a, A_a


    def compute_control_dynamics(self, q_a, d_q_a, q_a_des, d_q_a_des, d_d_q_a_des, u_vol):
        # Get dynamics matrices from current configuartion 
        M_a, C_a, D_a, K_a, A_a = self.actuated_dynamics_matrices(q_a, u_vol)
        # print(M_a)
        # print(d_d_q_a_des)

        M_a = np.zeros((3,3))

        # Compute commanded volume input from desired configuration
        v_cmd = np.linalg.inv(A_a) @ (
            M_a @ d_d_q_a_des + C_a @ d_q_a_des + D_a @ d_q_a_des + K_a @ q_a_des
        )
        return v_cmd

    def inverse_kinematics(self, error, tip_vel, q_a, q_u):
        ''' Compute q_a desired to reach goal_coords from current tip_coords using
            the current robot shape for jacobian estimation'''
        J_a = self.Actuated_Jacobian(q_u)
        J_a_dag = np.linalg.pinv(J_a)
        del_q_a = .4* J_a_dag @ error + .00*J_a_dag @ (-tip_vel)
        q_a_des = q_a + del_q_a
        return q_a_des, del_q_a
    



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
    
    def CLIK(self, robot_data, tip_coords, R_tip, force, path_coords):
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
        goal_coords = path_coords[:, self.i_goal]

        # 0. Collect force data and system data
        F_tip = -R_tip @ force  # transform to get force acting on tip in global frame
        F_tip = np.array([0,0,0])
        print(F_tip)

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
        
        q_a_des, d_q_a_des, d_d_q_a_des = np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0])


        # given current shape, def by q_a and q_u, find the q_a_des to minimize tip error
        q_a_des, d_q_a_des = self.inverse_kinematics(error, tip_vel, q_a, q_u) 

        # 3.a Compute inverse dynamics
        u_volumes = self.compute_control_dynamics(q_a, d_q_a, q_a_des, d_q_a_des, d_d_q_a_des, volumes)
        u_volumes = np.clip(u_volumes, 0, .55)*1000 # convert from mL to uL

        backbone_pred, rigid_pred, tip_pred = self.Compute_Soft_Curve(q_u)
        


        if np.linalg.norm(error) < 1:
            self.i_goal = (self.i_goal + 1) % num_points
            print("goal reached at:", goal_coords)


        return u_volumes, commands, backbone_pred, rigid_pred, tip_pred, dt, q_a, q_u_0 




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