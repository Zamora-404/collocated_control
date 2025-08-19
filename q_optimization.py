import numpy as np
from scipy.optimize import minimize
import time



class Q_Optimize:
    def __init__(self):
        self.maxiter = 30
        self.max_step = .25
        self.tol = 1e-3
        self.W  = np.diag([1.0,1.0,1.0, 0.5, 0.5, 0.5])
        self.lam0 = 1e-2
        self.lam_decay = 0.3
        self.last_central = 2


    def se3_log(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
    # Rotation Part
        trace_R = np.clip(np.trace(R), -1.0, 3.0)

        #when theta ~= 0 deg
        if trace_R >= 3.0 - 1e-6:
            omega = 0.5 * np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])

        # When theta ~= 180 deg
        elif trace_R <= -1.0 + 1e-6:
            angle = np.pi
            k = np.argmax(np.diag(R))
            axis = np.sqrt(max(0, (R[k,k] + 1) / 2))
            omega = np.zeros(3)
            omega[k] = axis
            for i in range(3):
                if i != k:
                    omega[i] = R[k,i] / (2 * axis) if axis > 1e-8 else 0
            omega *= angle
        # General Case
        else:
            angle = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))
            sin_angle = np.sin(angle)
            omega = angle / (2 * sin_angle) * np.array([
                R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]
            ])

    # Translation Part
        theta = np.linalg.norm(omega)

        if theta < 1e-8:
            # Small angle: V^-1 ≈ I
            V_inv = np.eye(3)
        else:
            # General case: V^-1 formula
            omega_hat = self.skew_symmetric(omega)

            coefficient = (2*np.sin(theta) - theta*(1 + np.cos(theta))) / (2*theta**2*np.sin(theta))

            V_inv = np.eye(3) - 0.5*omega_hat + coefficient*(omega_hat @ omega_hat)

        rho = V_inv @ t

        return np.concatenate([rho, omega])

    def skew_symmetric(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def error_se_3(self, q, fk_fun, target_pose):
        R = target_pose[:3,:3]
        P = target_pose[:3, 3]
        target_pose_inv = np.eye(4)
        target_pose_inv[:3,:3] = R.T
        target_pose_inv[:3, 3] = -R.T @ P
        return self.se3_log(target_pose_inv@ fk_fun(q))


    def error_jacobian(self, q, fk_fun, target_pose, scheme = 'fwd'):
        r0 = self.error_se_3(q, fk_fun, target_pose)
        m,n = r0.size, q.size

        J = np.zeros((m,n))

        h = 1e-6*np.maximum(np.abs(q), 1.0)
        h = np.clip(h, 1e-8, 1e-3)


        if scheme == 'fwd':
            for i in range(n):
                qf = q.copy()
                qf[i] += h[i]
                J[:, i] = (self.error_se_3(qf, fk_fun, target_pose) - r0)/h[i]

        elif scheme == 'cen':
            for i in range(n):
                qf = q.copy()
                qp = q.copy()
                qf[i] += h[i]
                qp[i] -= h[i]
                J[:, i] = (self.error_se_3(qf, fk_fun, target_pose) - self.error_se_3(qp, fk_fun, target_pose) )/ (2*h[i])

        return r0, J


    def solver(self, q_init, fk_fun, target_pose):
        q = q_init.copy()
        lam = self.lam0

        for k in range(self.maxiter):
            scheme = 'cen' if (self.last_central > 0 and k >= self.maxiter - self.last_central) else 'fwd'

            r, J = self.error_jacobian(q, fk_fun, target_pose, scheme)

            if np.sqrt(r.T @ self.W @ r) < self.tol:
                break

            JT_W = J.T @ self.W
            H = JT_W @ J + lam*np.eye(J.shape[1])
            G = JT_W @ r

            try:
                dq = -np.linalg.solve(H, G)
            except np.linalg.LinAlgError:
                dq = -np.linalg.lstsq(H, G, rcond=None)[0]

            sn = np.linalg.norm(dq)
            if sn > self.max_step:
                dq *= (self.max_step / (sn))

            q_new = q + dq
            r_new = self.error_se_3(q_new, fk_fun, target_pose)

            F, F_new = float(r.T @ self.W @ r), float(r_new.T @ self.W @ r_new)

            if F_new < F:
                q, lam = q_new, max(lam * self.lam_decay, 1e-6)
            else:
                lam *= 2.0

            if sn < self.tol:
                break

        return q


def main():
 return None

if __name__ == '__main__':
    main()













