# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

"""
    Extended Kalman Filter for MOT
"""

import numpy as np

from kalmanfilter.base_kalman import KF_Base
from utils.utils import norm_radian, norm_realative_radian


class KF_YAW(KF_Base):
    """
    kalman filter for yaw in traiking objects
    """

    def __init__(self, n=2, m=2, dt=0.1, P=None, Q=None, R=None, init_x=None, cfg=None):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=cfg)
        self.dt = dt
        self.JH = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x, dt=None):
        # State-transition function is identity
        dt = self.dt if dt is None else dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + dt * x[1]
        x_fil[0] = norm_radian(x_fil[0])
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        dt = self.dt
        F = np.matrix([[1.0, dt], [0.0, 1.0]])
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def step(self, z, dt=None):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        dt = self.dt if dt is None else dt
        # shape of x :[n,1]  only norm angle,not angle_ratio
        self.x[0][0] = norm_radian(self.x[0][0])
        self.x = self.f(self.x, dt)
        self.x[0][0] = norm_radian(self.x[0][0])

        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        hx = self.h(self.x)  # shape:(m,1)

        hx = norm_radian(hx).reshape(-1, 1)
        z = np.array(z).reshape(-1, 1)  # (m,1)
        z = norm_radian(np.array(z)).reshape(-1, 1)
        info_gain = z - hx
        info_gain = norm_realative_radian(info_gain).reshape(-1, 1)

        self.x += G * info_gain
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        self.x[0][0] = norm_radian(self.x[0][0])
        return np.array(self.x.reshape(self.n))


class KF_SIZE(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=None)
        self.dt = dt
        self.JH = np.matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[2]
        x_fil[1] = x[1] + self.dt * x[3]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        a13 = self.dt
        a14 = 0
        a23 = 0
        a24 = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14],
                [0.0, 1.0, a23, a24],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))


class EKF_CV(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=None)

        self.dt = dt
        self.m = m
        self.JH = np.matrix(np.eye(self.m, 4))
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[2]
        x_fil[1] = x[1] + self.dt * x[3]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        a13 = self.dt
        a14 = 0
        a23 = 0
        a24 = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14],
                [0.0, 1.0, a23, a24],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update(self, z,track_id):
        #print("cv : ")
        #print("z : ",z)
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q
        if(track_id==0):
            print("self.Q",self.Q)
            print("predict : ",self.x.flatten())
            print("predict P",np.diag(self.P))

        # Update -----------------------------------------------------
        S = self.H @ self.P @ self.H.T + self.R
        print("S =\n", S)
        print("det(S) =", np.linalg.det(S))
        print("S shape =", S.shape)

        S += 1e-6 * np.eye(S.shape[0])  # add epsilon to the diagonal
        S_inv = np.linalg.inv(S)
        G = self.P @ self.H.T @ S_inv       
        #G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()

        if(track_id==0):
            print("update : ",self.x.flatten())
            print("update P",np.diag(self.P))

        return np.array(self.x.reshape(self.n))

"""     def predict(self):
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return np.array(self.x.reshape(self.n)) """


class EKF_CA(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=cfg)

        self.dt = dt
        self.m = m
        self.JH = np.matrix(np.eye(self.m, 6))
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[2] + 0.5 * self.dt * self.dt * x[4]
        x_fil[1] = x[1] + self.dt * x[3] + 0.5 * self.dt * self.dt * x[5]
        x_fil[2] = x[2] + self.dt * x[4]
        x_fil[3] = x[3] + self.dt * x[5]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        a13 = self.dt
        a14 = 0
        a23 = 0
        a24 = self.dt
        dt = self.dt
        at2 = self.dt * self.dt * 0.5
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14, at2, 0],
                [0.0, 1.0, a23, a24, 0, at2],
                [0.0, 0.0, 1.0, 0.0, dt, 0],
                [0.0, 0.0, 0.0, 1.0, 0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        S = self.H @ self.P @ self.H.T + self.R
        S += 1e-6 * np.eye(S.shape[0])  # add epsilon to the diagonal
        S_inv = np.linalg.inv(S)
        G = self.P @ self.H.T @ S_inv
        #G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))

    def predict(self):
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return np.array(self.x.reshape(self.n))


class EKF_CTRA(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=cfg)

        self.dt = dt
        self.JH = np.matrix(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity
        # [x,y,yaw,v,w,a]

        x_fil = np.array(x)
        x_fil[0] = x[0] + (1 / x[4] ** 2) * (
            (x[3] * x[4] + x[5] * x[4] * self.dt) * np.sin(x[2] + x[4] * self.dt)
            + x[5] * np.cos(x[2] + x[4] * self.dt)
            - x[3] * x[4] * np.sin(x[2])
            - x[5] * np.cos(x[2])
        )
        x_fil[1] = x[1] + (1 / x[4] ** 2) * (
            (-x[3] * x[4] - x[5] * x[4] * self.dt) * np.cos(x[2] + x[4] * self.dt)
            + x[5] * np.sin(x[2] + x[4] * self.dt)
            + x[3] * x[4] * np.cos(x[2])
            - x[5] * np.sin(x[2])
        )
        x_fil[2] = x[2] + x[4] * self.dt
        x_fil[3] = x[3] + x[5] * self.dt
        return x_fil

    def getF(self, x):
        beta = x[4].item(0)
        beta_eps = beta if abs(beta) > 1e-3 else 1e-3
        x[4]=beta_eps
        # So state-transition Jacobian is identity matrix
        dt = self.dt
        a13 = (
            (
                -x[4] * x[3] * np.cos(x[2])
                + x[5] * np.sin(x[2])
                - x[5] * np.sin(dt * x[4] + x[2])
                + (dt * x[4] * x[5] + x[4] * x[3]) * np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)

        a14 = (
            (-x[4] * np.sin(x[2]) + x[4] * np.sin(dt * x[4] + x[2])) / x[4] ** 2
        ).item(0)

        a15 = (
            (
                -dt * x[5] * np.sin(dt * x[4] + x[2])
                + dt * (dt * x[4] * x[5] + x[4] * x[3]) * np.cos(dt * x[4] + x[2])
                - x[3] * np.sin(x[2])
                + (dt * x[5] + x[3]) * np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 2
            - 2
            * (
                -x[4] * x[3] * np.sin(x[2])
                - x[5] * np.cos(x[2])
                + x[5] * np.cos(dt * x[4] + x[2])
                + (dt * x[4] * x[5] + x[4] * x[3]) * np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 3
        ).item(0)

        a16 = (
            (
                dt * x[4] * np.sin(dt * x[4] + x[2])
                - np.cos(x[2])
                + np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)

        a23 = (
            (
                -x[4] * x[3] * np.sin(x[2])
                - x[5] * np.cos(x[2])
                + x[5] * np.cos(dt * x[4] + x[2])
                - (-dt * x[4] * x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)
        a24 = (
            (x[4] * np.cos(x[2]) - x[4] * np.cos(dt * x[4] + x[2])) / x[4] ** 2
        ).item(0)
        a25 = (
            (
                dt * x[5] * np.cos(dt * x[4] + x[2])
                - dt * (-dt * x[4] * x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2])
                + x[3] * np.cos(x[2])
                + (-dt * x[5] - x[3]) * np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 2
            - 2
            * (
                x[4] * x[3] * np.cos(x[2])
                - x[5] * np.sin(x[2])
                + x[5] * np.sin(dt * x[4] + x[2])
                + (-dt * x[4] * x[5] - x[4] * x[3]) * np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 3
        ).item(0)
        a26 = (
            (
                -dt * x[4] * np.cos(dt * x[4] + x[2])
                - np.sin(x[2])
                + np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)
        a35 = self.dt
        a46 = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14, a15, a16],
                [0.0, 1.0, a23, a24, a25, a26],
                [0.0, 0.0, 1.0, 0.0, a35, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, a46],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def update(self, z,track_id):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        # Update -----------------------------------------------------
        S = self.H @ self.P @ self.H.T + self.R
        print("F:\n", self.F)    
        print("Q:\n", self.Q)    

        print("H:\n", self.H)
        print("P:\n", self.P)
        print("R:\n", self.R)

        print("S =\n", S)
        print("det(S) =", np.linalg.det(S))
        print("S shape =", S.shape)
        S += 1e-6 * np.eye(S.shape[0])  # add epsilon to the diagonal
        S_inv = np.linalg.inv(S)
        G = self.P @ self.H.T @ S_inv
        #G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)    
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH


class EKF_RVBOX(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=cfg)

        self.dt = dt
        self.m = m
        self.JH = np.matrix(np.eye(self.m, 8))
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[4]
        x_fil[1] = x[1] + self.dt * x[5]
        x_fil[2] = x[2] + self.dt * x[6]
        x_fil[3] = x[3] + self.dt * x[7]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        F = np.matrix(
            [
                [1.0, 0.0, 0.0, 0.0, self.dt, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, self.dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, self.dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))
class WrappedCV:
    def __init__(self, ekf):
        self.model = ekf
        self.X = ekf.x
        self.P = ekf.P
        self.H = ekf.H
        self.R = ekf.R
        self.X_pre = None
        self.P_pre = None

    def filt(self, Z,track_id):
        #print("self.X_pre",self.X_pre)
        #print("self.model.x",self.model.x)
        self.model.x=self.X.reshape(4, 1)
        self.model.P=self.P
        self.X_pre = self.model.f(self.model.x)

        self.P_pre = self.model.getF(self.model.x) @ self.model.P @ self.model.getF(self.model.x).T + self.model.Q
        self.X = self.model.update(Z,track_id)
        self.P = self.model.P
        self.H = self.model.getH(self.X)
        self.R = self.model.R
    def assign_mixed_values(self):
        self.model.x=self.X
        self.model.P=self.P
                
class WrappedCA:
    def __init__(self, ekf):
        self.model = ekf
        self.X = ekf.x
        self.P = ekf.P
        self.H = ekf.H
        self.R = ekf.R
        self.X_pre = None
        self.P_pre = None

    def filt(self, Z):
        #print("self.X_pre",self.X_pre)
        #print("self.model.x",self.model.x)
        self.model.x=self.X.reshape(6, 1)
        self.model.P=self.P
        self.X_pre = self.model.f(self.model.x)

        self.P_pre = self.model.getF(self.model.x) @ self.model.P @ self.model.getF(self.model.x).T + self.model.Q
        self.X = self.model.update(Z)
        self.P = self.model.P
        self.H = self.model.getH(self.X)
        self.R = self.model.R
    def assign_mixed_values(self):
        self.model.x=self.X
        self.model.P=self.P        
class WrappedCTRA:
    def __init__(self, ekf):
        self.model = ekf
        self.X = ekf.x
        self.P = ekf.P
        self.H = ekf.H
        self.R = ekf.R
        self.X_pre = None
        self.P_pre = None

    def filt(self, Z,track_id):
        self.model.x=self.X.reshape(6, 1)
        self.model.P=self.P
        self.X_pre = self.model.f(self.model.x)
        self.P_pre = self.model.getF(self.model.x) @ self.model.P @ self.model.getF(self.model.x).T + self.model.Q
        self.X = self.model.update(Z,track_id)
        self.P = self.model.P
        self.H = self.model.getH(self.X)
        self.R = self.model.R        

    def assign_mixed_values(self):
        self.model.x=self.X
        self.model.P=self.P