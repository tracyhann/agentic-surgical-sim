"""dVRK PSM forward kinematics (modified DH, large needle driver), tool tip in the RCM frame."""
import numpy as np

# (alpha, a, theta_offset, d_offset, type)
DH = [(np.pi / 2, 0.0, np.pi / 2, 0.0, 'R'),
      (-np.pi / 2, 0.0, -np.pi / 2, 0.0, 'R'),
      (np.pi / 2, 0.0, 0.0, -0.4318, 'P'),
      (0.0, 0.0, 0.0, 0.4162, 'R'),
      (-np.pi / 2, 0.0, -np.pi / 2, 0.0, 'R'),
      (-np.pi / 2, 0.0091, -np.pi / 2, 0.0, 'R')]
TIP = np.array([[0, -1, 0, 0], [0, 0, 1, 0.0102], [-1, 0, 0, 0], [0, 0, 0, 1.0]])


def _mdh(alpha, a, theta, d):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -sa * d], [st * sa, ct * sa, ca, ca * d], [0, 0, 0, 1]])


def fk(q, upto=6):
    T = np.eye(4)
    for i, (al, a, th0, d0, kind) in enumerate(DH[:upto]):
        th, d = (th0 + q[i], d0) if kind == 'R' else (th0, d0 + q[i])
        T = T @ _mdh(al, a, th, d)
    return T @ TIP if upto == 6 else T


def tips(Q):
    return np.array([fk(q)[:3, 3] for q in Q])
