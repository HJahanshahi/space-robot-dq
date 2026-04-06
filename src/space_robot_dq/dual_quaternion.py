"""Dual quaternion mathematics for space robot pose representation."""
import numpy as np
import warnings
from scipy.spatial.transform import Rotation


class DualQuaternion:
    """
    Dual quaternion class for representing rigid body transformations.
    
    A dual quaternion consists of:
    - qr: real quaternion [w, x, y, z] (rotation)
    - qd: dual quaternion [w, x, y, z] (translation)
    """
    
    def __init__(self, qr=None, qd=None):
        if qr is None:
            self.qr = np.array([1.0, 0.0, 0.0, 0.0])
            self.qd = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.qr = np.array(qr, dtype=float).reshape(4)
            self.qd = np.array(qd, dtype=float).reshape(4)
        self.normalize()

    def normalize(self):
        """Normalize the dual quaternion - ensures unit length and orthogonality"""
        norm_r = np.sqrt(np.sum(self.qr**2))
        if norm_r > np.finfo(float).eps:
            self.qr = self.qr / norm_r
            self.qd = self.qd / norm_r
            dot_prod = np.dot(self.qr, self.qd)
            if abs(dot_prod) > np.finfo(float).eps:
                self.qd = self.qd - dot_prod * self.qr

    def __mul__(self, other):
        """Dual quaternion multiplication"""
        qr = quaternion_multiply(self.qr, other.qr)
        qd = quaternion_multiply(self.qr, other.qd) + quaternion_multiply(self.qd, other.qr)
        result = DualQuaternion(qr, qd)
        result.normalize()
        return result

    def multiply(self, other):
        """Alias for multiplication"""
        return self.__mul__(other)

    def conjugate(self):
        """Dual quaternion conjugate"""
        qr = np.array([self.qr[0], -self.qr[1], -self.qr[2], -self.qr[3]])
        qd = np.array([self.qd[0], -self.qd[1], -self.qd[2], -self.qd[3]])
        return DualQuaternion(qr, qd)

    def to_matrix(self):
        """Convert dual quaternion to 4x4 transformation matrix"""
        R = self.to_rotation_matrix()
        p = self.get_translation()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def to_rotation_matrix(self):
        """Convert quaternion to 3x3 rotation matrix"""
        w, x, y, z = self.qr
        R = np.array([
            [1 - 2*y**2 - 2*z**2,  2*x*y - 2*w*z,        2*x*z + 2*w*y],
            [2*x*y + 2*w*z,        1 - 2*x**2 - 2*z**2,  2*y*z - 2*w*x],
            [2*x*z - 2*w*y,        2*y*z + 2*w*x,        1 - 2*x**2 - 2*y**2]
        ])
        return R

    def get_translation(self):
        """Extract translation vector from dual part."""
        qr_conj = np.array([self.qr[0], -self.qr[1], -self.qr[2], -self.qr[3]])
        result = 2.0 * quaternion_multiply(self.qd, qr_conj)
        return result[1:4]

    def to_pose(self):
        """Extract pose components (rotation matrix and position vector)"""
        R = self.to_rotation_matrix()
        p = self.get_translation()
        return R, p

    @staticmethod
    def from_screw(theta, d, l, m):
        """
        Create dual quaternion from screw parameters.
        
        Args:
            theta: rotation angle (rad)
            d: translation along axis (pitch * theta)
            l: unit direction vector (3,)
            m: moment vector (3,)
        """
        half_theta = theta / 2.0
        sin_ht = np.sin(half_theta)
        cos_ht = np.cos(half_theta)
        
        qr = np.array([cos_ht, l[0]*sin_ht, l[1]*sin_ht, l[2]*sin_ht])
        qd = np.array([
            -d/2.0 * sin_ht,
            l[0]*d/2.0*cos_ht + m[0]*sin_ht,
            l[1]*d/2.0*cos_ht + m[1]*sin_ht,
            l[2]*d/2.0*cos_ht + m[2]*sin_ht
        ])
        
        obj = DualQuaternion(qr, qd)
        obj.normalize()
        return obj

    @staticmethod
    def from_pose(R, p):
        """
        Convert rotation matrix and position to dual quaternion.
        
        Args:
            R: 3x3 rotation matrix
            p: 3D position vector
        """
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        qr = np.array([w, x, y, z])
        qd = 0.5 * quaternion_multiply(np.array([0.0, p[0], p[1], p[2]]), qr)
        
        return DualQuaternion(qr, qd)


def quaternion_multiply(q1, q2):
    """
    Quaternion multiplication (Hamilton product).
    
    Args:
        q1, q2: quaternions [w, x, y, z]
    
    Returns:
        product quaternion [w, x, y, z]
    """
    q1 = np.array(q1).reshape(4)
    q2 = np.array(q2).reshape(4)
    
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def log_dq(dq):
    """
    Compute logarithm of dual quaternion (spatial velocity twist).
    
    Args:
        dq: DualQuaternion object
    
    Returns:
        xi: 6D spatial velocity [rotation; translation]
    """
    qr_w = dq.qr[0]
    qr_v = dq.qr[1:4]
    qd_w = dq.qd[0]
    qd_v = dq.qd[1:4]
    
    theta = 2.0 * np.arccos(np.clip(qr_w, -1.0, 1.0))
    
    if abs(theta) > np.pi:
        warnings.warn('Rotation angle outside stable range [-π,π].')
        theta = np.pi * np.sign(theta)
    
    if abs(theta) < 1e-10:
        xi = np.zeros(6)
        xi[3:6] = 2.0 * qd_v
        return xi
    
    sin_half_theta = np.sin(theta / 2.0)
    l = qr_v / sin_half_theta
    d = -2.0 * qd_w / sin_half_theta
    m = (qd_v - qr_w * d / 2.0 * l) / sin_half_theta
    
    xi_rotation = theta * l
    xi_translation = theta * m + d * l
    xi = np.concatenate([xi_rotation, xi_translation]).reshape(6)
    
    return xi


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2,  2*x*y - 2*w*z,        2*x*z + 2*w*y],
        [2*x*y + 2*w*z,        1 - 2*x**2 - 2*z**2,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,        2*y*z + 2*w*x,        1 - 2*x**2 - 2*y**2]
    ])
    return R


def quaternion_multiply_torch(q1, q2):
    """
    Quaternion multiplication in PyTorch.
    
    Args:
        q1, q2: quaternions [w, x, y, z] as torch tensors
    """
    import torch
    
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    result = torch.stack([w, x, y, z])
    norm = torch.norm(result)
    
    if norm > 1e-8:
        return result / norm
    else:
        return result
