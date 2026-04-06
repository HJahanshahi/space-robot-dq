"""
Independent reference implementation of 7-DOF space robot FK.

Uses the matrix exponential (Rodrigues formula) directly — NO dual quaternions.
This serves as ground truth to validate the dual quaternion implementation.

Based on: Lynch & Park, "Modern Robotics", Chapter 4 (Product of Exponentials).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def skew(w):
    """Skew-symmetric matrix from 3D vector."""
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def matrix_exp_screw(omega, q_point, theta):
    """
    Matrix exponential for a revolute joint (pure rotation about an axis).
    
    Computes the 4x4 homogeneous transform for rotation by angle theta
    about axis omega passing through point q_point.
    
    Uses the Rodrigues formula:
        R = I + sin(θ)[ω]× + (1-cos(θ))[ω]×²
        p = (I - R) * q_point
    
    Args:
        omega: unit rotation axis (3,)
        q_point: point on the rotation axis (3,)
        theta: rotation angle (rad)
    
    Returns:
        T: 4x4 homogeneous transformation matrix
    """
    omega = np.array(omega, dtype=float)
    q_point = np.array(q_point, dtype=float)
    
    W = skew(omega)
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * (W @ W)
    p = (np.eye(3) - R) @ q_point  # rotation about a point
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class ReferenceFKMatrixExp:
    """
    Reference 7-DOF space robot FK using matrix exponentials.
    
    Identical robot geometry to SpaceRobotKinematics, but implemented
    independently using 4x4 matrix products (no dual quaternions).
    
    FK formula: T_ee = T_base * exp(S0*q0) * ... * exp(S6*q6) * M
    """
    
    def __init__(self):
        # Link lengths (must match SpaceRobotKinematics exactly)
        self.d1 = 0.310
        self.d3 = 0.400
        self.d5 = 0.390
        self.d7 = 0.078
        
        # Home configuration
        self.M = np.eye(4)
        self.M[2, 3] = self.d1 + self.d3 + self.d5 + self.d7
        
        # Joint definitions: (axis, point_on_axis)
        # Must match the SRS layout in SpaceRobotKinematics exactly
        d1, d3, d5 = self.d1, self.d3, self.d5
        self.joints = [
            {'omega': [0, 0, 1], 'point': [0, 0, 0]},            # J0: shoulder yaw
            {'omega': [0, 1, 0], 'point': [0, 0, d1]},            # J1: shoulder pitch
            {'omega': [1, 0, 0], 'point': [0, 0, d1]},            # J2: shoulder roll
            {'omega': [0, 1, 0], 'point': [0, 0, d1+d3]},         # J3: elbow pitch
            {'omega': [1, 0, 0], 'point': [0, 0, d1+d3]},         # J4: forearm roll
            {'omega': [0, 1, 0], 'point': [0, 0, d1+d3+d5]},      # J5: wrist pitch
            {'omega': [1, 0, 0], 'point': [0, 0, d1+d3+d5]},      # J6: wrist roll
        ]
    
    def forward_kinematics(self, q, base_T=None):
        """
        Compute EE pose using matrix exponential PoE.
        
        Args:
            q: 7D joint angles
            base_T: 4x4 base transform (default: identity)
        
        Returns:
            T_ee: 4x4 end-effector transform
        """
        q = np.array(q).flatten()[:7]
        
        T = np.eye(4) if base_T is None else np.array(base_T)
        
        for i in range(7):
            omega = self.joints[i]['omega']
            point = self.joints[i]['point']
            T = T @ matrix_exp_screw(omega, point, q[i])
        
        T = T @ self.M
        return T
    
    def forward_kinematics_position(self, q, base_T=None):
        """Return position only."""
        T = self.forward_kinematics(q, base_T)
        return T[:3, 3]
    
    def forward_kinematics_6dof(self, q, base_T=None):
        """Return position and quaternion [w,x,y,z]."""
        T = self.forward_kinematics(q, base_T)
        pos = T[:3, 3]
        rot = Rotation.from_matrix(T[:3, :3])
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return pos, quat_wxyz
    
    def compute_jacobian(self, q, base_T=None):
        """
        Compute 6x7 geometric Jacobian via finite differences.
        
        Returns:
            J: (6,7) Jacobian, rows [angular(0:3); linear(3:6)]
        """
        q = np.array(q).flatten()[:7]
        epsilon = 1e-7
        J = np.zeros((6, 7))
        
        pos_0, quat_0 = self.forward_kinematics_6dof(q, base_T)
        
        for i in range(7):
            q_plus = q.copy(); q_plus[i] += epsilon
            q_minus = q.copy(); q_minus[i] -= epsilon
            
            pos_p = self.forward_kinematics_position(q_plus, base_T)
            pos_m = self.forward_kinematics_position(q_minus, base_T)
            J[3:6, i] = (pos_p - pos_m) / (2 * epsilon)
            
            _, quat_p = self.forward_kinematics_6dof(q_plus, base_T)
            _, quat_m = self.forward_kinematics_6dof(q_minus, base_T)
            R_p = Rotation.from_quat([quat_p[1], quat_p[2], quat_p[3], quat_p[0]])
            R_m = Rotation.from_quat([quat_m[1], quat_m[2], quat_m[3], quat_m[0]])
            rotvec = (R_p * R_m.inv()).as_rotvec()
            J[0:3, i] = rotvec / (2 * epsilon)
        
        return J


class ReferenceDHFK:
    """
    Reference FK using standard DH parameters.
    
    An alternative independent implementation using the Denavit-Hartenberg
    convention. This provides a SECOND independent check.
    
    DH parameters derived from the SRS joint layout.
    """
    
    def __init__(self):
        self.d1 = 0.310
        self.d3 = 0.400
        self.d5 = 0.390
        self.d7 = 0.078
        
        # Modified DH parameters: [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i]
        # Derived from the joint layout to produce identical FK
        #
        # The SRS robot with axes z,y,x,y,x,y,x requires frame rotations
        # to align each joint with the DH z-axis convention.
        #
        # We use elementary transforms instead for clarity.
        pass
    
    def _transform(self, axis, angle, translation):
        """Elementary rotation + translation."""
        T = np.eye(4)
        if axis == 'x':
            T[:3, :3] = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
        elif axis == 'y':
            T[:3, :3] = Rotation.from_rotvec([0, angle, 0]).as_matrix()
        elif axis == 'z':
            T[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        T[:3, 3] = translation
        return T
    
    def forward_kinematics(self, q):
        """
        FK using elementary transform sequence.
        
        Build T_ee by composing: translate to joint → rotate by q_i → ... → add EE offset
        """
        q = np.array(q).flatten()[:7]
        d1, d3, d5, d7 = self.d1, self.d3, self.d5, self.d7
        
        T = np.eye(4)
        
        # J0: rotate about z at origin
        T = T @ self._transform('z', q[0], [0, 0, 0])
        
        # Translate to shoulder
        T[:3, 3] += T[:3, :3] @ np.array([0, 0, d1])
        
        # J1: rotate about y at shoulder
        T = T @ self._transform('y', q[1], [0, 0, 0])
        
        # J2: rotate about x at shoulder (colocated)
        T = T @ self._transform('x', q[2], [0, 0, 0])
        
        # Translate to elbow
        T[:3, 3] += T[:3, :3] @ np.array([0, 0, d3])
        
        # J3: rotate about y at elbow
        T = T @ self._transform('y', q[3], [0, 0, 0])
        
        # J4: rotate about x at elbow (colocated)
        T = T @ self._transform('x', q[4], [0, 0, 0])
        
        # Translate to wrist
        T[:3, 3] += T[:3, :3] @ np.array([0, 0, d5])
        
        # J5: rotate about y at wrist
        T = T @ self._transform('y', q[5], [0, 0, 0])
        
        # J6: rotate about x at wrist (colocated)
        T = T @ self._transform('x', q[6], [0, 0, 0])
        
        # Translate to EE
        T[:3, 3] += T[:3, :3] @ np.array([0, 0, d7])
        
        return T
    
    def forward_kinematics_position(self, q):
        return self.forward_kinematics(q)[:3, 3]
    
    def forward_kinematics_6dof(self, q):
        T = self.forward_kinematics(q)
        pos = T[:3, 3]
        rot = Rotation.from_matrix(T[:3, :3])
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return pos, quat_wxyz
