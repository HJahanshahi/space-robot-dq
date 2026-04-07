"""
Generalized N-DOF kinematics for free-floating space robot manipulators.

Supports arbitrary serial chain robots defined by the user. The robot is
specified as a list of revolute joints, each with an axis direction and
position in the home (zero-angle) configuration.

FK uses the space-frame Product of Exponentials:
    T_ee = T_base * exp(S0*q0) * ... * exp(S_{n-1}*q_{n-1}) * M

where S_i are screw axes and M is the home end-effector pose.

Includes a built-in 7-DOF SRS preset matching standard space manipulators.

Author: Hadi Jahanshahi, Zheng H. Zhu
Affiliation: Department of Mechanical Engineering, York University
Date: 2025
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.optimize import minimize, Bounds
from scipy.spatial.transform import Rotation
from .dual_quaternion import DualQuaternion, quaternion_multiply


# ============================================================================
# ROBOT CONFIGURATION
# ============================================================================

@dataclass
class JointDef:
    """
    Definition of a single revolute joint.
    
    Attributes:
        axis: (3,) unit rotation axis in home frame (e.g. [0,0,1] for z)
        position: (3,) point on the rotation axis in home frame (meters)
        name: human-readable joint name
        q_min: lower joint limit (rad)
        q_max: upper joint limit (rad)
    """
    axis: np.ndarray
    position: np.ndarray
    name: str = ""
    q_min: float = -np.pi
    q_max: float = np.pi
    
    def __post_init__(self):
        self.axis = np.array(self.axis, dtype=float)
        self.position = np.array(self.position, dtype=float)
        # Normalize axis
        norm = np.linalg.norm(self.axis)
        if norm > 1e-12:
            self.axis = self.axis / norm


@dataclass
class RobotConfig:
    """
    Complete robot configuration for a free-floating space manipulator.
    
    Attributes:
        joints: list of JointDef objects defining each revolute joint
        ee_position: (3,) end-effector position in home frame
        ee_orientation: (3,3) end-effector orientation in home frame (default: I)
        name: robot name
    """
    joints: List[JointDef]
    ee_position: np.ndarray
    ee_orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    name: str = "SpaceRobot"
    
    def __post_init__(self):
        self.ee_position = np.array(self.ee_position, dtype=float)
        self.ee_orientation = np.array(self.ee_orientation, dtype=float)
    
    @property
    def num_joints(self):
        return len(self.joints)
    
    @property
    def q_min(self):
        return np.array([j.q_min for j in self.joints])
    
    @property
    def q_max(self):
        return np.array([j.q_max for j in self.joints])
    
    def summary(self):
        """Print robot configuration summary."""
        print(f"\n{'='*60}")
        print(f"Robot: {self.name} ({self.num_joints} DOF)")
        print(f"{'='*60}")
        print(f"  EE home position: {self.ee_position}")
        print(f"\n  {'#':<4} {'Name':<18} {'Axis':<16} {'Position (m)':<24} {'Limits (°)'}")
        print(f"  {'-'*80}")
        for i, j in enumerate(self.joints):
            ax = f"[{j.axis[0]:.0f},{j.axis[1]:.0f},{j.axis[2]:.0f}]"
            pos = f"[{j.position[0]:.3f},{j.position[1]:.3f},{j.position[2]:.3f}]"
            lim = f"[{np.rad2deg(j.q_min):.0f}, {np.rad2deg(j.q_max):.0f}]"
            print(f"  {i:<4} {j.name:<18} {ax:<16} {pos:<24} {lim}")
        print(f"{'='*60}")


# ============================================================================
# PRESET ROBOT CONFIGURATIONS
# ============================================================================

def create_7dof_srs():
    """
    Create preset 7-DOF SRS (Spherical-Revolute-Spherical) space manipulator.
    
    Joint layout at home configuration (all joints zero):
        J0 (shoulder yaw):   z-axis, at origin
        J1 (shoulder pitch):  y-axis, at [0, 0, 0.310]
        J2 (shoulder roll):   x-axis, at [0, 0, 0.310]
        J3 (elbow pitch):     y-axis, at [0, 0, 0.710]
        J4 (forearm roll):    x-axis, at [0, 0, 0.710]
        J5 (wrist pitch):     y-axis, at [0, 0, 1.100]
        J6 (wrist roll):      x-axis, at [0, 0, 1.100]
    
    End-effector at home: [0, 0, 1.178] m
    
    Link lengths: d1=0.310, d3=0.400, d5=0.390, d7=0.078 m
    """
    d1, d3, d5, d7 = 0.310, 0.400, 0.390, 0.078
    
    joints = [
        JointDef([0,0,1], [0,0,0],       "shoulder_yaw",   -np.pi,     np.pi),
        JointDef([0,1,0], [0,0,d1],      "shoulder_pitch",  -np.pi/2,  np.pi/2),
        JointDef([1,0,0], [0,0,d1],      "shoulder_roll",   -np.pi,    np.pi),
        JointDef([0,1,0], [0,0,d1+d3],   "elbow_pitch",     0.1,       np.pi-0.1),
        JointDef([1,0,0], [0,0,d1+d3],   "forearm_roll",    -np.pi,    np.pi),
        JointDef([0,1,0], [0,0,d1+d3+d5],"wrist_pitch",     -np.pi/2,  np.pi/2),
        JointDef([1,0,0], [0,0,d1+d3+d5],"wrist_roll",      -np.pi,    np.pi),
    ]
    
    ee_pos = np.array([0.0, 0.0, d1 + d3 + d5 + d7])
    
    return RobotConfig(joints=joints, ee_position=ee_pos, name="7-DOF SRS Space Manipulator")


def create_3dof_planar():
    """
    Create a simple 3-DOF planar manipulator (for testing / education).
    
    All joints rotate about the z-axis in the x-z plane.
    Link lengths: L1=0.4, L2=0.3, L3=0.2 m
    """
    L1, L2, L3 = 0.4, 0.3, 0.2
    
    joints = [
        JointDef([0,1,0], [0,0,0],      "joint_1", -np.pi, np.pi),
        JointDef([0,1,0], [0,0,L1],     "joint_2", -np.pi, np.pi),
        JointDef([0,1,0], [0,0,L1+L2],  "joint_3", -np.pi, np.pi),
    ]
    
    ee_pos = np.array([0.0, 0.0, L1 + L2 + L3])
    
    return RobotConfig(joints=joints, ee_position=ee_pos, name="3-DOF Planar")


def create_6dof_standard():
    """
    Create a 6-DOF standard industrial-style manipulator.
    
    Alternating axes (z,y,x,y,x,y) for full SE(3) reachability.
    """
    d1, d2, d3, d4 = 0.300, 0.400, 0.350, 0.080
    
    joints = [
        JointDef([0,0,1], [0,0,0],          "base_yaw",     -np.pi,    np.pi),
        JointDef([0,1,0], [0,0,d1],         "shoulder",      -np.pi/2, np.pi/2),
        JointDef([1,0,0], [0,0,d1],         "elbow_roll",    -np.pi,   np.pi),
        JointDef([0,1,0], [0,0,d1+d2],      "elbow_pitch",   0.1,      np.pi-0.1),
        JointDef([1,0,0], [0,0,d1+d2+d3],   "wrist_roll",    -np.pi,   np.pi),
        JointDef([0,1,0], [0,0,d1+d2+d3],   "wrist_pitch",   -np.pi/2, np.pi/2),
    ]
    
    ee_pos = np.array([0.0, 0.0, d1 + d2 + d3 + d4])
    
    return RobotConfig(joints=joints, ee_position=ee_pos, name="6-DOF Standard")


# ============================================================================
# KINEMATICS CLASS
# ============================================================================

class SpaceRobotKinematics:
    """
    Kinematics for an N-DOF free-floating space robot manipulator.
    
    Supports any serial chain of revolute joints defined via RobotConfig.
    Uses dual quaternion algebra and the Product of Exponentials formulation.
    
    Args:
        config: RobotConfig defining the robot, or None for default 7-DOF SRS
        verbose: print initialization info
    
    Example::
    
        # Default 7-DOF SRS
        kin = SpaceRobotKinematics()
        
        # Custom robot
        config = RobotConfig(joints=[...], ee_position=[...])
        kin = SpaceRobotKinematics(config)
        
        # Preset
        from space_robot_dq import create_3dof_planar
        kin = SpaceRobotKinematics(create_3dof_planar())
    """
    
    def __init__(self, config=None, verbose=False):
        if config is None:
            config = create_7dof_srs()
        
        self.config = config
        self.num_joints = config.num_joints
        self.q_min = config.q_min
        self.q_max = config.q_max
        
        # Base pose (identity = inertial frame)
        self.base_dq = DualQuaternion()
        
        # Home configuration
        self.home_position = config.ee_position.copy()
        self.M = np.eye(4)
        self.M[:3, :3] = config.ee_orientation
        self.M[:3, 3] = config.ee_position
        self.M_dq = DualQuaternion.from_pose(config.ee_orientation, config.ee_position)
        
        # Compute screw axes from joint definitions
        self.screw_axes = self._calculate_screw_axes()
        
        if verbose:
            config.summary()

    def _calculate_screw_axes(self):
        """Compute screw axes from joint definitions, transformed to base frame."""
        T_base = self.base_dq.to_matrix()
        R_base = T_base[:3, :3]
        t_base = T_base[:3, 3]
        
        screw_axes = []
        for j in self.config.joints:
            l_global = R_base @ j.axis
            l_global = l_global / (np.linalg.norm(l_global) + 1e-12)
            p_global = R_base @ j.position + t_base
            m_global = np.cross(p_global, l_global)
            
            screw_axes.append({
                'l': l_global,
                'p': p_global,
                'm': m_global,
                'name': j.name,
            })
        
        return screw_axes

    def forward_kinematics_dq(self, joint_angles=None):
        """
        FK using dual quaternions (Product of Exponentials).
        
        T_ee = T_base * exp(S0*q0) * ... * exp(S_{n-1}*q_{n-1}) * M
        
        Args:
            joint_angles: (N,) joint configuration
        
        Returns:
            DualQuaternion for end-effector pose
        """
        if joint_angles is not None:
            q = np.array(joint_angles).flatten()[:self.num_joints]
        else:
            q = np.zeros(self.num_joints)
        
        result = self.base_dq
        
        for i in range(self.num_joints):
            l = self.screw_axes[i]['l']
            m = self.screw_axes[i]['m']
            delta_i = DualQuaternion.from_screw(q[i], 0.0, l, m)
            result = result * delta_i
        
        result = result * self.M_dq
        return result

    def forward_kinematics(self, joint_angles):
        """
        FK returning position only.
        
        Args:
            joint_angles: (N,) joint angles
        
        Returns:
            (3,) end-effector position
        """
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        q = np.array(joint_angles).flatten()[:self.num_joints]
        R, p = self.forward_kinematics_dq(q).to_pose()
        return p

    def forward_kinematics_6dof(self, joint_angles):
        """
        FK returning full 6-DOF pose.
        
        Args:
            joint_angles: (N,) joint angles
        
        Returns:
            position: (3,) end-effector position
            quaternion: (4,) orientation [w, x, y, z]
        """
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        q = np.array(joint_angles).flatten()[:self.num_joints]
        R, p = self.forward_kinematics_dq(q).to_pose()
        rot = Rotation.from_matrix(R)
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return p, quat_wxyz

    def inverse_kinematics(self, target_position, initial_guess=None):
        """
        Position-only IK using L-BFGS-B.
        
        Args:
            target_position: (3,) target position
            initial_guess: (N,) initial joint angles
        
        Returns:
            (N,) joint angles
        """
        target_position = np.array(target_position).flatten()[:3]
        
        if initial_guess is None:
            initial_guess = (self.q_min + self.q_max) / 2.0
        
        def objective(joints):
            try:
                return np.linalg.norm(self.forward_kinematics(joints) - target_position)
            except Exception:
                return 1000.0
        
        result = minimize(objective, initial_guess,
                          bounds=list(zip(self.q_min, self.q_max)),
                          method='L-BFGS-B')
        return result.x

    def inverse_kinematics_6dof(self, target_position, target_quaternion,
                                 initial_guess=None,
                                 position_weight=10.0, orientation_weight=2.0):
        """
        6-DOF IK with configurable position/orientation weights.
        
        Args:
            target_position: (3,) target position
            target_quaternion: (4,) target quaternion [w,x,y,z]
            initial_guess: (N,) initial joint angles
            position_weight: weight for position error
            orientation_weight: weight for orientation error
        
        Returns:
            (N,) joint angles
        """
        target_position = np.array(target_position).flatten()[:3]
        target_quaternion = np.array(target_quaternion).flatten()[:4]
        target_quaternion = target_quaternion / (np.linalg.norm(target_quaternion) + 1e-12)
        
        if initial_guess is None:
            initial_guess = (self.q_min + self.q_max) / 2.0
        
        def objective(joints):
            try:
                pos, quat = self.forward_kinematics_6dof(joints)
                pos_err = np.linalg.norm(pos - target_position) * position_weight
                ori_err = quaternion_distance(quat, target_quaternion) * orientation_weight
                return pos_err + ori_err
            except Exception:
                return 1000.0
        
        bounds = Bounds(self.q_min, self.q_max)
        
        best = minimize(objective, initial_guess, bounds=bounds,
                         method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
        
        # Multi-restart
        for _ in range(5):
            guess = initial_guess + np.random.randn(self.num_joints) * 0.5
            guess = np.clip(guess, self.q_min, self.q_max)
            result = minimize(objective, guess, bounds=bounds,
                              method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            if result.fun < best.fun:
                best = result
        
        # Warm start from position-only IK
        pos_sol = self.inverse_kinematics(target_position, initial_guess)
        result = minimize(objective, pos_sol, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
        if result.fun < best.fun:
            best = result
        
        return best.x

    def calculate_jacobian(self, joint_angles=None):
        """
        Numerical geometric Jacobian (6×N) via central differences.
        
        Rows [0:3] = angular velocity, [3:6] = linear velocity.
        Guaranteed consistent with FK.
        
        Args:
            joint_angles: (N,) joint angles
        
        Returns:
            (6, N) Jacobian
        """
        q = np.array(joint_angles).flatten()[:self.num_joints] if joint_angles is not None \
            else np.zeros(self.num_joints)
        
        J = np.zeros((6, self.num_joints))
        epsilon = 1e-7
        
        for i in range(self.num_joints):
            q_p, q_m = q.copy(), q.copy()
            q_p[i] += epsilon
            q_m[i] -= epsilon
            
            pos_p = self.forward_kinematics(q_p)
            pos_m = self.forward_kinematics(q_m)
            J[3:6, i] = (pos_p - pos_m) / (2 * epsilon)
            
            _, quat_p = self.forward_kinematics_6dof(q_p)
            _, quat_m = self.forward_kinematics_6dof(q_m)
            R_p = Rotation.from_quat([quat_p[1], quat_p[2], quat_p[3], quat_p[0]])
            R_m = Rotation.from_quat([quat_m[1], quat_m[2], quat_m[3], quat_m[0]])
            J[0:3, i] = (R_p * R_m.inv()).as_rotvec() / (2 * epsilon)
        
        return J

    def calculate_jacobian_analytical(self, joint_angles=None):
        """
        Analytical geometric Jacobian (6×N) via Product of Exponentials.
        
        Args:
            joint_angles: (N,) joint angles
        
        Returns:
            (6, N) Jacobian [angular(0:3); linear(3:6)]
        """
        q = np.array(joint_angles).flatten()[:self.num_joints] if joint_angles is not None \
            else np.zeros(self.num_joints)
        
        J = np.zeros((6, self.num_joints))
        ee_pos = self.forward_kinematics(q)
        T_partial = self.base_dq.to_matrix()
        
        for i in range(self.num_joints):
            l_i = self.screw_axes[i]['l']
            p_i = self.screw_axes[i]['p']
            R = T_partial[:3, :3]
            t = T_partial[:3, 3]
            
            z_i = R @ l_i
            z_i = z_i / (np.linalg.norm(z_i) + 1e-12)
            o_i = R @ p_i + t
            
            J[0:3, i] = z_i
            J[3:6, i] = np.cross(z_i, ee_pos - o_i)
            
            theta_i = q[i]
            R_i = Rotation.from_rotvec(l_i * theta_i).as_matrix()
            T_joint = np.eye(4)
            T_joint[:3, :3] = R_i
            T_joint[:3, 3] = p_i - R_i @ p_i
            T_partial = T_partial @ T_joint
        
        return J


# ============================================================================
# QUATERNION UTILITIES
# ============================================================================

def quaternion_distance(q1, q2):
    """Angular distance between quaternions [w,x,y,z] in radians."""
    q1 = np.array(q1).flatten()[:4]
    q2 = np.array(q2).flatten()[:4]
    n1, n2 = np.linalg.norm(q1), np.linalg.norm(q2)
    if n1 < 1e-10 or n2 < 1e-10:
        return np.pi
    q1, q2 = q1 / n1, q2 / n2
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def quaternion_to_euler(q):
    """Quaternion [w,x,y,z] → Euler [roll, pitch, yaw] (xyz)."""
    q = np.array(q).flatten()[:4]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz')


def euler_to_quaternion(euler):
    """Euler [roll, pitch, yaw] → quaternion [w,x,y,z]."""
    quat_xyzw = Rotation.from_euler('xyz', euler).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


# ============================================================================
# STANDALONE CONVENIENCE FUNCTIONS (use default 7-DOF SRS)
# ============================================================================

_cached_kinematics = None

def _get_kinematics():
    global _cached_kinematics
    if _cached_kinematics is None:
        _cached_kinematics = SpaceRobotKinematics()
    return _cached_kinematics


def compute_forward_kinematics(joint_angles):
    """Standalone FK (default 7-DOF SRS)."""
    if isinstance(joint_angles, torch.Tensor):
        arr = joint_angles.cpu().numpy()
        return torch.tensor(_get_kinematics().forward_kinematics(arr), dtype=torch.float32)
    return _get_kinematics().forward_kinematics(np.array(joint_angles))


def forward_kinematics_simple(joint_angles):
    """Position-only FK (default 7-DOF SRS)."""
    if isinstance(joint_angles, torch.Tensor):
        joint_angles = joint_angles.cpu().numpy()
    return _get_kinematics().forward_kinematics(np.array(joint_angles))


def forward_kinematics_6dof(joint_angles):
    """6-DOF FK (default 7-DOF SRS)."""
    if isinstance(joint_angles, torch.Tensor):
        joint_angles = joint_angles.cpu().numpy()
    return _get_kinematics().forward_kinematics_6dof(np.array(joint_angles))


def inverse_kinematics_6dof(target_position, target_quaternion, initial_guess=None,
                            position_weight=10.0, orientation_weight=2.0):
    """6-DOF IK (default 7-DOF SRS)."""
    return _get_kinematics().inverse_kinematics_6dof(
        target_position, target_quaternion, initial_guess, position_weight, orientation_weight)


def inverse_kinematics_numerical_scipy(target_position, initial_guess=None, method='SLSQP'):
    """Position-only IK with multi-restart (default 7-DOF SRS)."""
    kin = _get_kinematics()
    target_position = np.array(target_position).flatten()[:3]
    if initial_guess is None:
        initial_guess = (kin.q_min + kin.q_max) / 2.0
    
    bounds = Bounds(kin.q_min, kin.q_max)
    
    def objective(q):
        return np.linalg.norm(kin.forward_kinematics(q) - target_position)
    
    best = minimize(objective, initial_guess, method=method, bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-6})
    
    for _ in range(3):
        guess = initial_guess + np.random.randn(kin.num_joints) * 0.3
        guess = np.clip(guess, kin.q_min, kin.q_max)
        result = minimize(objective, guess, method=method, bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-6})
        if result.fun < best.fun:
            best = result
    
    return best.x


def find_achievable_orientation(target_position, desired_orientation=None, n_samples=200,
                                 position_tolerance=0.005):
    """Find achievable orientation closest to desired at target position."""
    kin = _get_kinematics()
    target_position = np.array(target_position).flatten()[:3]
    if desired_orientation is None:
        desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        desired_orientation = np.array(desired_orientation).flatten()[:4]
        desired_orientation = desired_orientation / (np.linalg.norm(desired_orientation) + 1e-12)
    
    valid = []
    for _ in range(n_samples):
        q_init = np.random.uniform(kin.q_min, kin.q_max)
        q_sol = kin.inverse_kinematics(target_position, initial_guess=q_init)
        pos, quat = kin.forward_kinematics_6dof(q_sol)
        pos_err = np.linalg.norm(pos - target_position)
        if pos_err < position_tolerance:
            valid.append({'q': q_sol, 'quat': quat, 'pos_error': pos_err,
                          'orient_error': quaternion_distance(quat, desired_orientation)})
    
    for ow in [0.5, 1.0, 2.0, 5.0, 8.0]:
        for _ in range(10):
            try:
                q_sol = kin.inverse_kinematics_6dof(
                    target_position, desired_orientation,
                    np.random.uniform(kin.q_min, kin.q_max), 10.0, ow)
                pos, quat = kin.forward_kinematics_6dof(q_sol)
                pos_err = np.linalg.norm(pos - target_position)
                if pos_err < position_tolerance:
                    valid.append({'q': q_sol, 'quat': quat, 'pos_error': pos_err,
                                  'orient_error': quaternion_distance(quat, desired_orientation)})
            except Exception:
                pass
    
    if not valid:
        q_sol = kin.inverse_kinematics(target_position)
        pos, quat = kin.forward_kinematics_6dof(q_sol)
        return quat, np.linalg.norm(pos - target_position), \
               quaternion_distance(quat, desired_orientation), q_sol
    
    best = min(valid, key=lambda x: x['orient_error'])
    return best['quat'], best['pos_error'], best['orient_error'], best['q']


def find_achievable_orientation_at_position(target_position, n_samples=300,
                                             desired_orientation=None):
    """Find achievable orientation at position, closest to desired."""
    if desired_orientation is None:
        desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    quat, _, _, joints = find_achievable_orientation(
        target_position, desired_orientation, n_samples, 0.005)
    return quat, joints


def find_best_achievable_orientation(target_position, desired_orientation, n_samples=500):
    """Extensive search for best achievable orientation."""
    return find_achievable_orientation(target_position, desired_orientation, n_samples, 0.003)
