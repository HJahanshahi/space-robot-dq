"""
Kinematics for 7-DOF free-floating space robot manipulator.

UPDATED:
- Product of Exponentials (PoE) formulation with proper home configuration
- Full d7 end-effector offset included
- Clean joint geometry: no artificial offsets, physically meaningful layout
- 7-DOF SRS (Spherical-Revolute-Spherical) architecture:
    Shoulder: yaw(z) + pitch(y) + roll(x)
    Elbow:    pitch(y)
    Wrist:    roll(x) + pitch(y) + roll(x)
- Full 6DOF FK/IK using dual quaternions
- Numerical + analytical Jacobians
- Achievable orientation finder

Author: Hadi Jahanshahi, Zheng H. Zhu
Affiliation: Department of Mechanical Engineering, York University
Date: 2025
"""

import numpy as np
import torch
from scipy.optimize import minimize, Bounds
from scipy.spatial.transform import Rotation
from .dual_quaternion import DualQuaternion, quaternion_multiply


# Global cached kinematics instance for performance
_cached_kinematics = None


def _get_kinematics():
    """Get cached kinematics instance for performance"""
    global _cached_kinematics
    if _cached_kinematics is None:
        _cached_kinematics = SpaceRobotKinematics()
    return _cached_kinematics


class SpaceRobotKinematics:
    """
    Space robot kinematics using screw theory and dual quaternions.
    
    The robot has:
    - 7 DOF manipulator arm (SRS architecture)
    - Free-floating base (6 DOF)
    - Total 13 DOF system
    
    Joint layout at home configuration (all joints zero):
        J0 (shoulder yaw):   axis z, at origin
        J1 (shoulder pitch):  axis y, at [0, 0, d1]
        J2 (shoulder roll):   axis x, at [0, 0, d1]        (colocated with J1)
        J3 (elbow pitch):     axis y, at [0, 0, d1+d3]
        J4 (forearm roll):    axis x, at [0, 0, d1+d3]     (colocated with J3)
        J5 (wrist pitch):     axis y, at [0, 0, d1+d3+d5]
        J6 (wrist roll):      axis x, at [0, 0, d1+d3+d5]  (colocated with J5)
    
    End-effector at home: [0, 0, d1+d3+d5+d7]
    
    FK uses the space-frame Product of Exponentials:
        T_ee = T_base * exp(S0*q0) * exp(S1*q1) * ... * exp(S6*q6) * M
    where M is the home configuration and S_i are space-frame screw axes.
    """
    
    def __init__(self, verbose=False):
        """Initialize 7-DOF space robot kinematics
        
        Args:
            verbose: if True, print initialization info
        """
        self.num_joints = 7
        
        # Link lengths (meters) - standard 7-DOF space manipulator
        self.d1 = 0.310  # Base to shoulder
        self.d3 = 0.400  # Shoulder to elbow
        self.d5 = 0.390  # Elbow to wrist
        self.d7 = 0.078  # Wrist to end-effector
        
        # Current configuration
        self.q = np.zeros(7)  # Joint angles
        self.base_dq = DualQuaternion()  # Base pose (identity = inertial frame)
        
        # Home configuration: EE pose when all joints are zero
        self.home_position = np.array([0.0, 0.0, self.d1 + self.d3 + self.d5 + self.d7])
        self.M = np.eye(4)
        self.M[:3, 3] = self.home_position
        self.M_dq = DualQuaternion.from_pose(np.eye(3), self.home_position)
        
        # Calculate screw axes in base frame
        self.screw_axes = self._calculate_screw_axes()
        
        # Joint limits (radians)
        # J0: shoulder yaw, J1: shoulder pitch, J2: shoulder roll
        # J3: elbow pitch (restricted to prevent full extension)
        # J4: forearm roll, J5: wrist pitch, J6: wrist roll
        self.q_min = np.array([-np.pi,  -np.pi/2, -np.pi,    0.1,      -np.pi,    -np.pi/2, -np.pi])
        self.q_max = np.array([ np.pi,   np.pi/2,  np.pi,    np.pi-0.1,  np.pi,    np.pi/2,   np.pi])
        
        if verbose:
            print(f"✅ Initialized SpaceRobotKinematics with {self.num_joints} joints")
            print(f"   Home EE position: {self.home_position}")

    def _calculate_screw_axes(self):
        """
        Calculate screw axes in base frame at home configuration.
        
        Each screw axis is defined by:
            l: unit direction vector (rotation axis)
            p: a point on the axis
            m: moment vector = p × l
        
        For revolute joints, the screw has zero pitch (d=0).
        
        Returns:
            screw_axes: list of dicts with 'l', 'p', 'm', 'name' keys
        """
        # Joint definitions: position and axis at home configuration
        # SRS architecture with alternating axes for full 6DOF capability
        joints_local = [
            # Shoulder group
            {'p': [0, 0, 0],                           'l': [0, 0, 1], 'name': 'shoulder_yaw'},
            {'p': [0, 0, self.d1],                      'l': [0, 1, 0], 'name': 'shoulder_pitch'},
            {'p': [0, 0, self.d1],                      'l': [1, 0, 0], 'name': 'shoulder_roll'},
            # Elbow
            {'p': [0, 0, self.d1 + self.d3],            'l': [0, 1, 0], 'name': 'elbow_pitch'},
            # Wrist group
            {'p': [0, 0, self.d1 + self.d3],            'l': [1, 0, 0], 'name': 'forearm_roll'},
            {'p': [0, 0, self.d1 + self.d3 + self.d5],  'l': [0, 1, 0], 'name': 'wrist_pitch'},
            {'p': [0, 0, self.d1 + self.d3 + self.d5],  'l': [1, 0, 0], 'name': 'wrist_roll'},
        ]
        
        T_base = self.base_dq.to_matrix()
        R_base = T_base[:3, :3]
        t_base = T_base[:3, 3]
        
        screw_axes = []
        for j in joints_local:
            l_local = np.array(j['l'], dtype=float)
            p_local = np.array(j['p'], dtype=float)
            
            # Transform to base (world) frame
            l_global = R_base @ l_local
            l_global = l_global / np.linalg.norm(l_global)
            p_global = R_base @ p_local + t_base
            m_global = np.cross(p_global, l_global)
            
            screw_axes.append({
                'l': l_global,
                'p': p_global,
                'm': m_global,
                'name': j['name']
            })
        
        return screw_axes

    def forward_kinematics_dq(self, joint_angles=None):
        """
        Forward kinematics using dual quaternions (Product of Exponentials).
        
        Computes: T_ee = T_base * exp(S0*q0) * ... * exp(S6*q6) * M
        
        Args:
            joint_angles: 7D joint configuration
        
        Returns:
            DualQuaternion representing end-effector pose
        """
        if joint_angles is not None:
            q = np.array(joint_angles).flatten()[:7]
        else:
            q = self.q
        
        # Start with base transform
        result = self.base_dq
        
        # Apply each joint's screw motion (space-frame PoE, left to right)
        for i in range(self.num_joints):
            l = self.screw_axes[i]['l']
            m = self.screw_axes[i]['m']
            theta = q[i]
            
            # Pure revolute joint: d=0 (no translation along axis)
            delta_i = DualQuaternion.from_screw(theta, 0.0, l, m)
            result = result * delta_i
        
        # Apply home configuration
        result = result * self.M_dq
        
        return result

    def forward_kinematics(self, joint_angles):
        """
        Forward kinematics - returns position only.
        
        Args:
            joint_angles: 7D joint configuration (numpy array or torch tensor)
        
        Returns:
            position: 3D end-effector position
        """
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        
        joint_angles = np.array(joint_angles).flatten()[:7]
        ee_dq = self.forward_kinematics_dq(joint_angles)
        R, p = ee_dq.to_pose()
        
        return p

    def forward_kinematics_6dof(self, joint_angles):
        """
        Forward kinematics - returns FULL 6DOF pose.
        
        Args:
            joint_angles: 7D joint configuration
        
        Returns:
            position: 3D end-effector position
            quaternion: 4D orientation quaternion [w, x, y, z]
        """
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        
        joint_angles = np.array(joint_angles).flatten()[:7]
        
        ee_dq = self.forward_kinematics_dq(joint_angles)
        R, p = ee_dq.to_pose()
        
        rot = Rotation.from_matrix(R)
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return p, quat_wxyz

    def inverse_kinematics(self, target_position, initial_guess=None):
        """
        Position-only inverse kinematics using L-BFGS-B.
        
        Args:
            target_position: 3D target position
            initial_guess: 7D initial joint angles
        
        Returns:
            joint_angles: 7D solution
        """
        target_position = np.array(target_position).flatten()[:3]
        
        if initial_guess is None:
            initial_guess = np.array([0, 0.2, 0, 1.0, 0, -0.3, 0])
        
        def objective_function(joints):
            try:
                current_pos = self.forward_kinematics(joints)
                return np.linalg.norm(current_pos - target_position)
            except Exception:
                return 1000.0
        
        joint_limits = list(zip(self.q_min, self.q_max))
        
        result = minimize(
            objective_function,
            initial_guess,
            bounds=joint_limits,
            method='L-BFGS-B'
        )
        
        return result.x

    def inverse_kinematics_6dof(self, target_position, target_quaternion, initial_guess=None,
                                 position_weight=10.0, orientation_weight=2.0):
        """
        6DOF inverse kinematics with configurable weights.
        
        Args:
            target_position: [3] target position
            target_quaternion: [4] target quaternion [w,x,y,z]
            initial_guess: [7] initial joint angles
            position_weight: Weight for position error (default 10.0)
            orientation_weight: Weight for orientation error (default 2.0)
        
        Returns:
            [7] joint angles achieving the target pose
        """
        target_position = np.array(target_position).flatten()[:3]
        target_quaternion = np.array(target_quaternion).flatten()[:4]
        target_quaternion = target_quaternion / (np.linalg.norm(target_quaternion) + 1e-12)
        
        if initial_guess is None:
            initial_guess = np.array([0, 0.2, 0, 1.0, 0, -0.3, 0])
        
        def objective_function(joints):
            try:
                current_pos, current_quat = self.forward_kinematics_6dof(joints)
                
                pos_error = np.linalg.norm(current_pos - target_position)
                orient_error_rad = quaternion_distance(current_quat, target_quaternion)
                
                pos_cost = pos_error * position_weight
                orient_cost = orient_error_rad * orientation_weight
                
                return pos_cost + orient_cost
            except Exception:
                return 1000.0
        
        bounds = Bounds(self.q_min, self.q_max)
        
        result = minimize(
            objective_function,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        best_result = result
        best_cost = result.fun
        
        # Multiple restarts
        for i in range(5):
            random_guess = initial_guess + np.random.randn(7) * 0.5
            random_guess = np.clip(random_guess, self.q_min, self.q_max)
            
            result = minimize(
                objective_function,
                random_guess,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.fun < best_cost:
                best_result = result
                best_cost = result.fun
        
        # Warm start from position-only IK
        pos_only_solution = self.inverse_kinematics(target_position, initial_guess)
        
        result = minimize(
            objective_function,
            pos_only_solution,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if result.fun < best_cost:
            best_result = result
        
        return best_result.x

    def calculate_jacobian(self, joint_angles=None):
        """
        Calculate 6x7 geometric Jacobian using numerical central differences.
        
        Guaranteed consistent with FK. Uses the convention:
            [omega; v] = J * dq
        
        Rows 0-2: angular velocity
        Rows 3-5: linear velocity
        
        Args:
            joint_angles: 7D joint configuration (uses self.q if None)
        
        Returns:
            J: 6x7 Jacobian matrix
        """
        if joint_angles is not None:
            q = np.array(joint_angles).flatten()[:7]
        else:
            q = self.q
        
        J = np.zeros((6, self.num_joints))
        epsilon = 1e-7
        
        for i in range(self.num_joints):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += epsilon
            q_minus[i] -= epsilon
            
            # Linear velocity via position differences
            pos_plus = self.forward_kinematics(q_plus)
            pos_minus = self.forward_kinematics(q_minus)
            J[3:6, i] = (pos_plus - pos_minus) / (2.0 * epsilon)
            
            # Angular velocity via rotation differences
            _, quat_plus = self.forward_kinematics_6dof(q_plus)
            _, quat_minus = self.forward_kinematics_6dof(q_minus)
            
            R_plus = Rotation.from_quat([quat_plus[1], quat_plus[2], quat_plus[3], quat_plus[0]])
            R_minus = Rotation.from_quat([quat_minus[1], quat_minus[2], quat_minus[3], quat_minus[0]])
            
            R_delta = R_plus * R_minus.inv()
            rotvec = R_delta.as_rotvec()
            J[0:3, i] = rotvec / (2.0 * epsilon)
        
        return J

    def calculate_jacobian_analytical(self, joint_angles=None):
        """
        Analytical geometric Jacobian using Product of Exponentials.
        
        For joint i, the Jacobian column is computed as:
            angular: z_i  (current axis direction)
            linear:  z_i × (p_ee - o_i)  (cross product with EE offset)
        
        where z_i and o_i are the axis direction and position of joint i
        transformed by all preceding joint motions.
        
        Args:
            joint_angles: 7D joint configuration
        
        Returns:
            J: 6x7 Jacobian matrix [angular(0:3); linear(3:6)]
        """
        if joint_angles is not None:
            q = np.array(joint_angles).flatten()[:7]
        else:
            q = self.q
        
        J = np.zeros((6, self.num_joints))
        ee_pos = self.forward_kinematics(q)
        
        # Build cumulative transform: T_partial = base * exp(S0*q0) * ... * exp(S_{i-1}*q_{i-1})
        T_partial = self.base_dq.to_matrix()
        
        for i in range(self.num_joints):
            l_i = self.screw_axes[i]['l']  # axis direction at home
            p_i = self.screw_axes[i]['p']  # point on axis at home
            
            R = T_partial[:3, :3]
            t = T_partial[:3, 3]
            
            # Current axis direction and position in world frame
            z_i = R @ l_i
            z_i = z_i / (np.linalg.norm(z_i) + 1e-12)
            o_i = R @ p_i + t
            
            # Geometric Jacobian for revolute joint
            J[0:3, i] = z_i                           # angular velocity
            J[3:6, i] = np.cross(z_i, ee_pos - o_i)   # linear velocity
            
            # Update cumulative transform: T_partial *= exp(S_i * q_i)
            # exp(S_i * q_i) for revolute = rotation about l_i through p_i by q_i
            theta_i = q[i]
            R_i = Rotation.from_rotvec(l_i * theta_i).as_matrix()
            T_joint = np.eye(4)
            T_joint[:3, :3] = R_i
            T_joint[:3, 3] = p_i - R_i @ p_i  # rotation about point p_i
            
            T_partial = T_partial @ T_joint
        
        return J


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def compute_forward_kinematics(joint_angles):
    """Standalone forward kinematics (returns torch or numpy based on input)"""
    if isinstance(joint_angles, torch.Tensor):
        joint_angles_np = joint_angles.cpu().numpy()
        return_torch = True
    else:
        joint_angles_np = np.array(joint_angles)
        return_torch = False
    
    kin = _get_kinematics()
    ee_pos = kin.forward_kinematics(joint_angles_np)
    
    if return_torch:
        return torch.tensor(ee_pos, dtype=torch.float32)
    return ee_pos


def forward_kinematics_simple(joint_angles):
    """Position-only FK"""
    if hasattr(joint_angles, 'numpy'):
        q = joint_angles.numpy()
    elif isinstance(joint_angles, torch.Tensor):
        q = joint_angles.cpu().numpy()
    else:
        q = np.array(joint_angles)
    
    kin = _get_kinematics()
    return kin.forward_kinematics(q)


def forward_kinematics_6dof(joint_angles):
    """Full 6DOF FK"""
    if hasattr(joint_angles, 'numpy'):
        q = joint_angles.numpy()
    elif isinstance(joint_angles, torch.Tensor):
        q = joint_angles.cpu().numpy()
    else:
        q = np.array(joint_angles)
    
    kin = _get_kinematics()
    return kin.forward_kinematics_6dof(q)


def inverse_kinematics_6dof(target_position, target_quaternion, initial_guess=None,
                            position_weight=10.0, orientation_weight=2.0):
    """
    Full 6DOF IK with configurable weights.
    
    Args:
        target_position: [3] target position
        target_quaternion: [4] target quaternion [w,x,y,z]
        initial_guess: [7] initial joint angles
        position_weight: Weight for position error
        orientation_weight: Weight for orientation error
    
    Returns:
        [7] joint angles
    """
    kin = _get_kinematics()
    return kin.inverse_kinematics_6dof(target_position, target_quaternion, initial_guess,
                                        position_weight, orientation_weight)


def inverse_kinematics_numerical_scipy(target_position, initial_guess=None, method='SLSQP'):
    """Position-only numerical IK with multi-restart"""
    target_position = np.array(target_position).flatten()[:3]
    kin = _get_kinematics()
    
    if initial_guess is None:
        x, y, z = target_position
        r = np.sqrt(x**2 + y**2)
        q1 = np.arctan2(y, x)
        q2 = np.arctan2(z - 0.3, r) * 0.5
        q4 = np.pi * 0.6
        initial_guess = np.array([q1, q2, 0.0, q4, 0.0, -0.3, 0.0])
    
    bounds = Bounds(kin.q_min, kin.q_max)
    
    def objective(q):
        current_pos = forward_kinematics_simple(q)
        return np.linalg.norm(current_pos - target_position)
    
    def jacobian(q):
        epsilon = 1e-6
        grad = np.zeros(7)
        base_error = objective(q)
        for i in range(7):
            q_perturb = q.copy()
            q_perturb[i] += epsilon
            grad[i] = (objective(q_perturb) - base_error) / epsilon
        return grad
    
    result = minimize(
        objective, initial_guess, method=method, jac=jacobian, bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
    )
    
    if result.success and result.fun < 0.05:
        return result.x
    
    for _ in range(3):
        random_guess = initial_guess + np.random.randn(7) * 0.3
        random_guess = np.clip(random_guess, kin.q_min, kin.q_max)
        
        result = minimize(
            objective, random_guess, method=method, jac=jacobian, bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
        )
        
        if result.success and result.fun < 0.05:
            return result.x
    
    return result.x


# ============================================================================
# QUATERNION UTILITIES
# ============================================================================

def quaternion_distance(q1, q2):
    """
    Angular distance between quaternions in RADIANS.
    
    Handles the double-cover property: q and -q represent the same rotation.
    
    Args:
        q1, q2: Quaternions as [w, x, y, z]
    
    Returns:
        Angular distance in radians [0, pi]
    """
    q1 = np.array(q1).flatten()[:4]
    q2 = np.array(q2).flatten()[:4]
    
    n1 = np.linalg.norm(q1)
    n2 = np.linalg.norm(q2)
    if n1 < 1e-10 or n2 < 1e-10:
        return np.pi
    
    q1 = q1 / n1
    q2 = q2 / n2
    
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, 0.0, 1.0)
    
    return 2.0 * np.arccos(dot)


def quaternion_to_euler(q):
    """Convert quaternion [w,x,y,z] to Euler [roll, pitch, yaw]"""
    q = np.array(q).flatten()[:4]
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    return rot.as_euler('xyz', degrees=False)


def euler_to_quaternion(euler):
    """Convert Euler [roll, pitch, yaw] to quaternion [w,x,y,z]"""
    rot = Rotation.from_euler('xyz', euler)
    quat_xyzw = rot.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


# ============================================================================
# ACHIEVABLE ORIENTATION FINDER
# ============================================================================

def find_achievable_orientation(target_position, desired_orientation=None, n_samples=200,
                                 position_tolerance=0.005):
    """
    Find an achievable orientation at the target position CLOSEST to desired.
    
    Uses both position-only IK sampling and 6DOF IK with various weights.
    
    Args:
        target_position: [3] target position
        desired_orientation: [4] desired quaternion [w,x,y,z] or None (defaults to identity)
        n_samples: number of IK samples to try
        position_tolerance: maximum position error in meters (default 5mm)
    
    Returns:
        achievable_quat: [4] achievable quaternion
        position_error: position error of the solution
        orientation_error: angular distance to desired (radians)
        joint_solution: [7] joint angles achieving this pose
    """
    kin = _get_kinematics()
    
    target_position = np.array(target_position).flatten()[:3]
    
    if desired_orientation is None:
        desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        desired_orientation = np.array(desired_orientation).flatten()[:4]
        desired_orientation = desired_orientation / (np.linalg.norm(desired_orientation) + 1e-12)
    
    valid_solutions = []
    
    # Method 1: Sample many position-only IK solutions
    for i in range(n_samples):
        q_init = np.random.uniform(kin.q_min, kin.q_max)
        q_sol = kin.inverse_kinematics(target_position, initial_guess=q_init)
        pos, quat = kin.forward_kinematics_6dof(q_sol)
        
        pos_error = np.linalg.norm(pos - target_position)
        
        if pos_error < position_tolerance:
            orient_error = quaternion_distance(quat, desired_orientation)
            valid_solutions.append({
                'q': q_sol.copy(),
                'quat': quat.copy(),
                'pos_error': pos_error,
                'orient_error': orient_error
            })
    
    # Method 2: Try 6DOF IK with different orientation weights
    for orient_weight in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        for _ in range(10):
            q_init = np.random.uniform(kin.q_min, kin.q_max)
            
            try:
                q_sol = kin.inverse_kinematics_6dof(
                    target_position, desired_orientation, initial_guess=q_init,
                    position_weight=10.0, orientation_weight=orient_weight
                )
                
                pos, quat = kin.forward_kinematics_6dof(q_sol)
                pos_error = np.linalg.norm(pos - target_position)
                
                if pos_error < position_tolerance:
                    orient_error = quaternion_distance(quat, desired_orientation)
                    valid_solutions.append({
                        'q': q_sol.copy(),
                        'quat': quat.copy(),
                        'pos_error': pos_error,
                        'orient_error': orient_error
                    })
            except Exception:
                pass
    
    if len(valid_solutions) == 0:
        # Fallback: relaxed tolerance
        q_sol = kin.inverse_kinematics(target_position)
        pos, quat = kin.forward_kinematics_6dof(q_sol)
        pos_error = np.linalg.norm(pos - target_position)
        orient_error = quaternion_distance(quat, desired_orientation)
        return quat, pos_error, orient_error, q_sol
    
    best = min(valid_solutions, key=lambda x: x['orient_error'])
    return best['quat'], best['pos_error'], best['orient_error'], best['q']


def find_achievable_orientation_at_position(target_position, n_samples=300,
                                             desired_orientation=None):
    """
    Find achievable orientation at position, closest to desired.
    
    Args:
        target_position: [3] target position
        n_samples: number of samples
        desired_orientation: [4] desired quaternion (default: identity)
    
    Returns:
        achievable_quat: [4] quaternion
        joint_solution: [7] joint angles
    """
    if desired_orientation is None:
        desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    
    achievable_quat, pos_err, orient_err, joints = find_achievable_orientation(
        target_position,
        desired_orientation=desired_orientation,
        n_samples=n_samples,
        position_tolerance=0.005
    )
    
    return achievable_quat, joints


def find_best_achievable_orientation(target_position, desired_orientation, n_samples=500):
    """
    Extensive search for best achievable orientation closest to desired.
    
    Args:
        target_position: [3] target position
        desired_orientation: [4] desired quaternion
        n_samples: number of samples (default 500)
    
    Returns:
        achievable_quat, position_error, orientation_error, joint_solution
    """
    return find_achievable_orientation(
        target_position,
        desired_orientation=desired_orientation,
        n_samples=n_samples,
        position_tolerance=0.003
    )


# ============================================================================
# TESTING
# ============================================================================

def test_achievable_orientations():
    """Test achievable orientation finder"""
    print("\n" + "="*70)
    print("TESTING ACHIEVABLE ORIENTATION FINDER")
    print("="*70)
    
    test_positions = [
        np.array([0.4, 0.2, 0.5]),
        np.array([0.5, 0.1, 0.4]),
        np.array([0.45, 0.15, 0.55]),
    ]
    
    desired_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    for pos in test_positions:
        print(f"\n--- Position: {pos} ---")
        
        achievable_quat, pos_err, orient_err, joints = find_best_achievable_orientation(
            pos, desired_quat, n_samples=500
        )
        
        euler = np.rad2deg(quaternion_to_euler(achievable_quat))
        print(f"   Achievable quaternion: {achievable_quat}")
        print(f"   Euler (deg): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]")
        print(f"   Position error: {pos_err*1000:.2f} mm")
        print(f"   Orientation error from identity: {np.rad2deg(orient_err):.1f}°")
        
        if joints is not None:
            verify_pos, verify_quat = forward_kinematics_6dof(joints)
            verify_pos_err = np.linalg.norm(verify_pos - pos)
            verify_orient_err = quaternion_distance(verify_quat, achievable_quat)
            print(f"   Verification pos error: {verify_pos_err*1000:.2f} mm")
            print(f"   Verification orient error: {np.rad2deg(verify_orient_err):.2f}°")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("="*70)
    print("TESTING SPACE ROBOT KINEMATICS")
    print("="*70)
    
    # Test FK
    print("\n1. Testing Forward Kinematics...")
    kin = SpaceRobotKinematics()
    
    q_zero = np.zeros(7)
    ee_zero = kin.forward_kinematics(q_zero)
    print(f"   Zero config EE: {ee_zero}")
    print(f"   Expected:       [0, 0, {kin.d1+kin.d3+kin.d5+kin.d7:.3f}]")
    
    q_test = np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0])
    ee_pos = kin.forward_kinematics(q_test)
    print(f"   Test config EE: {ee_pos}")
    
    # Test IK
    print("\n2. Testing Inverse Kinematics...")
    target = np.array([0.4, 0.2, 0.5])
    q_solution = kin.inverse_kinematics(target)
    ee_achieved = kin.forward_kinematics(q_solution)
    ik_error = np.linalg.norm(ee_achieved - target)
    print(f"   Target: {target}")
    print(f"   Achieved: {ee_achieved}")
    print(f"   Error: {ik_error*1000:.4f} mm")
    
    # Test achievable orientations
    print("\n3. Testing Achievable Orientation Finder...")
    test_achievable_orientations()
    
    print("\n" + "="*70)
    print("✅ ALL KINEMATICS TESTS COMPLETED")
    print("="*70)
