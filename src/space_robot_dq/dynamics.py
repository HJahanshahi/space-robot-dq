"""
Dynamics and momentum conservation for N-DOF free-floating space robot.

Generalizes to arbitrary serial chain robots. The user provides mass
properties per link; if omitted, defaults are generated from geometry.

Key equations (zero initial momentum):
    H_b · ẋ_b + H_bm · q̇ = 0
    J_g = J_m - J_b · H_b⁻¹ · H_bm

References:
    - Umetani & Yoshida (1989), IEEE Trans. Robotics and Automation
    - Dubowsky & Papadopoulos (1993), IEEE Trans. Robotics and Automation

Author: Hadi Jahanshahi, Zheng H. Zhu
Affiliation: Department of Mechanical Engineering, York University
Date: 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.spatial.transform import Rotation
from .kinematics import SpaceRobotKinematics, RobotConfig, _get_kinematics


def skew(v):
    """3×3 skew-symmetric matrix from vector v: S(v)·a = v×a."""
    v = np.asarray(v).flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


@dataclass
class LinkProperties:
    """
    Mass properties for a single link body.
    
    Attributes:
        mass: link mass (kg)
        com_home: (3,) center of mass in home frame (meters)
        inertia: (3,3) inertia tensor about COM in body frame (kg·m²)
    """
    mass: float
    com_home: np.ndarray
    inertia: np.ndarray
    
    def __post_init__(self):
        self.com_home = np.array(self.com_home, dtype=float)
        self.inertia = np.array(self.inertia, dtype=float)


def default_link_properties(config: RobotConfig):
    """
    Generate default link mass properties from robot geometry.
    
    Each link's COM is placed between its joint and the next joint
    (or the EE for the last link). Mass is proportional to link length.
    Inertia uses a cylinder approximation.
    
    Args:
        config: RobotConfig
    
    Returns:
        list of LinkProperties
    """
    n = config.num_joints
    radius = 0.04  # 4 cm cylinder radius
    density_per_meter = 15.0  # kg per meter of link
    
    props = []
    for i in range(n):
        p_i = config.joints[i].position
        if i + 1 < n:
            p_next = config.joints[i + 1].position
        else:
            p_next = config.ee_position
        
        # Link length and COM
        link_vec = p_next - p_i
        link_length = max(np.linalg.norm(link_vec), 0.05)  # min 5 cm
        com = (p_i + p_next) / 2.0
        
        # Mass proportional to length
        mass = density_per_meter * link_length
        mass = max(mass, 1.0)  # minimum 1 kg
        
        # Cylinder inertia about COM
        L = link_length
        r = radius
        Ixx = mass * (3 * r**2 + L**2) / 12
        Iyy = Ixx
        Izz = mass * r**2 / 2
        
        props.append(LinkProperties(
            mass=mass,
            com_home=com,
            inertia=np.diag([Ixx, Iyy, Izz])
        ))
    
    return props


class SpaceRobotDynamics:
    """
    Dynamics for an N-DOF free-floating space robot with momentum conservation.
    
    Computes:
    - Spatial inertia matrices (H_b, H_bm)
    - Base reaction velocity from momentum conservation
    - Generalized Jacobian (Umetani-Yoshida)
    - Dynamic manipulability
    
    Args:
        kinematics: SpaceRobotKinematics instance
        base_mass: spacecraft mass (kg)
        base_inertia: (3,3) base inertia tensor (kg·m²)
        link_properties: list of LinkProperties, one per joint/link
            If None, defaults are generated from geometry.
    
    Example::
    
        kin = SpaceRobotKinematics()
        dyn = SpaceRobotDynamics(kinematics=kin, base_mass=100.0)
        
        J_g, J_m, J_b = dyn.compute_generalized_jacobian(q)
        xb_dot = dyn.compute_base_velocity(q, qdot)
    """
    
    def __init__(self, kinematics=None, base_mass=100.0, base_inertia=None,
                 link_properties=None,
                 # Legacy API: these are used only if link_properties is None
                 link_masses=None, link_inertias=None):
        
        self.kin = kinematics if kinematics is not None else _get_kinematics()
        self.n_joints = self.kin.num_joints
        
        # Base properties
        self.base_mass = base_mass
        self.base_inertia = np.array(base_inertia) if base_inertia is not None \
            else np.diag([base_mass * 0.1, base_mass * 0.1, base_mass * 0.08])
        self.base_com_local = np.array([0.0, 0.0, 0.0])
        
        # Link properties
        if link_properties is not None:
            self.link_props = link_properties
        elif link_masses is not None:
            # Legacy API: build from separate arrays
            link_masses = np.array(link_masses, dtype=float)
            self.link_props = []
            for i in range(self.n_joints):
                # Estimate COM
                p_i = self.kin.config.joints[i].position
                if i + 1 < self.n_joints:
                    p_next = self.kin.config.joints[i + 1].position
                else:
                    p_next = self.kin.config.ee_position
                com = (p_i + p_next) / 2.0
                
                # Use provided inertia or default
                if link_inertias is not None:
                    I_i = np.array(link_inertias[i], dtype=float)
                else:
                    L = max(np.linalg.norm(p_next - p_i), 0.05)
                    m = link_masses[i]
                    r = 0.04
                    Ixx = m * (3 * r**2 + L**2) / 12
                    I_i = np.diag([Ixx, Ixx, m * r**2 / 2])
                
                self.link_props.append(LinkProperties(
                    mass=link_masses[i], com_home=com, inertia=I_i))
        else:
            self.link_props = default_link_properties(self.kin.config)
        
        # Derived quantities
        self.link_masses = np.array([lp.mass for lp in self.link_props])
        self.link_coms_home = np.array([lp.com_home for lp in self.link_props])
        self.link_inertias = [lp.inertia for lp in self.link_props]
        self.total_mass = self.base_mass + np.sum(self.link_masses)
        
        # Initial momentum (zero = system starts at rest)
        self.h0 = np.zeros(6)
    
    # ================================================================
    # INTERMEDIATE COMPUTATIONS
    # ================================================================
    
    def compute_joint_transforms(self, q):
        """
        Cumulative 4×4 transforms at each joint stage.
        
        Returns:
            list of (N+1) transforms: T[0]=base, T[i+1]=after joint i
        """
        q = np.array(q).flatten()[:self.n_joints]
        transforms = []
        T = self.kin.base_dq.to_matrix()
        transforms.append(T.copy())
        
        for i in range(self.n_joints):
            l_i = self.kin.screw_axes[i]['l']
            p_i = self.kin.screw_axes[i]['p']
            R_i = Rotation.from_rotvec(l_i * q[i]).as_matrix()
            T_joint = np.eye(4)
            T_joint[:3, :3] = R_i
            T_joint[:3, 3] = p_i - R_i @ p_i
            T = T @ T_joint
            transforms.append(T.copy())
        
        return transforms
    
    def compute_link_states(self, q):
        """
        COM positions and orientations for all links in world frame.
        
        Returns:
            coms: (N, 3) COM positions
            rotations: list of N rotation matrices (3×3)
        """
        transforms = self.compute_joint_transforms(q)
        coms = np.zeros((self.n_joints, 3))
        rotations = []
        
        for i in range(self.n_joints):
            T_i = transforms[i + 1]
            R_i = T_i[:3, :3]
            t_i = T_i[:3, 3]
            coms[i] = R_i @ self.link_coms_home[i] + t_i
            rotations.append(R_i)
        
        return coms, rotations
    
    def compute_link_jacobians(self, q):
        """
        Jacobian of each link's COM w.r.t. joint angles.
        
        Returns:
            list of N Jacobians, each (6×N), rows [angular; linear]
        """
        q = np.array(q).flatten()[:self.n_joints]
        transforms = self.compute_joint_transforms(q)
        coms, _ = self.compute_link_states(q)
        
        jacobians = []
        for i in range(self.n_joints):
            J_i = np.zeros((6, self.n_joints))
            for j in range(i + 1):
                T_j = transforms[j]
                R_j = T_j[:3, :3]
                t_j = T_j[:3, 3]
                l_j = self.kin.screw_axes[j]['l']
                p_j = self.kin.screw_axes[j]['p']
                
                z_j = R_j @ l_j
                z_j = z_j / (np.linalg.norm(z_j) + 1e-12)
                o_j = R_j @ p_j + t_j
                
                J_i[0:3, j] = z_j
                J_i[3:6, j] = np.cross(z_j, coms[i] - o_j)
            
            jacobians.append(J_i)
        
        return jacobians
    
    # ================================================================
    # INERTIA MATRICES
    # ================================================================
    
    def compute_inertia_matrices(self, q):
        """
        Spatial inertia matrices for the free-floating system.
        
        Returns:
            H_b:  (6×6) base spatial inertia (SPD)
            H_bm: (6×N) base-manipulator coupling
        """
        q = np.array(q).flatten()[:self.n_joints]
        R_base, p_base = self.kin.base_dq.to_pose()
        coms, rotations = self.compute_link_states(q)
        link_jacs = self.compute_link_jacobians(q)
        
        H_b = np.zeros((6, 6))
        H_bm = np.zeros((6, self.n_joints))
        
        # Base body
        r_0 = R_base @ self.base_com_local
        S_r0 = skew(r_0)
        I_0_world = R_base @ self.base_inertia @ R_base.T
        
        H_b[:3, :3] += self.base_mass * np.eye(3)
        H_b[:3, 3:6] += -self.base_mass * S_r0
        H_b[3:6, :3] += self.base_mass * S_r0
        H_b[3:6, 3:6] += I_0_world + self.base_mass * (
            np.dot(r_0, r_0) * np.eye(3) - np.outer(r_0, r_0))
        
        # Link contributions
        for i in range(self.n_joints):
            m_i = self.link_masses[i]
            r_i = coms[i] - p_base
            S_ri = skew(r_i)
            I_i_world = rotations[i] @ self.link_inertias[i] @ rotations[i].T
            
            H_b[:3, :3] += m_i * np.eye(3)
            H_b[:3, 3:6] += -m_i * S_ri
            H_b[3:6, :3] += m_i * S_ri
            H_b[3:6, 3:6] += I_i_world + m_i * (
                np.dot(r_i, r_i) * np.eye(3) - np.outer(r_i, r_i))
            
            J_v_i = link_jacs[i][3:6, :]
            J_w_i = link_jacs[i][0:3, :]
            H_bm[:3, :] += m_i * J_v_i
            H_bm[3:6, :] += I_i_world @ J_w_i + m_i * S_ri @ J_v_i
        
        return H_b, H_bm
    
    # ================================================================
    # MOMENTUM CONSERVATION
    # ================================================================
    
    def compute_base_velocity(self, q, qdot):
        """
        Base velocity from momentum conservation.
        
        ẋ_b = H_b⁻¹ · (h₀ - H_bm · q̇)
        
        Returns:
            (6,) base velocity [v_b; ω_b]
        """
        q = np.array(q).flatten()[:self.n_joints]
        qdot = np.array(qdot).flatten()[:self.n_joints]
        H_b, H_bm = self.compute_inertia_matrices(q)
        return np.linalg.solve(H_b, self.h0 - H_bm @ qdot)
    
    def compute_system_momentum(self, q, qdot, xb_dot):
        """Total momentum (should equal h₀ when using compute_base_velocity)."""
        H_b, H_bm = self.compute_inertia_matrices(q)
        return H_b @ xb_dot + H_bm @ qdot
    
    # ================================================================
    # GENERALIZED JACOBIAN
    # ================================================================
    
    def compute_generalized_jacobian(self, q):
        """
        Generalized Jacobian accounting for base reaction.
        
        J_g = J_m - J_b · H_b⁻¹ · H_bm
        
        Returns:
            J_g: (6×N) generalized Jacobian
            J_m: (6×N) fixed-base Jacobian
            J_b: (6×6) base-to-EE Jacobian
        """
        q = np.array(q).flatten()[:self.n_joints]
        
        J_m = self.kin.calculate_jacobian(q)
        H_b, H_bm = self.compute_inertia_matrices(q)
        
        p_ee = self.kin.forward_kinematics(q)
        _, p_base = self.kin.base_dq.to_pose()
        r_ee = p_ee - p_base
        
        J_b = np.zeros((6, 6))
        J_b[0:3, 3:6] = np.eye(3)
        J_b[3:6, 0:3] = np.eye(3)
        J_b[3:6, 3:6] = -skew(r_ee)
        
        J_g = J_m - J_b @ np.linalg.inv(H_b) @ H_bm
        return J_g, J_m, J_b
    
    # ================================================================
    # UTILITIES
    # ================================================================
    
    def compute_system_com(self, q):
        """System center of mass in world frame."""
        R_base, p_base = self.kin.base_dq.to_pose()
        base_com = R_base @ self.base_com_local + p_base
        coms, _ = self.compute_link_states(q)
        
        com = self.base_mass * base_com
        for i in range(self.n_joints):
            com += self.link_masses[i] * coms[i]
        return com / self.total_mass
    
    def compute_dynamic_manipulability(self, q):
        """
        Dynamic manipulability: w = sqrt(det(J · J')).
        
        Returns:
            w_generalized: free-floating manipulability
            w_fixed: fixed-base manipulability
        """
        J_g, J_m, _ = self.compute_generalized_jacobian(q)
        w_gen = np.sqrt(max(0, np.linalg.det(J_g @ J_g.T)))
        w_fix = np.sqrt(max(0, np.linalg.det(J_m @ J_m.T)))
        return w_gen, w_fix
    
    def get_mass_properties_summary(self):
        """Print mass properties."""
        print(f"\n{'='*60}")
        print(f"MASS PROPERTIES ({self.kin.config.name})")
        print(f"{'='*60}")
        print(f"  Base mass:       {self.base_mass:.1f} kg")
        print(f"  Total arm mass:  {np.sum(self.link_masses):.1f} kg")
        print(f"  Total system:    {self.total_mass:.1f} kg")
        print(f"  Mass ratio (arm/base): {np.sum(self.link_masses)/self.base_mass:.3f}")
        print(f"\n  {'#':<4} {'Name':<18} {'Mass(kg)':<10} {'COM_home (m)'}")
        print(f"  {'-'*55}")
        for i in range(self.n_joints):
            name = self.kin.screw_axes[i]['name']
            m = self.link_masses[i]
            c = self.link_coms_home[i]
            print(f"  {i:<4} {name:<18} {m:<10.1f} [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
        print(f"{'='*60}")


# ============================================================================
# STANDALONE CONVENIENCE FUNCTIONS
# ============================================================================

_cached_dynamics = None

def _get_dynamics():
    global _cached_dynamics
    if _cached_dynamics is None:
        _cached_dynamics = SpaceRobotDynamics()
    return _cached_dynamics


def compute_generalized_jacobian(q):
    """Generalized Jacobian (default robot)."""
    return _get_dynamics().compute_generalized_jacobian(q)[0]


def compute_base_reaction(q, qdot):
    """Base velocity from momentum conservation (default robot)."""
    return _get_dynamics().compute_base_velocity(q, qdot)
