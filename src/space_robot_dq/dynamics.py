"""
Dynamics and momentum conservation for free-floating space robot.

In the absence of external forces/torques, the total system momentum is conserved:
    H_b * ẋ_b + H_bm * q̇ = h_0

where:
    H_b  (6×6):  base spatial inertia
    H_bm (6×7):  base-manipulator coupling inertia
    ẋ_b  (6×1):  base velocity [v_b; ω_b]
    q̇    (7×1):  joint velocities
    h_0  (6×1):  initial momentum (typically zero)

The generalized Jacobian maps joint velocities to EE velocity
accounting for the dynamic coupling (base reaction):
    ẋ_ee = J_g * q̇ = (J_m - J_b * H_b⁻¹ * H_bm) * q̇

Key references:
    - Umetani & Yoshida (1989), "Resolved motion rate control of space
      manipulators with generalized Jacobian matrix"
    - Dubowsky & Papadopoulos (1993), "The kinematics, dynamics, and
      control of free-flying and free-floating space robotic systems"

Author: Hadi Jahanshahi, Zheng H. Zhu
Affiliation: Department of Mechanical Engineering, York University
Date: 2025
"""

import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import SpaceRobotKinematics, _get_kinematics


def skew(v):
    """
    Skew-symmetric matrix from 3D vector.
    
    S(v) × a = v × a for any vector a.
    
    Args:
        v: 3D vector
    
    Returns:
        3×3 skew-symmetric matrix
    """
    v = np.asarray(v).flatten()
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])


class SpaceRobotDynamics:
    """
    Dynamics for a 7-DOF free-floating space robot with momentum conservation.
    
    Models the dynamic coupling between the manipulator and the free-floating
    base. When the manipulator moves, the base reacts to conserve total system
    momentum. The generalized Jacobian captures this coupling.
    
    Bodies:
        Body 0 (base): free-floating platform
        Bodies 1-7 (links): serial chain from shoulder to end-effector
    
    Each link i (i=0..6) is rigidly attached after joint i. Its motion
    depends on joints 0..i and the base.
    """
    
    def __init__(self, kinematics=None,
                 base_mass=100.0, base_inertia=None,
                 link_masses=None, link_inertias=None):
        """
        Initialize dynamics with mass properties.
        
        Args:
            kinematics: SpaceRobotKinematics instance (creates default if None)
            base_mass: base platform mass in kg (default 100)
            base_inertia: 3×3 base inertia about COM in body frame (kg·m²)
            link_masses: list of 7 link masses in kg
            link_inertias: list of 7 inertia tensors (3×3, about COM, body frame)
        """
        self.kin = kinematics if kinematics is not None else _get_kinematics()
        self.n_joints = self.kin.num_joints
        
        # ---- Base properties ----
        self.base_mass = base_mass
        self.base_inertia = np.array(base_inertia) if base_inertia is not None \
            else np.diag([10.0, 10.0, 8.0])
        self.base_com_local = np.array([0.0, 0.0, 0.0])  # COM at base origin
        
        # ---- Link dimensions (from kinematics) ----
        d1 = self.kin.d1
        d3 = self.kin.d3
        d5 = self.kin.d5
        d7 = self.kin.d7
        
        # ---- Link masses ----
        if link_masses is not None:
            self.link_masses = np.array(link_masses, dtype=float)
        else:
            # Defaults: ~28 kg total arm mass, reasonable for space manipulator
            self.link_masses = np.array([5.0, 3.0, 7.0, 3.0, 6.0, 2.0, 2.0])
        
        # ---- Link COM positions in HOME frame (space frame at q=0) ----
        # Link i is the body attached after joint i.
        # Colocated joints (shoulder pitch/roll, etc.) have COM at joint location.
        self.link_coms_home = np.array([
            [0, 0, d1 / 2],                    # Link 0: shoulder yaw body (midpoint)
            [0, 0, d1],                         # Link 1: shoulder pitch body (at joint)
            [0, 0, d1 + d3 / 2],                # Link 2: upper arm (midpoint)
            [0, 0, d1 + d3],                    # Link 3: elbow body (at joint)
            [0, 0, d1 + d3 + d5 / 2],           # Link 4: forearm (midpoint)
            [0, 0, d1 + d3 + d5],               # Link 5: wrist body (at joint)
            [0, 0, d1 + d3 + d5 + d7 / 2],      # Link 6: end-effector (midpoint)
        ])
        
        # ---- Link inertias (about COM, body frame) ----
        if link_inertias is not None:
            self.link_inertias = [np.array(I, dtype=float) for I in link_inertias]
        else:
            # Cylinder approximation for each link
            link_lengths = [d1, 0.05, d3, 0.05, d5, 0.05, d7]
            radius = 0.04  # ~4 cm radius
            self.link_inertias = []
            for i in range(self.n_joints):
                m = self.link_masses[i]
                L = link_lengths[i]
                r = radius
                Ixx = m * (3 * r**2 + L**2) / 12
                Iyy = Ixx
                Izz = m * r**2 / 2
                self.link_inertias.append(np.diag([Ixx, Iyy, Izz]))
        
        self.total_mass = self.base_mass + np.sum(self.link_masses)
        
        # Initial momentum (zero = system starts at rest)
        self.h0 = np.zeros(6)
    
    # ================================================================
    # INTERMEDIATE COMPUTATIONS
    # ================================================================
    
    def compute_joint_transforms(self, q):
        """
        Compute cumulative 4×4 transforms at each joint.
        
        T[0] = T_base                                    (before any joint)
        T[i+1] = T_base * exp(S0*q0) * ... * exp(Si*qi)  (after joint i)
        
        Args:
            q: 7D joint angles
        
        Returns:
            list of 8 homogeneous transforms (4×4)
        """
        q = np.array(q).flatten()[:self.n_joints]
        
        transforms = []
        T = self.kin.base_dq.to_matrix()
        transforms.append(T.copy())
        
        for i in range(self.n_joints):
            l_i = self.kin.screw_axes[i]['l']
            p_i = self.kin.screw_axes[i]['p']
            theta_i = q[i]
            
            R_i = Rotation.from_rotvec(l_i * theta_i).as_matrix()
            T_joint = np.eye(4)
            T_joint[:3, :3] = R_i
            T_joint[:3, 3] = p_i - R_i @ p_i  # rotation about point p_i
            
            T = T @ T_joint
            transforms.append(T.copy())
        
        return transforms
    
    def compute_link_states(self, q):
        """
        Compute COM positions and orientations of all links in world frame.
        
        Args:
            q: 7D joint angles
        
        Returns:
            coms: (7, 3) COM positions in world frame
            rotations: list of 7 rotation matrices (3×3)
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
        Compute the Jacobian of each link's COM w.r.t. joint angles.
        
        For link i, the Jacobian maps q̇ to [ω_i; v_com_i]:
            [ω_i    ]   [J_ω_i]
            [v_com_i] = [J_v_i] * q̇
        
        Only joints j ≤ i contribute to link i's motion (serial chain).
        
        Args:
            q: 7D joint angles
        
        Returns:
            list of 7 Jacobians, each (6×7), rows [angular; linear]
        """
        q = np.array(q).flatten()[:self.n_joints]
        transforms = self.compute_joint_transforms(q)
        coms, _ = self.compute_link_states(q)
        
        jacobians = []
        
        for i in range(self.n_joints):
            J_i = np.zeros((6, self.n_joints))
            p_com_i = coms[i]
            
            for j in range(i + 1):
                T_j = transforms[j]
                R_j = T_j[:3, :3]
                t_j = T_j[:3, 3]
                
                l_j = self.kin.screw_axes[j]['l']
                p_j = self.kin.screw_axes[j]['p']
                
                # Current axis direction and position of joint j
                z_j = R_j @ l_j
                z_j = z_j / (np.linalg.norm(z_j) + 1e-12)
                o_j = R_j @ p_j + t_j
                
                J_i[0:3, j] = z_j                              # angular
                J_i[3:6, j] = np.cross(z_j, p_com_i - o_j)     # linear
            
            jacobians.append(J_i)
        
        return jacobians
    
    # ================================================================
    # INERTIA MATRICES
    # ================================================================
    
    def compute_inertia_matrices(self, q):
        """
        Compute the spatial inertia matrices for the free-floating system.
        
        The momentum equation is:
            [P]       [v_b]         
            [L] = H_b [ω_b] + H_bm * q̇  =  h_0
        
        where P is linear momentum, L is angular momentum about the base.
        
        Args:
            q: 7D joint angles
        
        Returns:
            H_b:  (6×6) base spatial inertia (symmetric positive-definite)
            H_bm: (6×7) base-manipulator coupling inertia
        """
        q = np.array(q).flatten()[:self.n_joints]
        
        R_base, p_base = self.kin.base_dq.to_pose()
        coms, rotations = self.compute_link_states(q)
        link_jacobians = self.compute_link_jacobians(q)
        
        H_b = np.zeros((6, 6))
        H_bm = np.zeros((6, self.n_joints))
        
        # ---- Base body contribution to H_b ----
        r_0 = R_base @ self.base_com_local  # base COM offset from base origin
        S_r0 = skew(r_0)
        I_0_world = R_base @ self.base_inertia @ R_base.T
        
        H_b[:3, :3] += self.base_mass * np.eye(3)
        H_b[:3, 3:6] += -self.base_mass * S_r0
        H_b[3:6, :3] += self.base_mass * S_r0
        H_b[3:6, 3:6] += I_0_world + self.base_mass * (
            np.dot(r_0, r_0) * np.eye(3) - np.outer(r_0, r_0)
        )
        # Note: the last term is the parallel axis theorem:
        # ||r||²I - r⊗r = -S(r)²
        
        # ---- Link contributions to H_b and H_bm ----
        for i in range(self.n_joints):
            m_i = self.link_masses[i]
            r_i = coms[i] - p_base                        # base → link COM
            S_ri = skew(r_i)
            I_i_world = rotations[i] @ self.link_inertias[i] @ rotations[i].T
            
            # H_b: spatial inertia contribution
            H_b[:3, :3] += m_i * np.eye(3)
            H_b[:3, 3:6] += -m_i * S_ri
            H_b[3:6, :3] += m_i * S_ri
            H_b[3:6, 3:6] += I_i_world + m_i * (
                np.dot(r_i, r_i) * np.eye(3) - np.outer(r_i, r_i)
            )
            
            # H_bm: coupling between base and manipulator
            J_v_i = link_jacobians[i][3:6, :]   # linear Jacobian of COM_i
            J_w_i = link_jacobians[i][0:3, :]   # angular Jacobian of link i
            
            H_bm[:3, :] += m_i * J_v_i
            H_bm[3:6, :] += I_i_world @ J_w_i + m_i * S_ri @ J_v_i
        
        return H_b, H_bm
    
    # ================================================================
    # MOMENTUM CONSERVATION
    # ================================================================
    
    def compute_base_velocity(self, q, qdot):
        """
        Compute base velocity from momentum conservation.
        
        Solves: H_b * ẋ_b = h_0 - H_bm * q̇
                ẋ_b = H_b⁻¹ * (h_0 - H_bm * q̇)
        
        For zero initial momentum (h_0 = 0):
                ẋ_b = -H_b⁻¹ * H_bm * q̇
        
        Args:
            q: 7D joint angles
            qdot: 7D joint velocities
        
        Returns:
            xb_dot: [6] base spatial velocity [v_b; ω_b]
        """
        q = np.array(q).flatten()[:self.n_joints]
        qdot = np.array(qdot).flatten()[:self.n_joints]
        
        H_b, H_bm = self.compute_inertia_matrices(q)
        
        xb_dot = np.linalg.solve(H_b, self.h0 - H_bm @ qdot)
        return xb_dot
    
    def compute_system_momentum(self, q, qdot, xb_dot):
        """
        Compute total system momentum (for verification).
        
        Should equal h_0 when xb_dot is computed from momentum conservation.
        
        Args:
            q: 7D joint angles
            qdot: 7D joint velocities
            xb_dot: [6] base velocity [v_b; ω_b]
        
        Returns:
            h: [6] total momentum [P; L]
        """
        H_b, H_bm = self.compute_inertia_matrices(q)
        return H_b @ xb_dot + H_bm @ qdot
    
    # ================================================================
    # GENERALIZED JACOBIAN
    # ================================================================
    
    def compute_generalized_jacobian(self, q):
        """
        Compute the generalized Jacobian accounting for base reaction.
        
        Standard (fixed-base) Jacobian:
            ẋ_ee = J_m * q̇  (assumes base is stationary)
        
        Generalized Jacobian (free-floating):
            ẋ_ee = J_g * q̇ = (J_m - J_b * H_b⁻¹ * H_bm) * q̇
        
        The correction term J_b * H_b⁻¹ * H_bm captures the fact that
        the base moves in reaction to manipulator motion, which in turn
        shifts the end-effector.
        
        Args:
            q: 7D joint angles
        
        Returns:
            J_g: (6×7) generalized Jacobian
            J_m: (6×7) fixed-base manipulator Jacobian
            J_b: (6×6) base-to-EE Jacobian
        """
        q = np.array(q).flatten()[:self.n_joints]
        
        # Fixed-base manipulator Jacobian
        J_m = self.kin.calculate_jacobian(q)
        
        # Inertia matrices
        H_b, H_bm = self.compute_inertia_matrices(q)
        
        # Base-to-EE Jacobian: how base motion affects EE
        # v_ee = v_b + ω_b × r_ee   → linear part
        # ω_ee = ω_b                 → angular part
        p_ee = self.kin.forward_kinematics(q)
        _, p_base = self.kin.base_dq.to_pose()
        r_ee = p_ee - p_base
        
        J_b = np.zeros((6, 6))
        J_b[0:3, 0:3] = np.zeros((3, 3))     # ω_ee independent of v_b
        J_b[0:3, 3:6] = np.eye(3)             # ω_ee = ω_b (direct transfer)
        J_b[3:6, 0:3] = np.eye(3)             # v_ee depends on v_b directly
        J_b[3:6, 3:6] = -skew(r_ee)           # v_ee += ω_b × r_ee
        
        # Generalized Jacobian
        H_b_inv = np.linalg.inv(H_b)
        J_g = J_m - J_b @ H_b_inv @ H_bm
        
        return J_g, J_m, J_b
    
    # ================================================================
    # UTILITY
    # ================================================================
    
    def compute_system_com(self, q):
        """
        Compute the total system center of mass.
        
        For a free-floating system with zero linear momentum,
        the system COM is stationary.
        
        Args:
            q: 7D joint angles
        
        Returns:
            com: [3] system COM position in world frame
        """
        R_base, p_base = self.kin.base_dq.to_pose()
        
        base_com = R_base @ self.base_com_local + p_base
        coms, _ = self.compute_link_states(q)
        
        com = self.base_mass * base_com
        for i in range(self.n_joints):
            com += self.link_masses[i] * coms[i]
        com /= self.total_mass
        
        return com
    
    def compute_dynamic_manipulability(self, q):
        """
        Compute dynamic manipulability using the generalized Jacobian.
        
        w = sqrt(det(J_g * J_g'))
        
        This measures the end-effector's velocity capability
        accounting for base reaction (lower than fixed-base manipulability).
        
        Args:
            q: 7D joint angles
        
        Returns:
            w_generalized: dynamic manipulability
            w_fixed: fixed-base manipulability (for comparison)
        """
        J_g, J_m, _ = self.compute_generalized_jacobian(q)
        
        w_generalized = np.sqrt(max(0, np.linalg.det(J_g @ J_g.T)))
        w_fixed = np.sqrt(max(0, np.linalg.det(J_m @ J_m.T)))
        
        return w_generalized, w_fixed
    
    def get_mass_properties_summary(self):
        """Print summary of all mass properties."""
        print(f"\n{'='*60}")
        print("MASS PROPERTIES SUMMARY")
        print(f"{'='*60}")
        print(f"  Base mass:       {self.base_mass:.1f} kg")
        print(f"  Base inertia:    diag({np.diag(self.base_inertia)})")
        print(f"  Total arm mass:  {np.sum(self.link_masses):.1f} kg")
        print(f"  Total system:    {self.total_mass:.1f} kg")
        print(f"  Mass ratio (arm/base): {np.sum(self.link_masses)/self.base_mass:.3f}")
        print(f"\n  {'Link':<6} {'Name':<16} {'Mass(kg)':<10} {'COM_home (m)'}")
        print(f"  {'-'*55}")
        for i in range(self.n_joints):
            name = self.kin.screw_axes[i]['name']
            m = self.link_masses[i]
            c = self.link_coms_home[i]
            print(f"  {i:<6} {name:<16} {m:<10.1f} [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
        print(f"{'='*60}")


# ============================================================================
# STANDALONE CONVENIENCE FUNCTIONS
# ============================================================================

_cached_dynamics = None

def _get_dynamics():
    """Get cached dynamics instance."""
    global _cached_dynamics
    if _cached_dynamics is None:
        _cached_dynamics = SpaceRobotDynamics()
    return _cached_dynamics


def compute_generalized_jacobian(q):
    """
    Compute generalized Jacobian at configuration q.
    
    Returns:
        J_g: (6×7) generalized Jacobian
    """
    dyn = _get_dynamics()
    J_g, _, _ = dyn.compute_generalized_jacobian(q)
    return J_g


def compute_base_reaction(q, qdot):
    """
    Compute base velocity from momentum conservation.
    
    Args:
        q: 7D joint angles
        qdot: 7D joint velocities
    
    Returns:
        xb_dot: [6] base velocity [v_b; ω_b]
    """
    dyn = _get_dynamics()
    return dyn.compute_base_velocity(q, qdot)
