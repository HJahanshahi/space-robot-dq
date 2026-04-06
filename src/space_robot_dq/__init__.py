"""
space_robot_dq — Dual quaternion kinematics and dynamics for free-floating space robots.

A Python library for modeling 7-DOF free-floating space robot manipulators
using dual quaternion algebra and screw theory.

Features:
    - Singularity-free pose representation via dual quaternions
    - Product of Exponentials (PoE) forward kinematics
    - Numerical and analytical Jacobians
    - 6-DOF inverse kinematics with configurable weights
    - Momentum conservation for free-floating base
    - Generalized Jacobian (Umetani-Yoshida formulation)
    - Dynamic manipulability analysis

Quick start::

    from space_robot_dq import SpaceRobotKinematics, SpaceRobotDynamics

    # Kinematics
    kin = SpaceRobotKinematics()
    pos = kin.forward_kinematics([0, 0.3, 0, 1.2, 0, -0.4, 0])
    pos, quat = kin.forward_kinematics_6dof([0, 0.3, 0, 1.2, 0, -0.4, 0])

    # Dynamics with momentum conservation
    dyn = SpaceRobotDynamics(kinematics=kin)
    J_g = dyn.compute_generalized_jacobian(q)[0]
    xb_dot = dyn.compute_base_velocity(q, qdot)
"""

__version__ = "0.1.0"

# Core classes
from .dual_quaternion import (
    DualQuaternion,
    quaternion_multiply,
    quaternion_multiply_torch,
    quaternion_to_rotation_matrix,
    log_dq,
)

from .kinematics import (
    SpaceRobotKinematics,
    # Standalone functions
    compute_forward_kinematics,
    forward_kinematics_simple,
    forward_kinematics_6dof,
    inverse_kinematics_6dof,
    inverse_kinematics_numerical_scipy,
    # Quaternion utilities
    quaternion_distance,
    quaternion_to_euler,
    euler_to_quaternion,
    # Achievable orientation
    find_achievable_orientation,
    find_achievable_orientation_at_position,
    find_best_achievable_orientation,
)

from .dynamics import (
    SpaceRobotDynamics,
    compute_generalized_jacobian,
    compute_base_reaction,
    skew,
)

__all__ = [
    # Version
    "__version__",
    # Dual quaternion
    "DualQuaternion",
    "quaternion_multiply",
    "quaternion_multiply_torch",
    "quaternion_to_rotation_matrix",
    "log_dq",
    # Kinematics
    "SpaceRobotKinematics",
    "compute_forward_kinematics",
    "forward_kinematics_simple",
    "forward_kinematics_6dof",
    "inverse_kinematics_6dof",
    "inverse_kinematics_numerical_scipy",
    "quaternion_distance",
    "quaternion_to_euler",
    "euler_to_quaternion",
    "find_achievable_orientation",
    "find_achievable_orientation_at_position",
    "find_best_achievable_orientation",
    # Dynamics
    "SpaceRobotDynamics",
    "compute_generalized_jacobian",
    "compute_base_reaction",
    "skew",
]
