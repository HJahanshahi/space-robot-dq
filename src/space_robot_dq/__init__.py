"""
space_robot_dq — Dual quaternion kinematics and dynamics for free-floating space robots.

Supports arbitrary N-DOF serial chain manipulators on a free-floating base.
Includes built-in presets for common configurations.

Quick start::

    from space_robot_dq import SpaceRobotKinematics, SpaceRobotDynamics

    # Default 7-DOF SRS
    kin = SpaceRobotKinematics()
    
    # Custom 4-DOF robot
    from space_robot_dq import RobotConfig, JointDef
    config = RobotConfig(
        joints=[
            JointDef(axis=[0,0,1], position=[0,0,0], name="yaw"),
            JointDef(axis=[0,1,0], position=[0,0,0.3], name="pitch1"),
            JointDef(axis=[0,1,0], position=[0,0,0.6], name="pitch2"),
            JointDef(axis=[1,0,0], position=[0,0,0.9], name="roll"),
        ],
        ee_position=[0, 0, 1.0],
        name="Custom 4-DOF",
    )
    kin = SpaceRobotKinematics(config)
    dyn = SpaceRobotDynamics(kinematics=kin, base_mass=50.0)
"""

__version__ = "0.2.0"

# Dual quaternion
from .dual_quaternion import (
    DualQuaternion,
    quaternion_multiply,
    quaternion_multiply_torch,
    quaternion_to_rotation_matrix,
    log_dq,
)

# Robot configuration
from .kinematics import (
    JointDef,
    RobotConfig,
    # Presets
    create_7dof_srs,
    create_3dof_planar,
    create_6dof_standard,
    # Kinematics class
    SpaceRobotKinematics,
    # Quaternion utilities
    quaternion_distance,
    quaternion_to_euler,
    euler_to_quaternion,
    # Standalone functions (default 7-DOF SRS)
    compute_forward_kinematics,
    forward_kinematics_simple,
    forward_kinematics_6dof,
    inverse_kinematics_6dof,
    inverse_kinematics_numerical_scipy,
    find_achievable_orientation,
    find_achievable_orientation_at_position,
    find_best_achievable_orientation,
)

# Dynamics
from .dynamics import (
    LinkProperties,
    default_link_properties,
    SpaceRobotDynamics,
    compute_generalized_jacobian,
    compute_base_reaction,
    skew,
)

__all__ = [
    "__version__",
    # Dual quaternion
    "DualQuaternion", "quaternion_multiply", "quaternion_multiply_torch",
    "quaternion_to_rotation_matrix", "log_dq",
    # Configuration
    "JointDef", "RobotConfig", "LinkProperties", "default_link_properties",
    # Presets
    "create_7dof_srs", "create_3dof_planar", "create_6dof_standard",
    # Kinematics
    "SpaceRobotKinematics",
    "quaternion_distance", "quaternion_to_euler", "euler_to_quaternion",
    "compute_forward_kinematics", "forward_kinematics_simple",
    "forward_kinematics_6dof", "inverse_kinematics_6dof",
    "inverse_kinematics_numerical_scipy",
    "find_achievable_orientation", "find_achievable_orientation_at_position",
    "find_best_achievable_orientation",
    # Dynamics
    "SpaceRobotDynamics", "compute_generalized_jacobian", "compute_base_reaction", "skew",
]
