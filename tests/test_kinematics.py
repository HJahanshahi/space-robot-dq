"""Tests for generalized N-DOF kinematics."""
import numpy as np
import pytest
from space_robot_dq import (
    SpaceRobotKinematics, RobotConfig, JointDef,
    create_7dof_srs, create_3dof_planar, create_6dof_standard,
    quaternion_distance, quaternion_to_euler, euler_to_quaternion,
    forward_kinematics_simple,
)


# ============================================================================
# FIXTURES: multiple robot configurations
# ============================================================================

@pytest.fixture
def kin7():
    """Default 7-DOF SRS."""
    return SpaceRobotKinematics(create_7dof_srs())

@pytest.fixture
def kin3():
    """3-DOF planar."""
    return SpaceRobotKinematics(create_3dof_planar())

@pytest.fixture
def kin6():
    """6-DOF standard."""
    return SpaceRobotKinematics(create_6dof_standard())

@pytest.fixture
def kin_custom():
    """Custom 4-DOF robot."""
    config = RobotConfig(
        joints=[
            JointDef([0,0,1], [0,0,0],   "yaw",   -np.pi, np.pi),
            JointDef([0,1,0], [0,0,0.25], "pitch1", -np.pi/2, np.pi/2),
            JointDef([0,1,0], [0,0,0.55], "pitch2", -np.pi/2, np.pi/2),
            JointDef([1,0,0], [0,0,0.85], "roll",   -np.pi, np.pi),
        ],
        ee_position=[0, 0, 1.0],
        name="Custom 4-DOF",
    )
    return SpaceRobotKinematics(config)


ALL_ROBOTS = ["kin7", "kin3", "kin6", "kin_custom"]


# ============================================================================
# FK TESTS (all robots)
# ============================================================================

class TestForwardKinematics:
    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_zero_config_matches_ee_home(self, robot, request):
        kin = request.getfixturevalue(robot)
        pos = kin.forward_kinematics(np.zeros(kin.num_joints))
        np.testing.assert_allclose(pos, kin.home_position, atol=1e-10)

    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_fk_is_deterministic(self, robot, request):
        kin = request.getfixturevalue(robot)
        q = np.random.uniform(kin.q_min, kin.q_max)
        np.testing.assert_allclose(kin.forward_kinematics(q), kin.forward_kinematics(q))

    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_fk_6dof_agrees_with_position(self, robot, request):
        kin = request.getfixturevalue(robot)
        q = np.random.uniform(kin.q_min, kin.q_max)
        pos = kin.forward_kinematics(q)
        pos6, _ = kin.forward_kinematics_6dof(q)
        np.testing.assert_allclose(pos, pos6, atol=1e-10)

    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_quaternion_is_unit(self, robot, request):
        kin = request.getfixturevalue(robot)
        q = np.random.uniform(kin.q_min, kin.q_max)
        _, quat = kin.forward_kinematics_6dof(q)
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-6

    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_fk_finite_at_limits(self, robot, request):
        kin = request.getfixturevalue(robot)
        assert np.all(np.isfinite(kin.forward_kinematics(kin.q_min)))
        assert np.all(np.isfinite(kin.forward_kinematics(kin.q_max)))

    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_non_zero_joints_change_position(self, robot, request):
        kin = request.getfixturevalue(robot)
        q_test = kin.q_min * 0.3 + kin.q_max * 0.7  # Asymmetric to avoid zero
        pos_zero = kin.forward_kinematics(np.zeros(kin.num_joints))
        pos_test = kin.forward_kinematics(q_test)
        # At least some change
        assert np.linalg.norm(pos_test - pos_zero) > 1e-6


class TestForwardKinematics7DOF:
    """7-DOF SRS specific tests."""
    def test_zero_config_on_z_axis(self, kin7):
        pos = kin7.forward_kinematics(np.zeros(7))
        assert abs(pos[0]) < 1e-10
        assert abs(pos[1]) < 1e-10
        assert abs(pos[2] - 1.178) < 1e-10

    def test_standalone_matches_class(self, kin7):
        q = np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0])
        np.testing.assert_allclose(forward_kinematics_simple(q), kin7.forward_kinematics(q), atol=1e-10)


class TestForwardKinematics3DOF:
    """3-DOF planar specific tests."""
    def test_correct_num_joints(self, kin3):
        assert kin3.num_joints == 3

    def test_zero_config_position(self, kin3):
        pos = kin3.forward_kinematics(np.zeros(3))
        np.testing.assert_allclose(pos, [0, 0, 0.9], atol=1e-10)


# ============================================================================
# IK TESTS
# ============================================================================

class TestInverseKinematics:
    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_fk_ik_roundtrip_warm(self, robot, request):
        kin = request.getfixturevalue(robot)
        q_orig = np.clip(
            (kin.q_min + kin.q_max) / 2.0 + np.random.randn(kin.num_joints) * 0.1,
            kin.q_min, kin.q_max)
        target = kin.forward_kinematics(q_orig)
        q_sol = kin.inverse_kinematics(target, initial_guess=q_orig)
        achieved = kin.forward_kinematics(q_sol)
        assert np.linalg.norm(achieved - target) < 0.01  # 10mm

    def test_6dof_roundtrip_7dof(self, kin7):
        q_orig = np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0])
        tgt_pos, tgt_quat = kin7.forward_kinematics_6dof(q_orig)
        q_sol = kin7.inverse_kinematics_6dof(tgt_pos, tgt_quat, initial_guess=q_orig)
        ach_pos, ach_quat = kin7.forward_kinematics_6dof(q_sol)
        assert np.linalg.norm(ach_pos - tgt_pos) < 0.01
        assert quaternion_distance(ach_quat, tgt_quat) < np.deg2rad(5)


# ============================================================================
# JACOBIAN TESTS
# ============================================================================

class TestJacobian:
    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_numerical_jacobian_matches_fd(self, robot, request):
        kin = request.getfixturevalue(robot)
        q = np.clip(
            (kin.q_min + kin.q_max) / 2.0 + np.random.randn(kin.num_joints) * 0.1,
            kin.q_min, kin.q_max)
        J = kin.calculate_jacobian(q)
        eps = 1e-6
        p0 = kin.forward_kinematics(q)
        J_fd = np.zeros((3, kin.num_joints))
        for i in range(kin.num_joints):
            qp = q.copy(); qp[i] += eps
            J_fd[:, i] = (kin.forward_kinematics(qp) - p0) / eps
        np.testing.assert_allclose(J[3:6, :], J_fd, atol=1e-4)

    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_analytical_matches_numerical(self, robot, request):
        kin = request.getfixturevalue(robot)
        q = np.clip(
            (kin.q_min + kin.q_max) / 2.0 + np.random.randn(kin.num_joints) * 0.1,
            kin.q_min, kin.q_max)
        J_num = kin.calculate_jacobian(q)
        J_ana = kin.calculate_jacobian_analytical(q)
        np.testing.assert_allclose(J_num, J_ana, atol=1e-5)

    def test_jacobian_rank_7dof(self, kin7):
        q = np.array([0.2, 0.3, 0.1, 1.0, -0.1, 0.2, 0.05])
        assert np.linalg.matrix_rank(kin7.calculate_jacobian(q), tol=1e-6) == 6


# ============================================================================
# QUATERNION UTILITIES
# ============================================================================

class TestQuaternionUtils:
    def test_identity_distance_zero(self):
        q = np.array([1, 0, 0, 0.0])
        assert quaternion_distance(q, q) < 1e-10

    def test_double_cover(self):
        q = np.array([1, 0, 0, 0.0])
        assert quaternion_distance(q, -q) < 1e-10

    def test_90deg(self):
        q1 = np.array([1, 0, 0, 0.0])
        q2 = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
        assert abs(quaternion_distance(q1, q2) - np.pi/2) < 1e-6

    def test_euler_roundtrip(self):
        e_in = np.array([0.3, -0.5, 0.7])
        np.testing.assert_allclose(quaternion_to_euler(euler_to_quaternion(e_in)), e_in, atol=1e-10)


# ============================================================================
# SCREW AXES
# ============================================================================

class TestScrewAxes:
    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_axes_are_unit(self, robot, request):
        kin = request.getfixturevalue(robot)
        for s in kin.screw_axes:
            assert abs(np.linalg.norm(s['l']) - 1.0) < 1e-10

    @pytest.mark.parametrize("robot", ALL_ROBOTS)
    def test_workspace_spread(self, robot, request):
        kin = request.getfixturevalue(robot)
        np.random.seed(42)
        positions = np.array([
            kin.forward_kinematics(np.random.uniform(kin.q_min, kin.q_max))
            for _ in range(200)])
        # At least 2 axes should have spread
        assert np.sum(np.std(positions, axis=0) > 0.01) >= 2


# ============================================================================
# ROBOT CONFIG
# ============================================================================

class TestRobotConfig:
    def test_num_joints(self):
        cfg = create_7dof_srs()
        assert cfg.num_joints == 7

    def test_3dof_preset(self):
        cfg = create_3dof_planar()
        assert cfg.num_joints == 3

    def test_6dof_preset(self):
        cfg = create_6dof_standard()
        assert cfg.num_joints == 6

    def test_custom_config(self):
        config = RobotConfig(
            joints=[
                JointDef([0,0,1], [0,0,0], "j1"),
                JointDef([0,1,0], [0,0,0.5], "j2"),
            ],
            ee_position=[0, 0, 1.0],
        )
        kin = SpaceRobotKinematics(config)
        assert kin.num_joints == 2
        pos = kin.forward_kinematics(np.zeros(2))
        np.testing.assert_allclose(pos, [0, 0, 1.0], atol=1e-10)

    def test_joint_limits_from_config(self):
        config = RobotConfig(
            joints=[
                JointDef([0,0,1], [0,0,0], "j1", q_min=-1.0, q_max=1.0),
                JointDef([0,1,0], [0,0,0.5], "j2", q_min=-0.5, q_max=0.5),
            ],
            ee_position=[0, 0, 1.0],
        )
        kin = SpaceRobotKinematics(config)
        np.testing.assert_allclose(kin.q_min, [-1.0, -0.5])
        np.testing.assert_allclose(kin.q_max, [1.0, 0.5])
