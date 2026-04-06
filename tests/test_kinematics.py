"""Tests for kinematics module."""
import numpy as np
import pytest
from space_robot_dq import (
    SpaceRobotKinematics, forward_kinematics_simple, forward_kinematics_6dof,
    quaternion_distance, quaternion_to_euler, euler_to_quaternion,
)


@pytest.fixture
def kin():
    return SpaceRobotKinematics()


class TestForwardKinematics:
    def test_zero_config_position(self, kin):
        pos = kin.forward_kinematics(np.zeros(7))
        expected = np.array([0, 0, kin.d1 + kin.d3 + kin.d5 + kin.d7])
        np.testing.assert_allclose(pos, expected, atol=1e-10)

    def test_zero_config_includes_d7(self, kin):
        pos = kin.forward_kinematics(np.zeros(7))
        assert abs(pos[2] - 1.178) < 1e-10

    def test_fk_deterministic(self, kin):
        q = np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0])
        np.testing.assert_allclose(kin.forward_kinematics(q), kin.forward_kinematics(q))

    def test_fk_6dof_agrees_with_position(self, kin):
        q = np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0])
        pos = kin.forward_kinematics(q)
        pos6, _ = kin.forward_kinematics_6dof(q)
        np.testing.assert_allclose(pos, pos6, atol=1e-10)

    def test_quaternion_is_unit(self, kin):
        q = np.array([0.5, 0.1, 0.5, 0.8, -0.5, 0.2, 0.3])
        q = np.clip(q, kin.q_min, kin.q_max)
        _, quat = kin.forward_kinematics_6dof(q)
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-6

    def test_standalone_matches_class(self, kin):
        q = np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0])
        np.testing.assert_allclose(forward_kinematics_simple(q), kin.forward_kinematics(q), atol=1e-10)


class TestInverseKinematics:
    @pytest.mark.parametrize("q_orig", [
        np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0]),
        np.array([0.5, 0.1, 0.1, 0.8, -0.1, 0.2, 0.0]),
        np.zeros(7),
    ])
    def test_fk_ik_roundtrip(self, kin, q_orig):
        q_orig = np.clip(q_orig, kin.q_min, kin.q_max)
        target = kin.forward_kinematics(q_orig)
        q_sol = kin.inverse_kinematics(target, initial_guess=q_orig)
        achieved = kin.forward_kinematics(q_sol)
        assert np.linalg.norm(achieved - target) < 0.01  # 10mm

    @pytest.mark.parametrize("q_orig", [
        np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0]),
        np.array([0.5, 0.1, 0.1, 0.8, -0.1, 0.2, 0.0]),
    ])
    def test_6dof_roundtrip(self, kin, q_orig):
        q_orig = np.clip(q_orig, kin.q_min, kin.q_max)
        tgt_pos, tgt_quat = kin.forward_kinematics_6dof(q_orig)
        q_sol = kin.inverse_kinematics_6dof(tgt_pos, tgt_quat, initial_guess=q_orig)
        ach_pos, ach_quat = kin.forward_kinematics_6dof(q_sol)
        assert np.linalg.norm(ach_pos - tgt_pos) < 0.01
        assert quaternion_distance(ach_quat, tgt_quat) < np.deg2rad(5)


class TestJacobian:
    def test_numerical_jacobian_matches_fd(self, kin):
        q = np.clip(np.array([0.1, 0.2, 0.05, 1.0, 0.1, -0.2, 0.05]), kin.q_min, kin.q_max)
        J = kin.calculate_jacobian(q)
        eps = 1e-6
        p0 = kin.forward_kinematics(q)
        J_fd = np.zeros((3, 7))
        for i in range(7):
            qp = q.copy(); qp[i] += eps
            J_fd[:, i] = (kin.forward_kinematics(qp) - p0) / eps
        np.testing.assert_allclose(J[3:6, :], J_fd, atol=1e-4)

    def test_analytical_jacobian_matches_fd(self, kin):
        q = np.clip(np.array([0.1, 0.2, 0.05, 1.0, 0.1, -0.2, 0.05]), kin.q_min, kin.q_max)
        J = kin.calculate_jacobian_analytical(q)
        eps = 1e-6
        p0 = kin.forward_kinematics(q)
        J_fd = np.zeros((3, 7))
        for i in range(7):
            qp = q.copy(); qp[i] += eps
            J_fd[:, i] = (kin.forward_kinematics(qp) - p0) / eps
        np.testing.assert_allclose(J[3:6, :], J_fd, atol=1e-4)

    def test_jacobian_rank(self, kin):
        q = np.clip(np.array([0.2, 0.3, 0.1, 1.0, -0.1, 0.2, 0.05]), kin.q_min, kin.q_max)
        rank = np.linalg.matrix_rank(kin.calculate_jacobian(q), tol=1e-6)
        assert rank == 6


class TestQuaternionUtils:
    def test_identity_distance_zero(self):
        q = np.array([1, 0, 0, 0.0])
        assert quaternion_distance(q, q) < 1e-10

    def test_double_cover(self):
        q = np.array([1, 0, 0, 0.0])
        assert quaternion_distance(q, -q) < 1e-10

    def test_90deg_distance(self):
        q1 = np.array([1, 0, 0, 0.0])
        q2 = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
        assert abs(quaternion_distance(q1, q2) - np.pi/2) < 1e-6

    def test_euler_roundtrip(self):
        euler_in = np.array([0.3, -0.5, 0.7])
        euler_out = quaternion_to_euler(euler_to_quaternion(euler_in))
        np.testing.assert_allclose(euler_in, euler_out, atol=1e-10)


class TestScrewAxes:
    def test_axes_are_unit(self, kin):
        for i, s in enumerate(kin.screw_axes):
            assert abs(np.linalg.norm(s['l']) - 1.0) < 1e-10

    def test_adjacent_not_parallel(self, kin):
        for i in range(6):
            dot = abs(np.dot(kin.screw_axes[i]['l'], kin.screw_axes[i+1]['l']))
            assert dot < 0.99

    def test_workspace_spread(self, kin):
        np.random.seed(42)
        positions = np.array([kin.forward_kinematics(np.random.uniform(kin.q_min, kin.q_max))
                              for _ in range(500)])
        assert np.all(np.std(positions, axis=0) > 0.01)
