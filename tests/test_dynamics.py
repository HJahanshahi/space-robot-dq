"""Tests for dynamics and momentum conservation."""
import numpy as np
import pytest
from space_robot_dq import SpaceRobotKinematics, SpaceRobotDynamics


@pytest.fixture
def kin():
    return SpaceRobotKinematics()


@pytest.fixture
def dyn(kin):
    return SpaceRobotDynamics(kinematics=kin)


class TestInertiaMatrices:
    @pytest.mark.parametrize("q", [
        np.zeros(7),
        np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0]),
        np.array([0.5, 0.1, 0.5, 0.8, -0.5, 0.2, 0.3]),
    ])
    def test_hb_symmetric(self, dyn, kin, q):
        q = np.clip(q, kin.q_min, kin.q_max)
        H_b, _ = dyn.compute_inertia_matrices(q)
        np.testing.assert_allclose(H_b, H_b.T, atol=1e-10)

    @pytest.mark.parametrize("q", [
        np.zeros(7),
        np.array([0.5, 0.1, 0.5, 0.8, -0.5, 0.2, 0.3]),
    ])
    def test_hb_positive_definite(self, dyn, kin, q):
        q = np.clip(q, kin.q_min, kin.q_max)
        H_b, _ = dyn.compute_inertia_matrices(q)
        assert np.min(np.linalg.eigvalsh(H_b)) > 0

    def test_hb_mass_block(self, dyn):
        H_b, _ = dyn.compute_inertia_matrices(np.zeros(7))
        np.testing.assert_allclose(H_b[:3, :3], dyn.total_mass * np.eye(3), atol=1e-8)


class TestMomentumConservation:
    @pytest.mark.parametrize("q,qdot", [
        (np.zeros(7), np.array([0.1, 0, 0, 0, 0, 0, 0])),
        (np.zeros(7), np.array([0, 0.2, 0, 0.3, 0, -0.1, 0])),
        (np.array([0.3, 0.2, 0.1, 1.0, -0.2, 0.1, 0.05]), np.ones(7) * 0.1),
    ])
    def test_momentum_conserved(self, dyn, kin, q, qdot):
        q = np.clip(q, kin.q_min, kin.q_max)
        xb_dot = dyn.compute_base_velocity(q, qdot)
        h = dyn.compute_system_momentum(q, qdot, xb_dot)
        np.testing.assert_allclose(h, dyn.h0, atol=1e-10)

    def test_base_reacts_to_motion(self, dyn):
        xb = dyn.compute_base_velocity(np.zeros(7), np.array([0.1, 0, 0, 0, 0, 0, 0]))
        assert np.linalg.norm(xb) > 1e-10

    def test_zero_motion_zero_reaction(self, dyn):
        xb = dyn.compute_base_velocity(np.zeros(7), np.zeros(7))
        np.testing.assert_allclose(xb, np.zeros(6), atol=1e-12)


class TestGeneralizedJacobian:
    def test_jg_differs_from_jm(self, dyn, kin):
        q = np.clip(np.array([0.2, 0.3, 0.1, 1.0, -0.1, 0.2, 0.05]), kin.q_min, kin.q_max)
        J_g, J_m, _ = dyn.compute_generalized_jacobian(q)
        assert np.max(np.abs(J_g - J_m)) > 1e-6

    def test_jg_prediction_consistent(self, dyn, kin):
        """J_g * qdot should equal J_m * qdot + J_b * xb_dot"""
        q = np.clip(np.array([0.2, 0.3, 0.1, 1.0, -0.1, 0.2, 0.05]), kin.q_min, kin.q_max)
        qdot = np.array([0.1, 0.05, 0.02, -0.1, 0.03, -0.05, 0.01])
        J_g, J_m, J_b = dyn.compute_generalized_jacobian(q)
        xb_dot = dyn.compute_base_velocity(q, qdot)
        np.testing.assert_allclose(J_g @ qdot, J_m @ qdot + J_b @ xb_dot, atol=1e-8)

    def test_jg_rank(self, dyn, kin):
        q = np.clip(np.array([0.2, 0.3, 0.1, 1.0, -0.1, 0.2, 0.05]), kin.q_min, kin.q_max)
        J_g, _, _ = dyn.compute_generalized_jacobian(q)
        assert np.linalg.matrix_rank(J_g, tol=1e-6) == 6


class TestHeavyBaseLimit:
    def test_coupling_decreases_with_mass(self, kin):
        q = np.clip(np.array([0.2, 0.3, 0.1, 1.0, -0.1, 0.2, 0.05]), kin.q_min, kin.q_max)
        diffs = []
        for base_mass in [28.0, 280.0, 2800.0, 280000.0]:
            dyn = SpaceRobotDynamics(kinematics=kin, base_mass=base_mass,
                                      base_inertia=np.diag([base_mass * 0.1] * 3))
            J_g, J_m, _ = dyn.compute_generalized_jacobian(q)
            diffs.append(np.max(np.abs(J_g - J_m)))
        assert all(diffs[i] >= diffs[i+1] for i in range(len(diffs)-1))
        assert diffs[-1] < 1e-3

    def test_light_base_reacts_more(self, kin):
        q = np.clip(np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0]), kin.q_min, kin.q_max)
        qdot = np.array([0.0, 0.1, 0.0, 0.2, 0.0, -0.1, 0.0])
        dyn_heavy = SpaceRobotDynamics(kinematics=kin, base_mass=1000.0, base_inertia=np.diag([100.]*3))
        dyn_light = SpaceRobotDynamics(kinematics=kin, base_mass=30.0, base_inertia=np.diag([3.]*3))
        assert np.linalg.norm(dyn_light.compute_base_velocity(q, qdot)) > \
               np.linalg.norm(dyn_heavy.compute_base_velocity(q, qdot))


class TestLinkJacobians:
    def test_link_jacobians_match_fd(self, dyn, kin):
        q = np.clip(np.array([0.2, 0.3, 0.1, 1.0, -0.1, 0.2, 0.05]), kin.q_min, kin.q_max)
        J_links = dyn.compute_link_jacobians(q)
        coms_base, _ = dyn.compute_link_states(q)
        eps = 1e-7
        for i in range(7):
            J_fd = np.zeros((3, 7))
            for j in range(7):
                qp = q.copy(); qp[j] += eps
                coms_p, _ = dyn.compute_link_states(qp)
                J_fd[:, j] = (coms_p[i] - coms_base[i]) / eps
            np.testing.assert_allclose(J_links[i][3:6, :], J_fd, atol=1e-4)

    def test_causal_structure(self, dyn, kin):
        """Joint j > i should not affect link i."""
        q = np.clip(np.array([0.2, 0.3, 0.1, 1.0, -0.1, 0.2, 0.05]), kin.q_min, kin.q_max)
        J_links = dyn.compute_link_jacobians(q)
        for i in range(7):
            for j in range(i + 1, 7):
                assert np.linalg.norm(J_links[i][:, j]) < 1e-12


class TestSystemCOM:
    def test_zero_config_on_z_axis(self, dyn):
        com = dyn.compute_system_com(np.zeros(7))
        assert abs(com[0]) < 1e-10
        assert abs(com[1]) < 1e-10
        assert com[2] > 0

    def test_com_below_ee(self, dyn, kin):
        com = dyn.compute_system_com(np.zeros(7))
        assert com[2] < kin.home_position[2]
