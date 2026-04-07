"""Tests for N-DOF dynamics and momentum conservation."""
import numpy as np
import pytest
from space_robot_dq import (
    SpaceRobotKinematics, SpaceRobotDynamics,
    create_7dof_srs, create_3dof_planar, create_6dof_standard,
    RobotConfig, JointDef, LinkProperties,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def dyn7():
    kin = SpaceRobotKinematics(create_7dof_srs())
    return SpaceRobotDynamics(kinematics=kin)

@pytest.fixture
def dyn3():
    kin = SpaceRobotKinematics(create_3dof_planar())
    return SpaceRobotDynamics(kinematics=kin, base_mass=50.0)

@pytest.fixture
def dyn6():
    kin = SpaceRobotKinematics(create_6dof_standard())
    return SpaceRobotDynamics(kinematics=kin, base_mass=80.0)

@pytest.fixture
def dyn_custom():
    config = RobotConfig(
        joints=[
            JointDef([0,0,1], [0,0,0], "yaw"),
            JointDef([0,1,0], [0,0,0.3], "pitch"),
        ],
        ee_position=[0, 0, 0.6],
        name="2-DOF",
    )
    kin = SpaceRobotKinematics(config)
    return SpaceRobotDynamics(kinematics=kin, base_mass=30.0)


ALL_DYNAMICS = ["dyn7", "dyn3", "dyn6", "dyn_custom"]


# ============================================================================
# INERTIA MATRICES
# ============================================================================

class TestInertiaMatrices:
    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_hb_symmetric(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        q = np.zeros(dyn.n_joints)
        H_b, _ = dyn.compute_inertia_matrices(q)
        np.testing.assert_allclose(H_b, H_b.T, atol=1e-10)

    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_hb_positive_definite(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        q = np.zeros(dyn.n_joints)
        H_b, _ = dyn.compute_inertia_matrices(q)
        assert np.min(np.linalg.eigvalsh(H_b)) > 0

    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_hb_mass_block(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        H_b, _ = dyn.compute_inertia_matrices(np.zeros(dyn.n_joints))
        np.testing.assert_allclose(H_b[:3, :3], dyn.total_mass * np.eye(3), atol=1e-8)

    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_shapes(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        H_b, H_bm = dyn.compute_inertia_matrices(np.zeros(dyn.n_joints))
        assert H_b.shape == (6, 6)
        assert H_bm.shape == (6, dyn.n_joints)


# ============================================================================
# MOMENTUM CONSERVATION
# ============================================================================

class TestMomentumConservation:
    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_momentum_conserved(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        q = np.zeros(dyn.n_joints)
        qdot = np.random.randn(dyn.n_joints) * 0.1
        xb_dot = dyn.compute_base_velocity(q, qdot)
        h = dyn.compute_system_momentum(q, qdot, xb_dot)
        np.testing.assert_allclose(h, dyn.h0, atol=1e-10)

    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_base_reacts(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        qdot = np.ones(dyn.n_joints) * 0.1
        xb = dyn.compute_base_velocity(np.zeros(dyn.n_joints), qdot)
        assert np.linalg.norm(xb) > 1e-10

    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_zero_gives_zero(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        xb = dyn.compute_base_velocity(np.zeros(dyn.n_joints), np.zeros(dyn.n_joints))
        np.testing.assert_allclose(xb, np.zeros(6), atol=1e-12)

    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_nonzero_config_conserved(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        kin = dyn.kin
        q = np.clip(np.random.randn(dyn.n_joints) * 0.3, kin.q_min, kin.q_max)
        qdot = np.random.randn(dyn.n_joints) * 0.2
        xb = dyn.compute_base_velocity(q, qdot)
        h = dyn.compute_system_momentum(q, qdot, xb)
        np.testing.assert_allclose(h, dyn.h0, atol=1e-10)


# ============================================================================
# GENERALIZED JACOBIAN
# ============================================================================

class TestGeneralizedJacobian:
    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_jg_differs_from_jm(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        q = np.zeros(dyn.n_joints)
        J_g, J_m, _ = dyn.compute_generalized_jacobian(q)
        assert np.max(np.abs(J_g - J_m)) > 1e-6

    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_jg_prediction_consistent(self, dyn_name, request):
        """J_g · q̇ = J_m · q̇ + J_b · ẋ_b"""
        dyn = request.getfixturevalue(dyn_name)
        kin = dyn.kin
        q = np.clip(np.random.randn(dyn.n_joints) * 0.3, kin.q_min, kin.q_max)
        qdot = np.random.randn(dyn.n_joints) * 0.1
        J_g, J_m, J_b = dyn.compute_generalized_jacobian(q)
        xb = dyn.compute_base_velocity(q, qdot)
        np.testing.assert_allclose(J_g @ qdot, J_m @ qdot + J_b @ xb, atol=1e-8)

    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_jg_shape(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        J_g, _, _ = dyn.compute_generalized_jacobian(np.zeros(dyn.n_joints))
        assert J_g.shape == (6, dyn.n_joints)


# ============================================================================
# HEAVY BASE LIMIT
# ============================================================================

class TestHeavyBaseLimit:
    @pytest.mark.parametrize("dyn_name", ["dyn7", "dyn3", "dyn6"])
    def test_coupling_decreases(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        kin = dyn.kin
        q = np.clip(np.random.randn(dyn.n_joints) * 0.3, kin.q_min, kin.q_max)
        arm_mass = np.sum(dyn.link_masses)
        
        diffs = []
        for ratio in [1, 10, 100, 10000]:
            bm = arm_mass * ratio
            d = SpaceRobotDynamics(kinematics=kin, base_mass=bm,
                                    base_inertia=np.diag([bm*0.1]*3))
            Jg, Jm, _ = d.compute_generalized_jacobian(q)
            diffs.append(np.max(np.abs(Jg - Jm)))
        
        assert all(diffs[i] >= diffs[i+1] for i in range(len(diffs)-1))
        assert diffs[-1] < 1e-3


# ============================================================================
# LINK JACOBIANS
# ============================================================================

class TestLinkJacobians:
    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_link_jacobians_match_fd(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        kin = dyn.kin
        q = np.clip(np.random.randn(dyn.n_joints) * 0.3, kin.q_min, kin.q_max)
        J_links = dyn.compute_link_jacobians(q)
        coms_0, _ = dyn.compute_link_states(q)
        eps = 1e-7
        for i in range(dyn.n_joints):
            J_fd = np.zeros((3, dyn.n_joints))
            for j in range(dyn.n_joints):
                qp = q.copy(); qp[j] += eps
                coms_p, _ = dyn.compute_link_states(qp)
                J_fd[:, j] = (coms_p[i] - coms_0[i]) / eps
            np.testing.assert_allclose(J_links[i][3:6, :], J_fd, atol=1e-4)

    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_causal_structure(self, dyn_name, request):
        """Joint j > i should not affect link i."""
        dyn = request.getfixturevalue(dyn_name)
        kin = dyn.kin
        q = np.clip(np.random.randn(dyn.n_joints) * 0.3, kin.q_min, kin.q_max)
        J_links = dyn.compute_link_jacobians(q)
        for i in range(dyn.n_joints):
            for j in range(i + 1, dyn.n_joints):
                assert np.linalg.norm(J_links[i][:, j]) < 1e-12


# ============================================================================
# SYSTEM COM
# ============================================================================

class TestSystemCOM:
    @pytest.mark.parametrize("dyn_name", ALL_DYNAMICS)
    def test_com_finite(self, dyn_name, request):
        dyn = request.getfixturevalue(dyn_name)
        com = dyn.compute_system_com(np.zeros(dyn.n_joints))
        assert np.all(np.isfinite(com))


# ============================================================================
# CUSTOM LINK PROPERTIES
# ============================================================================

class TestCustomLinkProperties:
    def test_custom_masses(self):
        kin = SpaceRobotKinematics(create_3dof_planar())
        props = [
            LinkProperties(mass=5.0, com_home=[0, 0, 0.2], inertia=np.diag([0.1]*3)),
            LinkProperties(mass=3.0, com_home=[0, 0, 0.55], inertia=np.diag([0.05]*3)),
            LinkProperties(mass=2.0, com_home=[0, 0, 0.8], inertia=np.diag([0.02]*3)),
        ]
        dyn = SpaceRobotDynamics(kinematics=kin, base_mass=20.0, link_properties=props)
        assert dyn.total_mass == 30.0
        
        # Should still conserve momentum
        qdot = np.array([0.1, 0.2, -0.1])
        xb = dyn.compute_base_velocity(np.zeros(3), qdot)
        h = dyn.compute_system_momentum(np.zeros(3), qdot, xb)
        np.testing.assert_allclose(h, np.zeros(6), atol=1e-10)

    def test_legacy_api(self):
        """Test backward-compatible link_masses parameter."""
        kin = SpaceRobotKinematics(create_7dof_srs())
        dyn = SpaceRobotDynamics(
            kinematics=kin, base_mass=100.0,
            link_masses=[5, 3, 7, 3, 6, 2, 2])
        assert dyn.total_mass == 128.0
