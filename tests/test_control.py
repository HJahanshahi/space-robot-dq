"""
Tests for resolved-rate control and free-floating trajectory simulation.

Covers:
- Damped least-squares velocity mapping
- Closed-loop tracking: generalized Jacobian controller must strongly
  outperform the fixed-base (naive) controller on the free-floating plant
- Physics invariants during moving-base simulation:
  momentum conservation and stationarity of the system center of mass
- Heavy-base limit: both controllers converge
- Base attitude drift metrics
"""

import numpy as np
import pytest

from space_robot_dq import (
    SpaceRobotKinematics,
    SpaceRobotDynamics,
    resolved_rate_qdot,
    simulate_resolved_rate_tracking,
    simulate_free_floating_trajectory,
    rotation_angle_deg,
    quintic_line,
)


# --------------------------------------------------------------------
# Shared scenario: 7-DOF SRS, dexterous-workspace line move
# --------------------------------------------------------------------

Q0 = np.array([0.3, 0.4, -0.2, 0.8, 0.1, 0.5, 0.0])
DISPLACEMENT = np.array([-0.20, 0.10, -0.15])
DURATION = 4.0
DT = 0.01


def make_system(base_mass=100.0):
    kin = SpaceRobotKinematics()
    dyn = SpaceRobotDynamics(kinematics=kin, base_mass=base_mass)
    return kin, dyn


_CACHE = {}


def run_tracking(dyn, kin, jacobian, kp=1.0, dt=DT, base_mass=100.0):
    key = (jacobian, kp, dt, base_mass)
    if key not in _CACHE:
        p0 = kin.forward_kinematics(Q0)
        p_des, v_des = quintic_line(p0, p0 + DISPLACEMENT, DURATION)
        _CACHE[key] = simulate_resolved_rate_tracking(
            dyn, Q0, p_des, v_des, DURATION, dt=dt,
            jacobian=jacobian, kp=kp)
    return _CACHE[key]


# --------------------------------------------------------------------
# resolved_rate_qdot
# --------------------------------------------------------------------

class TestResolvedRateQdot:

    def test_matches_pinv_when_well_conditioned(self):
        rng = np.random.default_rng(0)
        J = rng.standard_normal((3, 7))
        v = rng.standard_normal(3)
        qdot = resolved_rate_qdot(J, v, damping=1e-12)
        qdot_pinv = np.linalg.pinv(J) @ v
        assert np.allclose(qdot, qdot_pinv, atol=1e-8)

    def test_achieves_task_velocity(self):
        rng = np.random.default_rng(1)
        J = rng.standard_normal((3, 7))
        v = rng.standard_normal(3)
        qdot = resolved_rate_qdot(J, v, damping=1e-12)
        assert np.allclose(J @ qdot, v, atol=1e-8)

    def test_bounded_near_singularity(self):
        J = np.zeros((3, 7))
        J[0, 0] = 1e-9  # nearly rank-deficient
        v = np.array([1.0, 1.0, 1.0])
        qdot = resolved_rate_qdot(J, v, damping=1e-6)
        assert np.all(np.isfinite(qdot))
        assert np.linalg.norm(qdot) < 1e4


# --------------------------------------------------------------------
# Closed-loop tracking: the central claim
# --------------------------------------------------------------------

class TestTrackingPerformance:

    def test_generalized_controller_tracks_submillimeter(self):
        kin, dyn = make_system()
        h = run_tracking(dyn, kin, "generalized")
        rms = np.sqrt(np.mean(h["err"] ** 2))
        assert rms < 1e-3, f"J_g controller RMS {rms*1000:.2f} mm >= 1 mm"
        assert h["err"][-1] < 1e-3

    def test_fixed_base_controller_has_large_error(self):
        kin, dyn = make_system()
        h = run_tracking(dyn, kin, "fixed")
        rms = np.sqrt(np.mean(h["err"] ** 2))
        assert rms > 5e-3, "naive controller unexpectedly accurate"

    def test_generalized_beats_fixed_by_order_of_magnitude(self):
        kin, dyn = make_system()
        h_g = run_tracking(dyn, kin, "generalized")
        h_f = run_tracking(dyn, kin, "fixed")
        rms_g = np.sqrt(np.mean(h_g["err"] ** 2))
        rms_f = np.sqrt(np.mean(h_f["err"] ** 2))
        assert rms_f / rms_g > 10.0

    def test_feedback_reduces_error_for_both(self):
        kin, dyn = make_system()
        for jac in ("generalized", "fixed"):
            rms_ff = np.sqrt(np.mean(
                run_tracking(dyn, kin, jac, kp=0.0)["err"] ** 2))
            rms_fb = np.sqrt(np.mean(
                run_tracking(dyn, kin, jac, kp=1.0)["err"] ** 2))
            assert rms_fb < rms_ff

    def test_invalid_jacobian_raises(self):
        kin, dyn = make_system()
        p0 = kin.forward_kinematics(Q0)
        p_des, v_des = quintic_line(p0, p0 + DISPLACEMENT, 1.0)
        with pytest.raises(ValueError):
            simulate_resolved_rate_tracking(
                dyn, Q0, p_des, v_des, 1.0, jacobian="magic")


# --------------------------------------------------------------------
# Physics invariants during moving-base simulation
# --------------------------------------------------------------------

class TestPhysicsInvariants:

    def test_momentum_conserved_during_tracking(self):
        kin, dyn = make_system()
        for jac in ("generalized", "fixed"):
            h = run_tracking(dyn, kin, jac)
            assert h["momentum"].max() < 1e-12

    def test_system_com_stationary(self):
        # Zero total linear momentum implies the system center of mass
        # must not move. This is NOT enforced by construction, so it
        # independently validates the moving-base propagation.
        kin, dyn = make_system()
        h = run_tracking(dyn, kin, "generalized")
        com_drift = np.max(np.linalg.norm(h["com"] - h["com"][0], axis=1))
        assert com_drift < 1e-4, f"COM drifted {com_drift:.2e} m"

    def test_com_drift_shrinks_with_dt(self):
        # COM drift should be integrator error: first-order in dt.
        kin, dyn = make_system()
        h1 = run_tracking(dyn, kin, "generalized", dt=0.02)
        h2 = run_tracking(dyn, kin, "generalized", dt=0.005)
        d1 = np.max(np.linalg.norm(h1["com"] - h1["com"][0], axis=1))
        d2 = np.max(np.linalg.norm(h2["com"] - h2["com"][0], axis=1))
        assert d2 < d1

    def test_base_pose_restored_after_simulation(self):
        kin, dyn = make_system()
        R_before, p_before = kin.base_dq.to_pose()
        run_tracking(dyn, kin, "generalized")
        R_after, p_after = kin.base_dq.to_pose()
        assert np.allclose(R_before, R_after)
        assert np.allclose(p_before, p_after)


# --------------------------------------------------------------------
# Heavy-base limit
# --------------------------------------------------------------------

class TestHeavyBaseLimit:

    def test_controllers_converge_for_heavy_base(self):
        kin, dyn = make_system(base_mass=1e6)
        h_g = run_tracking(dyn, kin, "generalized", base_mass=1e6)
        h_f = run_tracking(dyn, kin, "fixed", base_mass=1e6)
        rms_g = np.sqrt(np.mean(h_g["err"] ** 2))
        rms_f = np.sqrt(np.mean(h_f["err"] ** 2))
        assert abs(rms_f - rms_g) < 1e-4

    def test_negligible_drift_for_heavy_base(self):
        kin, dyn = make_system(base_mass=1e6)
        h = run_tracking(dyn, kin, "generalized", base_mass=1e6)
        assert h["base_att_deg"][-1] < 0.05


# --------------------------------------------------------------------
# Open-loop trajectory / attitude drift
# --------------------------------------------------------------------

class TestFreeFloatingTrajectory:

    def test_attitude_drift_positive_and_momentum_conserved(self):
        kin, dyn = make_system()
        q0 = np.array([0.3, 0.5, -0.3, 1.0, 0.2, 0.6, 0.1])
        A = np.array([0.4, 0.3, 0.5, 0.4, 0.3, 0.2, 0.3])
        w = np.array([1.0, 1.5, 0.8, 1.2, 2.0, 1.0, 1.5]) * np.pi / 2

        h = simulate_free_floating_trajectory(
            dyn,
            q_fn=lambda t: q0 + A * np.sin(w * t),
            qdot_fn=lambda t: A * w * np.cos(w * t),
            duration=2.0, dt=0.01)

        assert h["base_att_deg"].max() > 0.1
        assert h["momentum"].max() < 1e-12
        com_drift = np.max(np.linalg.norm(h["com"] - h["com"][0], axis=1))
        assert com_drift < 1e-3

    def test_free_speed_below_fixed_speed_on_average(self):
        kin, dyn = make_system()
        q0 = np.array([0.3, 0.5, -0.3, 1.0, 0.2, 0.6, 0.1])
        A = 0.4 * np.ones(7)
        w = np.pi * np.ones(7)
        h = simulate_free_floating_trajectory(
            dyn,
            q_fn=lambda t: q0 + A * np.sin(w * t),
            qdot_fn=lambda t: A * w * np.cos(w * t),
            duration=2.0, dt=0.01)
        assert np.mean(h["ee_speed_free"]) < np.mean(h["ee_speed_fixed"])


# --------------------------------------------------------------------
# rotation_angle_deg
# --------------------------------------------------------------------

class TestRotationAngle:

    def test_identity_is_zero(self):
        assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0)

    def test_known_angle(self):
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        assert rotation_angle_deg(R) == pytest.approx(30.0, abs=1e-9)


# --------------------------------------------------------------------
# 6-DOF pose tracking of a tumbling target
# --------------------------------------------------------------------

from space_robot_dq import simulate_pose_tracking, tumbling_target


class TestPoseTracking:

    TUMBLE_AXIS = np.array([0.3, 1.0, 0.5]) / np.linalg.norm([0.3, 1.0, 0.5])

    def _system_and_target(self, rate_dps=5.0):
        kin, dyn = make_system()
        R0, p0 = kin.forward_kinematics_dq(Q0).to_pose()
        pose_fn, twist_fn = tumbling_target(
            R0, p0, np.radians(rate_dps) * self.TUMBLE_AXIS, np.zeros(3))
        return kin, dyn, pose_fn, twist_fn

    def test_static_target_converges(self):
        kin, dyn = make_system()
        from scipy.spatial.transform import Rotation
        R0, p0 = kin.forward_kinematics_dq(Q0).to_pose()
        R_t = Rotation.from_rotvec(np.radians(10) * np.array([0, 0, 1])
                                   ).as_matrix() @ R0
        pose_fn = lambda t: (R_t, p0 + np.array([0.05, 0.0, 0.0]))
        twist_fn = lambda t: (np.zeros(3), np.zeros(3))
        h = simulate_pose_tracking(dyn, Q0, pose_fn, twist_fn, 8.0,
                                   dt=0.01, jacobian="generalized")
        assert h["pos_err"][-1] < 1e-4      # < 0.1 mm
        assert h["ori_err_deg"][-1] < 0.01  # < 0.01 deg

    def test_generalized_tracks_tumbling_target(self):
        kin, dyn, pose_fn, twist_fn = self._system_and_target()
        h = simulate_pose_tracking(dyn, Q0, pose_fn, twist_fn, 10.0,
                                   dt=0.01, jacobian="generalized")
        rms_pos = np.sqrt(np.mean(h["pos_err"] ** 2))
        rms_ori = np.sqrt(np.mean(h["ori_err_deg"] ** 2))
        assert rms_pos < 1e-4   # < 0.1 mm RMS
        assert rms_ori < 0.05   # deg

    def test_generalized_beats_fixed_on_pose(self):
        kin, dyn, pose_fn, twist_fn = self._system_and_target()
        h_g = simulate_pose_tracking(dyn, Q0, pose_fn, twist_fn, 10.0,
                                     dt=0.01, jacobian="generalized")
        h_f = simulate_pose_tracking(dyn, Q0, pose_fn, twist_fn, 10.0,
                                     dt=0.01, jacobian="fixed")
        assert (np.sqrt(np.mean(h_f["pos_err"] ** 2))
                / np.sqrt(np.mean(h_g["pos_err"] ** 2))) > 50.0
        assert (np.sqrt(np.mean(h_f["ori_err_deg"] ** 2))
                / np.sqrt(np.mean(h_g["ori_err_deg"] ** 2))) > 50.0

    def test_momentum_and_com_during_pose_tracking(self):
        kin, dyn, pose_fn, twist_fn = self._system_and_target()
        h = simulate_pose_tracking(dyn, Q0, pose_fn, twist_fn, 10.0,
                                   dt=0.01, jacobian="generalized")
        assert h["momentum"].max() < 1e-12
        com_drift = np.max(np.linalg.norm(h["com"] - h["com"][0], axis=1))
        assert com_drift < 1e-3

    def test_base_pose_restored(self):
        kin, dyn, pose_fn, twist_fn = self._system_and_target()
        R_b, p_b = kin.base_dq.to_pose()
        simulate_pose_tracking(dyn, Q0, pose_fn, twist_fn, 2.0,
                               dt=0.01, jacobian="fixed")
        R_a, p_a = kin.base_dq.to_pose()
        assert np.allclose(R_b, R_a) and np.allclose(p_b, p_a)
