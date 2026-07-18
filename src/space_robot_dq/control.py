"""
Resolved-rate control and free-floating trajectory simulation.

Implements closed-loop Cartesian resolved-rate control for free-floating
space manipulators, with the spacecraft base propagated from momentum
conservation at every step. Supports both the fixed-base Jacobian J_m
(a "naive" controller that ignores base reaction) and the generalized
Jacobian J_g of Umetani & Yoshida (1989), enabling quantitative
comparison of tracking performance and spacecraft attitude drift.

Key functions:
    resolved_rate_qdot          damped least-squares velocity mapping
    simulate_resolved_rate_tracking
                                closed-loop Cartesian tracking on the
                                free-floating truth model
    simulate_free_floating_trajectory
                                open-loop joint trajectory with base
                                pose propagation and drift metrics
    rotation_angle_deg          rotation angle of a rotation matrix

References:
    - Umetani & Yoshida (1989), IEEE Trans. Robotics and Automation 5(3)
    - Dubowsky & Papadopoulos (1993), IEEE Trans. Robotics and Automation 9(5)

Author: Hadi Jahanshahi, Zheng H. Zhu
Affiliation: Department of Mechanical Engineering, York University
Date: 2025
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .dual_quaternion import DualQuaternion


# ====================================================================
# UTILITIES
# ====================================================================

def rotation_angle_deg(R):
    """
    Rotation angle (degrees) of a 3x3 rotation matrix.

    angle = arccos((trace(R) - 1) / 2)
    """
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def propagate_base_pose(kinematics, xb_dot, dt):
    """
    Integrate the base pose by one time step.

    The base twist xb_dot = [v_b; w_b] (world frame, as returned by
    SpaceRobotDynamics.compute_base_velocity) is applied with a
    first-order exponential update:

        p_b <- p_b + v_b * dt
        R_b <- exp(skew(w_b) * dt) @ R_b

    The kinematics object's base_dq is updated in place, so all
    subsequent FK / Jacobian / inertia evaluations see the new pose.
    """
    xb_dot = np.asarray(xb_dot, dtype=float).flatten()
    v_b, w_b = xb_dot[:3], xb_dot[3:6]
    R_b, p_b = kinematics.base_dq.to_pose()
    p_new = p_b + v_b * dt
    R_new = Rotation.from_rotvec(w_b * dt).as_matrix() @ R_b
    kinematics.base_dq = DualQuaternion.from_pose(R_new, p_new)


def resolved_rate_qdot(J, v_des, damping=1e-8):
    """
    Damped least-squares (Levenberg-Marquardt) inverse velocity mapping.

        qdot = J^T (J J^T + damping * I)^-1 v_des

    Args:
        J: (m, N) task Jacobian
        v_des: (m,) desired task velocity
        damping: Tikhonov damping factor (singularity robustness)

    Returns:
        (N,) joint velocity vector
    """
    J = np.atleast_2d(np.asarray(J, dtype=float))
    v_des = np.asarray(v_des, dtype=float).flatten()
    m = J.shape[0]
    JJt = J @ J.T + damping * np.eye(m)
    return J.T @ np.linalg.solve(JJt, v_des)


def quintic_line(p0, p1, T):
    """
    Straight-line Cartesian path with quintic (rest-to-rest) time scaling.

    Returns (p_des(t), v_des(t)) callables for t in [0, T].
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0

    def s(t):
        tau = np.clip(t / T, 0.0, 1.0)
        return 10 * tau**3 - 15 * tau**4 + 6 * tau**5

    def sdot(t):
        tau = np.clip(t / T, 0.0, 1.0)
        return (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T

    return (lambda t: p0 + s(t) * d,
            lambda t: sdot(t) * d)


# ====================================================================
# CLOSED-LOOP RESOLVED-RATE TRACKING
# ====================================================================

def simulate_resolved_rate_tracking(dynamics, q0, p_des_fn, v_des_fn,
                                    duration, dt=0.01,
                                    jacobian="generalized",
                                    kp=1.0, damping=1e-6,
                                    k_null=0.5, q_ref=None,
                                    qdot_max=1.5,
                                    jacobian_method="analytical"):
    """
    Closed-loop Cartesian position tracking on the free-floating truth model.

    At each step the controller commands

        v_cmd = v_des(t) + kp * (p_des(t) - p_ee)
        qdot  = DLS(J_pos) @ v_cmd

    where J_pos is the position block (rows 3:6) of either the
    generalized Jacobian J_g (jacobian="generalized") or the fixed-base
    Jacobian J_m (jacobian="fixed"). Regardless of the controller's
    Jacobian choice, the *plant* is always the free-floating system:
    the base twist follows from momentum conservation and the base pose
    is propagated every step, so a controller using J_m experiences
    model mismatch exactly as a real spacecraft-mounted arm would.

    The kinematics base pose is restored to its initial value on exit.

    Args:
        dynamics: SpaceRobotDynamics instance (defines plant + h0)
        q0: (N,) initial joint configuration
        p_des_fn, v_des_fn: callables t -> (3,) desired EE position / velocity
        duration: simulated time (s)
        dt: integration step (s)
        jacobian: "generalized" or "fixed" (controller model)
        kp: proportional position feedback gain (1/s)
        damping: DLS damping factor
        k_null: null-space posture gain pulling joints toward q_ref
            (uses the redundancy of the position task to stay away from
            singularities); set 0.0 to disable
        q_ref: (N,) reference posture for the null-space term
            (default: q0)
        qdot_max: element-wise joint rate limit (rad/s); None to disable
        jacobian_method: "analytical" (default; ~10x faster) or
            "numerical" — passed to compute_generalized_jacobian

    Returns:
        dict with time histories:
            t           (K,)   time stamps
            q           (K, N) joint angles
            p_des       (K, 3) desired EE position
            p_act       (K, 3) actual EE position (free-floating plant)
            err         (K,)   Cartesian tracking error norm (m)
            base_att_deg(K,)   base attitude excursion from start (deg)
            base_pos    (K, 3) base position
            momentum    (K,)   ||h|| system momentum norm (conservation check)
            com         (K, 3) system center of mass (should stay fixed)
    """
    if jacobian not in ("generalized", "fixed"):
        raise ValueError("jacobian must be 'generalized' or 'fixed'")

    kin = dynamics.kin
    n = dynamics.n_joints
    q = np.array(q0, dtype=float).flatten()[:n].copy()
    q_ref_arr = (np.array(q_ref, dtype=float).flatten()[:n]
                 if q_ref is not None else q.copy())

    base_dq_initial = kin.base_dq
    R_b0, _ = base_dq_initial.to_pose()

    steps = int(round(duration / dt))
    K = steps + 1
    hist = {
        "t": np.zeros(K),
        "q": np.zeros((K, n)),
        "p_des": np.zeros((K, 3)),
        "p_act": np.zeros((K, 3)),
        "err": np.zeros(K),
        "base_att_deg": np.zeros(K),
        "base_pos": np.zeros((K, 3)),
        "momentum": np.zeros(K),
        "com": np.zeros((K, 3)),
    }

    try:
        for k in range(K):
            t = k * dt
            p_des = np.asarray(p_des_fn(t), dtype=float)
            v_des = np.asarray(v_des_fn(t), dtype=float)

            p_act = kin.forward_kinematics(q)
            err_vec = p_des - p_act

            R_b, p_b = kin.base_dq.to_pose()
            hist["t"][k] = t
            hist["q"][k] = q
            hist["p_des"][k] = p_des
            hist["p_act"][k] = p_act
            hist["err"][k] = np.linalg.norm(err_vec)
            hist["base_att_deg"][k] = rotation_angle_deg(R_b @ R_b0.T)
            hist["base_pos"][k] = p_b
            hist["com"][k] = dynamics.compute_system_com(q)

            if k == steps:
                # record momentum of last commanded step below; final row
                # keeps previous value (no new command at terminal sample)
                hist["momentum"][k] = hist["momentum"][max(k - 1, 0)]
                break

            # --- controller ---
            J_g, J_m, _, H_b, H_bm = dynamics.compute_generalized_jacobian(
                q, method=jacobian_method, return_inertia=True)
            J_ctrl = (J_g if jacobian == "generalized" else J_m)[3:6, :]
            v_cmd = v_des + kp * err_vec
            qdot = resolved_rate_qdot(J_ctrl, v_cmd, damping=damping)
            if k_null > 0.0:
                # Null-space posture task: stay near q_ref without
                # disturbing the Cartesian task.
                J_pinv = J_ctrl.T @ np.linalg.solve(
                    J_ctrl @ J_ctrl.T + damping * np.eye(3), np.eye(3))
                N_proj = np.eye(n) - J_pinv @ J_ctrl
                qdot = qdot + N_proj @ (k_null * (q_ref_arr - q))
            if qdot_max is not None:
                qdot = np.clip(qdot, -qdot_max, qdot_max)

            # --- free-floating plant (truth model) ---
            xb_dot = np.linalg.solve(H_b, dynamics.h0 - H_bm @ qdot)
            h = H_b @ xb_dot + H_bm @ qdot
            hist["momentum"][k] = np.linalg.norm(h - dynamics.h0)

            q = q + qdot * dt
            propagate_base_pose(kin, xb_dot, dt)
    finally:
        kin.base_dq = base_dq_initial

    return hist


# ====================================================================
# OPEN-LOOP JOINT TRAJECTORY WITH BASE PROPAGATION
# ====================================================================

def simulate_free_floating_trajectory(dynamics, q_fn, qdot_fn,
                                      duration, dt=0.01,
                                      jacobian_method="analytical"):
    """
    Open-loop joint-space trajectory on the free-floating system.

    The joint trajectory q(t), qdot(t) is imposed; the base twist follows
    from momentum conservation and the base pose is integrated, yielding
    the accumulated spacecraft translation and attitude drift caused by
    the arm motion (Option-B analysis: pointing loss in degrees).

    The kinematics base pose is restored to its initial value on exit.

    Returns:
        dict with keys t, q, base_att_deg, base_pos, xb_dot (K,6),
        ee_speed_fixed, ee_speed_free, momentum, com
    """
    kin = dynamics.kin
    n = dynamics.n_joints

    base_dq_initial = kin.base_dq
    R_b0, _ = base_dq_initial.to_pose()

    steps = int(round(duration / dt))
    K = steps + 1
    hist = {
        "t": np.zeros(K),
        "q": np.zeros((K, n)),
        "base_att_deg": np.zeros(K),
        "base_pos": np.zeros((K, 3)),
        "xb_dot": np.zeros((K, 6)),
        "ee_speed_fixed": np.zeros(K),
        "ee_speed_free": np.zeros(K),
        "momentum": np.zeros(K),
        "com": np.zeros((K, 3)),
    }

    try:
        for k in range(K):
            t = k * dt
            q = np.asarray(q_fn(t), dtype=float).flatten()[:n]
            qdot = np.asarray(qdot_fn(t), dtype=float).flatten()[:n]

            J_g, J_m, _, H_b, H_bm = dynamics.compute_generalized_jacobian(
                q, method=jacobian_method, return_inertia=True)
            xb_dot = np.linalg.solve(H_b, dynamics.h0 - H_bm @ qdot)
            h = H_b @ xb_dot + H_bm @ qdot

            R_b, p_b = kin.base_dq.to_pose()
            hist["t"][k] = t
            hist["q"][k] = q
            hist["base_att_deg"][k] = rotation_angle_deg(R_b @ R_b0.T)
            hist["base_pos"][k] = p_b
            hist["xb_dot"][k] = xb_dot
            hist["ee_speed_fixed"][k] = np.linalg.norm((J_m @ qdot)[3:6])
            hist["ee_speed_free"][k] = np.linalg.norm((J_g @ qdot)[3:6])
            hist["momentum"][k] = np.linalg.norm(h - dynamics.h0)
            hist["com"][k] = dynamics.compute_system_com(q)

            if k < steps:
                propagate_base_pose(kin, xb_dot, dt)
    finally:
        kin.base_dq = base_dq_initial

    return hist


# ====================================================================
# 6-DOF POSE TRACKING OF A MOVING / TUMBLING TARGET (v0.3.0)
# ====================================================================

def tumbling_target(R0, p0, omega, v_drift):
    """
    Rigid target with constant tumble rate and translational drift.

    Models a realistic capture scenario: the target pose is
        R(t) = exp(skew(omega) t) R0,      p(t) = p0 + v_drift t
    with omega the world-frame tumble angular velocity (rad/s) and
    v_drift the world-frame drift velocity (m/s). Debris and defunct
    satellites typically tumble at 1-10 deg/s.

    Returns:
        pose_fn(t) -> (R (3,3), p (3,))
        twist_fn(t) -> (w (3,), v (3,))   (constant)
    """
    R0 = np.asarray(R0, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    omega = np.asarray(omega, dtype=float)
    v_drift = np.asarray(v_drift, dtype=float)

    def pose_fn(t):
        R = Rotation.from_rotvec(omega * t).as_matrix() @ R0
        return R, p0 + v_drift * t

    def twist_fn(t):
        return omega, v_drift

    return pose_fn, twist_fn


def simulate_pose_tracking(dynamics, q0, target_pose_fn, target_twist_fn,
                           duration, dt=0.01,
                           jacobian="generalized",
                           kp=1.0, ko=1.0, damping=1e-6,
                           qdot_max=1.5,
                           jacobian_method="analytical"):
    """
    Closed-loop 6-DOF pose tracking on the free-floating truth model.

    The end-effector tracks a moving target pose (rotation AND
    translation), e.g. a tumbling client satellite during capture
    approach. The pose error is computed with dual quaternion algebra:
    the orientation feedback is the rotation part of the logarithm of
    the error dual quaternion

        q_err = q_target * conj(q_ee),   xi = log(q_err)

    and the commanded spatial velocity is

        [w_cmd; v_cmd] = [w_t + ko * xi_rot ; v_t + kp * (p_t - p_ee)]

    mapped to joint rates through the full 6xN Jacobian (J_g or J_m)
    by damped least squares. The plant is always free-floating: the
    base twist follows from momentum conservation and the base pose is
    propagated every step. The base pose is restored on exit.

    Args:
        dynamics: SpaceRobotDynamics (plant + h0)
        q0: (N,) initial joint configuration
        target_pose_fn: t -> (R_des (3,3), p_des (3,))
        target_twist_fn: t -> (w_des (3,), v_des (3,)) world frame
        duration, dt: simulated time / step (s)
        jacobian: "generalized" or "fixed" (controller model)
        kp, ko: position / orientation feedback gains (1/s)
        damping: DLS damping factor
        qdot_max: element-wise joint rate limit (rad/s); None disables
        jacobian_method: "analytical" (default) or "numerical"

    Returns:
        dict with keys:
            t, q, pos_err (m), ori_err_deg, base_att_deg, base_pos,
            momentum, com, p_des, p_act
    """
    from .dual_quaternion import DualQuaternion as _DQ, log_dq as _log_dq

    if jacobian not in ("generalized", "fixed"):
        raise ValueError("jacobian must be 'generalized' or 'fixed'")

    kin = dynamics.kin
    n = dynamics.n_joints
    q = np.array(q0, dtype=float).flatten()[:n].copy()

    base_dq_initial = kin.base_dq
    R_b0, _ = base_dq_initial.to_pose()

    steps = int(round(duration / dt))
    K = steps + 1
    hist = {
        "t": np.zeros(K), "q": np.zeros((K, n)),
        "pos_err": np.zeros(K), "ori_err_deg": np.zeros(K),
        "base_att_deg": np.zeros(K), "base_pos": np.zeros((K, 3)),
        "momentum": np.zeros(K), "com": np.zeros((K, 3)),
        "p_des": np.zeros((K, 3)), "p_act": np.zeros((K, 3)),
    }

    try:
        for k in range(K):
            t = k * dt
            R_des, p_des = target_pose_fn(t)
            w_des, v_des = target_twist_fn(t)

            dq_ee = kin.forward_kinematics_dq(q)
            R_act, p_act = dq_ee.to_pose()

            # Dual quaternion pose error: orientation feedback from the
            # rotation part of log(q_target * conj(q_ee)).
            dq_des = _DQ.from_pose(R_des, np.asarray(p_des, dtype=float))
            xi = _log_dq(dq_des * dq_ee.conjugate())
            rot_err = xi[:3]
            pos_err_vec = np.asarray(p_des, dtype=float) - p_act

            R_b, p_b = kin.base_dq.to_pose()
            hist["t"][k] = t
            hist["q"][k] = q
            hist["pos_err"][k] = np.linalg.norm(pos_err_vec)
            hist["ori_err_deg"][k] = np.degrees(np.linalg.norm(rot_err))
            hist["base_att_deg"][k] = rotation_angle_deg(R_b @ R_b0.T)
            hist["base_pos"][k] = p_b
            hist["com"][k] = dynamics.compute_system_com(q)
            hist["p_des"][k] = p_des
            hist["p_act"][k] = p_act

            if k == steps:
                hist["momentum"][k] = hist["momentum"][max(k - 1, 0)]
                break

            # --- controller (full 6-DOF task) ---
            J_g, J_m, _, H_b, H_bm = dynamics.compute_generalized_jacobian(
                q, method=jacobian_method, return_inertia=True)
            J_ctrl = J_g if jacobian == "generalized" else J_m
            v_task = np.concatenate([
                np.asarray(w_des, dtype=float) + ko * rot_err,
                np.asarray(v_des, dtype=float) + kp * pos_err_vec])
            qdot = resolved_rate_qdot(J_ctrl, v_task, damping=damping)
            if qdot_max is not None:
                qdot = np.clip(qdot, -qdot_max, qdot_max)

            # --- free-floating plant ---
            xb_dot = np.linalg.solve(H_b, dynamics.h0 - H_bm @ qdot)
            h = H_b @ xb_dot + H_bm @ qdot
            hist["momentum"][k] = np.linalg.norm(h - dynamics.h0)

            q = q + qdot * dt
            propagate_base_pose(kin, xb_dot, dt)
    finally:
        kin.base_dq = base_dq_initial

    return hist
