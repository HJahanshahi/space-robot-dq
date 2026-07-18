"""
Reproduce the figures of the space-robot-dq paper.

Generates (300 dpi PNG in the working directory):
    figure3_trajectory.png   sinusoidal trajectory: EE speed (fixed vs
                             free-floating), base reaction, momentum
    figure4_mass_ratio.png   manipulability ratio and Jacobian difference
                             vs base-to-arm mass ratio
    figure5_tracking.png     resolved-rate tracking: fixed-base vs
                             generalized-Jacobian controller + base
                             attitude drift

All parameters are stated explicitly and every run is deterministic,
so the numbers printed here are the numbers cited in the paper.

Usage:
    python examples/generate_paper_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from space_robot_dq import (
    SpaceRobotKinematics,
    SpaceRobotDynamics,
    simulate_free_floating_trajectory,
    simulate_resolved_rate_tracking,
    simulate_pose_tracking,
    tumbling_target,
    quintic_line,
)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 110,
})

DPI = 300

# --------------------------------------------------------------------
# Common system: built-in 7-DOF SRS preset
# --------------------------------------------------------------------
BASE_MASS = 100.0

kin = SpaceRobotKinematics()
dyn = SpaceRobotDynamics(kinematics=kin, base_mass=BASE_MASS)
ARM_MASS = float(np.sum(dyn.link_masses))
print(f"System: 7-DOF SRS preset, base {BASE_MASS:.0f} kg, "
      f"total arm mass {ARM_MASS:.2f} kg "
      f"(auto-generated link properties)")

# Sinusoidal joint trajectory q(t) = q0 + A sin(w t)  (Figures 3)
Q0_TRAJ = np.array([0.3, 0.5, -0.3, 1.0, 0.2, 0.6, 0.1])
A_TRAJ = np.array([0.4, 0.3, 0.5, 0.4, 0.3, 0.2, 0.3])
W_TRAJ = np.array([1.0, 1.5, 0.8, 1.2, 2.0, 1.0, 1.5]) * np.pi / 2
T_TRAJ, DT_TRAJ = 2.0, 0.01


# ====================================================================
# FIGURE 3 — trajectory simulation
# ====================================================================
def figure3():
    h = simulate_free_floating_trajectory(
        dyn,
        q_fn=lambda t: Q0_TRAJ + A_TRAJ * np.sin(W_TRAJ * t),
        qdot_fn=lambda t: A_TRAJ * W_TRAJ * np.cos(W_TRAJ * t),
        duration=T_TRAJ, dt=DT_TRAJ)

    t = h["t"]
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True,
                             gridspec_kw={"hspace": 0.42})

    ax = axes[0]
    ax.plot(t, h["ee_speed_fixed"], "b-", lw=2,
            label=r"Fixed-base $\|\mathbf{J}_m\dot{\mathbf{q}}\|$")
    ax.plot(t, h["ee_speed_free"], "r--", lw=2,
            label=r"Free-floating $\|\mathbf{J}_g\dot{\mathbf{q}}\|$")
    ax.fill_between(t, h["ee_speed_free"], h["ee_speed_fixed"],
                    color="gray", alpha=0.25, label="Coupling loss")
    ax.set_ylabel("EE linear speed (m/s)")
    ax.set_title("(a)", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=3, frameon=False, columnspacing=1.2)

    ax = axes[1]
    ax.plot(t, np.linalg.norm(h["xb_dot"][:, :3], axis=1), "k-", lw=2)
    ax.set_ylabel(r"$\|\dot{\mathbf{x}}_{b,\mathrm{lin}}\|$ (m/s)")
    ax.set_ylim(bottom=0)
    ax.set_title("(b)", loc="left", fontweight="bold")

    ax = axes[2]
    ax.semilogy(t, np.maximum(h["momentum"], 1e-18), color="#1a1a2e", lw=1.5)
    ax.axhline(1e-15, color="red", ls=":", lw=1.5)
    ax.text(t[-1], 1.35e-15, "machine precision", color="red",
            ha="right", va="bottom", fontsize=10)
    ax.set_ylim(1e-17, 5e-15)
    ax.set_ylabel(r"$\|\mathbf{h}\|$")
    ax.set_xlabel("Time (s)")
    ax.set_title("(c)", loc="left", fontweight="bold")

    fig.savefig("figure3_trajectory.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    loss = 1.0 - h["ee_speed_free"] / np.maximum(h["ee_speed_fixed"], 1e-12)
    print("\n[Figure 3] sinusoidal trajectory "
          f"(q0={Q0_TRAJ.tolist()}, A={A_TRAJ.tolist()}, "
          f"w={np.round(W_TRAJ, 4).tolist()} rad/s, T={T_TRAJ}s, dt={DT_TRAJ}s)")
    print(f"  EE speed at t=0: fixed {h['ee_speed_fixed'][0]:.3f} m/s, "
          f"free {h['ee_speed_free'][0]:.3f} m/s "
          f"({100*loss[0]:.0f}% overestimation by fixed-base model)")
    print(f"  Velocity loss range: {100*loss.min():.0f}%–{100*loss.max():.0f}%")
    print(f"  Max base linear reaction: "
          f"{np.linalg.norm(h['xb_dot'][:, :3], axis=1).max():.3f} m/s")
    print(f"  Max |momentum|: {h['momentum'].max():.2e}")
    print(f"  Base attitude drift after {T_TRAJ}s: "
          f"{h['base_att_deg'][-1]:.2f} deg")


# ====================================================================
# FIGURE 4 — mass ratio sweep
# ====================================================================
def figure4():
    q_eval = Q0_TRAJ  # evaluate coupling at the trajectory's start config
    ratios = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 50, 100, 200])
    w_ratio, j_diff = [], []

    for r in ratios:
        d = SpaceRobotDynamics(kinematics=kin, base_mass=r * ARM_MASS)
        J_g, J_m, _ = d.compute_generalized_jacobian(q_eval)
        w_g, w_f = d.compute_dynamic_manipulability(q_eval)
        w_ratio.append(w_g / w_f)
        j_diff.append(np.max(np.abs(J_g - J_m)))

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()
    l1, = ax1.semilogx(ratios, w_ratio, "s-", color="tab:blue", lw=2,
                       label=r"$w_{free}/w_{fixed}$")
    l2, = ax2.semilogx(ratios, j_diff, "o--", color="tab:orange", lw=2,
                       label=r"$\max|\mathbf{J}_g-\mathbf{J}_m|$")
    ax1.axhline(1.0, color="gray", ls=":", lw=1)
    ax1.set_xlabel("Base-to-arm mass ratio")
    ax1.set_ylabel("Manipulability ratio", color="tab:blue")
    ax2.set_ylabel(r"$\max|\mathbf{J}_g-\mathbf{J}_m|$", color="tab:orange")
    ax1.legend(handles=[l1, l2], loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
               columnspacing=1.5)
    fig.savefig("figure4_mass_ratio.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Figure 4] mass-ratio sweep at q={q_eval.tolist()}")
    for r, w, jd in zip(ratios, w_ratio, j_diff):
        if r in (0.5, 5, 10, 100):
            print(f"  ratio {r:>5}:1  w_free/w_fixed = {w:.3f}  "
                  f"max|Jg-Jm| = {jd:.4f}")


# ====================================================================
# FIGURE 5 — resolved-rate tracking (new in v0.3.0)
# ====================================================================
def figure5():
    q0 = np.array([0.3, 0.4, -0.2, 0.8, 0.1, 0.5, 0.0])
    displacement = np.array([-0.20, 0.10, -0.15])
    T, dt, kp = 4.0, 0.01, 1.0

    p0 = kin.forward_kinematics(q0)
    p_des_fn, v_des_fn = quintic_line(p0, p0 + displacement, T)

    runs = {}
    for jac in ("fixed", "generalized"):
        runs[jac] = simulate_resolved_rate_tracking(
            dyn, q0, p_des_fn, v_des_fn, T, dt=dt, jacobian=jac, kp=kp)

    hf, hg = runs["fixed"], runs["generalized"]
    t = hg["t"]

    fig, axes = plt.subplots(3, 1, figsize=(7, 9.5), sharex=True,
                             gridspec_kw={"hspace": 0.45})

    ax = axes[0]
    ax.plot(t, hf["p_des"][:, 0], "k:", lw=1.5, label="Desired $x$")
    ax.plot(t, hf["p_act"][:, 0], "b-", lw=2,
            label=r"Fixed-base ctrl ($\mathbf{J}_m$)")
    ax.plot(t, hg["p_act"][:, 0], "r--", lw=2,
            label=r"Generalized ctrl ($\mathbf{J}_g$)")
    ax.set_ylabel("EE $x$-position (m)")
    ax.set_title("(a)", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=3, frameon=False, columnspacing=1.2)

    ax = axes[1]
    ax.semilogy(t, np.maximum(hf["err"] * 1000, 1e-3), "b-", lw=2,
                label=r"Fixed-base ctrl ($\mathbf{J}_m$)")
    ax.semilogy(t, np.maximum(hg["err"] * 1000, 1e-3), "r--", lw=2,
                label=r"Generalized ctrl ($\mathbf{J}_g$)")
    ax.set_ylim(bottom=1e-3)
    ax.set_ylabel("Tracking error (mm)")
    ax.set_title("(b)", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=2, frameon=False, columnspacing=1.2)

    ax = axes[2]
    ax.plot(t, hg["base_att_deg"], "k-", lw=2)
    ax.set_ylabel("Base attitude drift (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(bottom=0)
    ax.set_title("(c)", loc="left", fontweight="bold")

    fig.savefig("figure5_tracking.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    rms_f = np.sqrt(np.mean(hf["err"] ** 2))
    rms_g = np.sqrt(np.mean(hg["err"] ** 2))
    print(f"\n[Figure 5] resolved-rate tracking "
          f"(q0={q0.tolist()}, line move {displacement.tolist()} m, "
          f"T={T}s, dt={dt}s, kp={kp})")
    print(f"  Fixed-base ctrl:   RMS {rms_f*1000:8.3f} mm, "
          f"final {hf['err'][-1]*1000:8.3f} mm")
    print(f"  Generalized ctrl:  RMS {rms_g*1000:8.3f} mm, "
          f"final {hg['err'][-1]*1000:8.3f} mm")
    print(f"  Error ratio (fixed/generalized): {rms_f/rms_g:.0f}x")
    print(f"  Base attitude drift over the move: "
          f"{hg['base_att_deg'][-1]:.2f} deg")
    print(f"  Momentum conservation during tracking: "
          f"max |h| = {max(hf['momentum'].max(), hg['momentum'].max()):.2e}")


# ====================================================================
# FIGURE 6 — 6-DOF pose tracking of a tumbling target (new in v0.3.0)
# ====================================================================
def figure6():
    q0 = np.array([0.3, 0.4, -0.2, 0.8, 0.1, 0.5, 0.0])
    rate_dps = 5.0                     # realistic debris tumble rate
    axis = np.array([0.3, 1.0, 0.5])
    axis = axis / np.linalg.norm(axis)
    T, dt, kp, ko = 10.0, 0.01, 1.0, 1.0

    R0, p0 = kin.forward_kinematics_dq(q0).to_pose()
    pose_fn, twist_fn = tumbling_target(
        R0, p0, np.radians(rate_dps) * axis, np.zeros(3))

    runs = {}
    for jac in ("fixed", "generalized"):
        runs[jac] = simulate_pose_tracking(
            dyn, q0, pose_fn, twist_fn, T, dt=dt,
            jacobian=jac, kp=kp, ko=ko)

    hf, hg = runs["fixed"], runs["generalized"]
    t = hg["t"]
    FLOOR_MM, FLOOR_DEG = 1e-3, 1e-4

    fig, axes = plt.subplots(3, 1, figsize=(7, 9.5), sharex=True,
                             gridspec_kw={"hspace": 0.45})

    ax = axes[0]
    ax.semilogy(t, np.maximum(hf["pos_err"] * 1000, FLOOR_MM), "b-", lw=2,
                label=r"Fixed-base ctrl ($\mathbf{J}_m$)")
    ax.semilogy(t, np.maximum(hg["pos_err"] * 1000, FLOOR_MM), "r--", lw=2,
                label=r"Generalized ctrl ($\mathbf{J}_g$)")
    ax.set_ylabel("Position error (mm)")
    ax.set_title("(a)", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=2, frameon=False, columnspacing=1.2)

    ax = axes[1]
    ax.semilogy(t, np.maximum(hf["ori_err_deg"], FLOOR_DEG), "b-", lw=2,
                label=r"Fixed-base ctrl ($\mathbf{J}_m$)")
    ax.semilogy(t, np.maximum(hg["ori_err_deg"], FLOOR_DEG), "r--", lw=2,
                label=r"Generalized ctrl ($\mathbf{J}_g$)")
    ax.set_ylabel("Orientation error (deg)")
    ax.set_title("(b)", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=2, frameon=False, columnspacing=1.2)

    ax = axes[2]
    ax.plot(t, hg["base_att_deg"], "k-", lw=2)
    ax.set_ylabel("Base attitude drift (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(bottom=0)
    ax.set_title("(c)", loc="left", fontweight="bold")

    fig.savefig("figure6_pose_tracking.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    ss = slice(100, 1001)   # steady state after initial transient
    print(f"\n[Figure 6] tumbling-target pose tracking "
          f"(q0={q0.tolist()}, tumble {rate_dps} deg/s about "
          f"{np.round(axis, 3).tolist()}, T={T}s, dt={dt}s, kp={kp}, ko={ko})")
    for name, h in (("Fixed-base ctrl", hf), ("Generalized ctrl", hg)):
        print(f"  {name}: pos RMS {np.sqrt(np.mean(h['pos_err']**2))*1000:8.4f} mm, "
              f"ori RMS {np.sqrt(np.mean(h['ori_err_deg']**2)):8.5f} deg, "
              f"steady-state pos {np.mean(h['pos_err'][ss])*1000:8.4f} mm, "
              f"ori {np.mean(h['ori_err_deg'][ss]):8.5f} deg")
    rp = np.sqrt(np.mean(hf['pos_err']**2))/np.sqrt(np.mean(hg['pos_err']**2))
    ro = np.sqrt(np.mean(hf['ori_err_deg']**2))/np.sqrt(np.mean(hg['ori_err_deg']**2))
    print(f"  Error ratio (fixed/generalized): {rp:.0f}x position, {ro:.0f}x orientation")
    print(f"  Base attitude drift over {T}s of station-keeping: "
          f"{hg['base_att_deg'][-1]:.2f} deg")


if __name__ == "__main__":
    figure3()
    figure4()
    figure5()
    figure6()
    print("\nSaved: figure3_trajectory.png, figure4_mass_ratio.png, "
          "figure5_tracking.png, figure6_pose_tracking.png")
