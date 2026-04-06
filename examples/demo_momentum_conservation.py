"""
Example: Free-floating space robot — momentum conservation and generalized Jacobian.

Demonstrates:
1. Forward/inverse kinematics
2. Base reaction to manipulator motion
3. Generalized vs fixed-base Jacobian
4. Effect of mass ratio on dynamic coupling
"""

import numpy as np
from space_robot_dq import (
    SpaceRobotKinematics, SpaceRobotDynamics,
    quaternion_distance, quaternion_to_euler,
)


def main():
    # ---- Setup ----
    kin = SpaceRobotKinematics(verbose=True)
    dyn = SpaceRobotDynamics(kinematics=kin)
    dyn.get_mass_properties_summary()

    # ---- Forward kinematics ----
    print("\n" + "="*60)
    print("1. FORWARD KINEMATICS")
    print("="*60)

    q = np.array([0.0, 0.3, 0.0, 1.2, 0.0, -0.4, 0.0])
    pos = kin.forward_kinematics(q)
    pos6, quat = kin.forward_kinematics_6dof(q)
    euler = np.rad2deg(quaternion_to_euler(quat))

    print(f"  Joint angles: {q}")
    print(f"  EE position:  {pos}")
    print(f"  EE quaternion: {quat}")
    print(f"  EE euler (deg): roll={euler[0]:.1f}, pitch={euler[1]:.1f}, yaw={euler[2]:.1f}")

    # ---- Inverse kinematics ----
    print("\n" + "="*60)
    print("2. INVERSE KINEMATICS")
    print("="*60)

    target = np.array([0.4, 0.2, 0.5])
    q_ik = kin.inverse_kinematics(target)
    achieved = kin.forward_kinematics(q_ik)
    error = np.linalg.norm(achieved - target)
    print(f"  Target:   {target}")
    print(f"  Achieved: {achieved}")
    print(f"  Error:    {error*1000:.4f} mm")

    # ---- Momentum conservation ----
    print("\n" + "="*60)
    print("3. MOMENTUM CONSERVATION")
    print("="*60)

    qdot = np.array([0.0, 0.1, 0.0, 0.2, 0.0, -0.1, 0.0])
    xb_dot = dyn.compute_base_velocity(q, qdot)

    print(f"  Joint velocities: {qdot}")
    print(f"  Base linear vel:  [{xb_dot[0]:.6f}, {xb_dot[1]:.6f}, {xb_dot[2]:.6f}] m/s")
    print(f"  Base angular vel: [{xb_dot[3]:.6f}, {xb_dot[4]:.6f}, {xb_dot[5]:.6f}] rad/s")

    # Verify conservation
    h = dyn.compute_system_momentum(q, qdot, xb_dot)
    print(f"  Total momentum:   |h| = {np.linalg.norm(h):.2e} (should be ≈0)")

    # ---- Generalized vs fixed-base Jacobian ----
    print("\n" + "="*60)
    print("4. GENERALIZED vs FIXED-BASE JACOBIAN")
    print("="*60)

    J_g, J_m, _ = dyn.compute_generalized_jacobian(q)

    ee_vel_fixed = J_m @ qdot
    ee_vel_free = J_g @ qdot

    print(f"  Fixed-base EE velocity: {ee_vel_fixed[3:]}")
    print(f"  Free-float EE velocity: {ee_vel_free[3:]}")
    print(f"  Difference (linear):    {ee_vel_free[3:] - ee_vel_fixed[3:]}")

    w_free, w_fixed = dyn.compute_dynamic_manipulability(q)
    print(f"\n  Manipulability — fixed: {w_fixed:.6f}, free: {w_free:.6f}")
    print(f"  Ratio (free/fixed):     {w_free/w_fixed:.3f}")

    # ---- Mass ratio study ----
    print("\n" + "="*60)
    print("5. EFFECT OF MASS RATIO ON COUPLING")
    print("="*60)

    arm_mass = np.sum(dyn.link_masses)
    print(f"  {'Mass ratio':>12}  {'|J_g - J_m|':>12}  {'w_free/w_fixed':>14}  {'|ẋ_base|':>10}")
    print(f"  {'-'*55}")

    for ratio in [0.5, 1, 2, 5, 10, 50, 100]:
        base_m = arm_mass * ratio
        d = SpaceRobotDynamics(kinematics=kin, base_mass=base_m,
                                base_inertia=np.diag([base_m * 0.1]*3))
        Jg, Jm, _ = d.compute_generalized_jacobian(q)
        wf, wfx = d.compute_dynamic_manipulability(q)
        xb = d.compute_base_velocity(q, qdot)
        print(f"  {ratio:>10.0f}:1  {np.max(np.abs(Jg-Jm)):>12.6f}  "
              f"{wf/wfx if wfx > 0 else 0:>14.4f}  {np.linalg.norm(xb):>10.6f}")

    print("\n" + "="*60)
    print("✅ Example complete")
    print("="*60)


if __name__ == "__main__":
    main()
