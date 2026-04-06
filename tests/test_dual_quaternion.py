"""Tests for dual quaternion module."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from space_robot_dq import DualQuaternion, quaternion_multiply, log_dq


class TestDualQuaternionBasics:
    def test_identity_matrix(self):
        dq = DualQuaternion()
        T = dq.to_matrix()
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_pure_rotation_roundtrip(self):
        R = Rotation.from_rotvec([0, 0, np.pi / 4]).as_matrix()
        p = np.zeros(3)
        dq = DualQuaternion.from_pose(R, p)
        R_out, p_out = dq.to_pose()
        np.testing.assert_allclose(R, R_out, atol=1e-10)
        np.testing.assert_allclose(p, p_out, atol=1e-10)

    def test_pure_translation_roundtrip(self):
        R = np.eye(3)
        p = np.array([1.0, 2.0, 3.0])
        dq = DualQuaternion.from_pose(R, p)
        R_out, p_out = dq.to_pose()
        np.testing.assert_allclose(R, R_out, atol=1e-10)
        np.testing.assert_allclose(p, p_out, atol=1e-10)

    def test_general_pose_roundtrip(self):
        R = Rotation.from_euler('xyz', [0.3, -0.5, 0.7]).as_matrix()
        p = np.array([0.5, -1.2, 0.8])
        dq = DualQuaternion.from_pose(R, p)
        R_out, p_out = dq.to_pose()
        np.testing.assert_allclose(R, R_out, atol=1e-10)
        np.testing.assert_allclose(p, p_out, atol=1e-10)

    def test_multiplication_matches_matrix(self):
        R1 = Rotation.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
        p1 = np.array([1.0, 0.0, 0.0])
        R2 = Rotation.from_euler('xyz', [-0.2, 0.4, 0.1]).as_matrix()
        p2 = np.array([0.0, 1.0, 0.5])

        dq12 = DualQuaternion.from_pose(R1, p1) * DualQuaternion.from_pose(R2, p2)

        T1 = np.eye(4); T1[:3,:3] = R1; T1[:3,3] = p1
        T2 = np.eye(4); T2[:3,:3] = R2; T2[:3,3] = p2
        np.testing.assert_allclose(dq12.to_matrix(), T1 @ T2, atol=1e-8)

    def test_conjugate_is_inverse(self):
        R = Rotation.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
        dq = DualQuaternion.from_pose(R, [1, 0, 0])
        T_id = (dq * dq.conjugate()).to_matrix()
        np.testing.assert_allclose(T_id, np.eye(4), atol=1e-8)

    def test_unit_norm_after_multiply(self):
        dq1 = DualQuaternion.from_pose(np.eye(3), [1, 2, 3])
        dq2 = DualQuaternion.from_pose(Rotation.from_rotvec([0.1, 0, 0]).as_matrix(), [0, 0, 0])
        dq12 = dq1 * dq2
        assert abs(np.linalg.norm(dq12.qr) - 1.0) < 1e-10
        assert abs(np.dot(dq12.qr, dq12.qd)) < 1e-8


class TestLogDQ:
    def test_identity_gives_zero(self):
        xi = log_dq(DualQuaternion())
        np.testing.assert_allclose(xi, np.zeros(6), atol=1e-8)

    def test_pure_translation_zero_rotation(self):
        dq = DualQuaternion.from_pose(np.eye(3), [0.5, 0, 0])
        xi = log_dq(dq)
        np.testing.assert_allclose(xi[:3], np.zeros(3), atol=1e-6)

    def test_pure_rotation_zero_translation(self):
        R = Rotation.from_rotvec([0, 0, 0.5]).as_matrix()
        dq = DualQuaternion.from_pose(R, [0, 0, 0])
        xi = log_dq(dq)
        np.testing.assert_allclose(xi[3:], np.zeros(3), atol=1e-6)
