import numpy as np


def update_RK4(P0, q0, h, phi, b, lamb=1.0):
    """
    Applies Runge-Kutta (RK4) scheme to solve P and q in from Ricatti ODEs, regarding
    one new data point. Note that P and q are not updated at the same time: P is updated
    twice when q is updated once.

        Args:
            P0: The initial state of P for current update, with shape [dim, dim].
                It is also the terminal state of the previous update.
            q0: The initial state of q for current update, with shape [dim, 1].
                It is also the terminal state of the previous update.
            h: The step size.
            phi: The value of basis functions for the current data point, with
                shape [1, dim].
            b: The targets, with shape [1, 1].
            lamb: The length of the interval.

        Returns:
            P: Updated P from Ricatti ODEs.
            q: Updated q from Ricatti ODEs on current update.
    """
    dim = P0.shape[0]
    # Ps = np.zeros(shape=[int(1.0/step), dim, dim])
    # qs = np.zeros(shape=[int(1.0/step), dim, 1])
    P_current, q_current = P0, q0
    hP, hq = 0.5 * h, h
    T = np.matmul(phi.T, phi)
    f = np.matmul(phi.T, b)
    for i in range(int(lamb / h)):
        # update P first, two times
        Pk1 = -np.matmul(P_current.T, np.matmul(T, P_current))
        Pk2 = -np.matmul(
            (P_current + 0.5 * hP * Pk1).T, np.matmul(T, P_current + 0.5 * hP * Pk1)
        )
        Pk3 = -np.matmul(
            (P_current + 0.5 * hP * Pk2).T, np.matmul(T, P_current + 0.5 * hP * Pk2)
        )
        Pk4 = -np.matmul((P_current + hP * Pk3).T, np.matmul(T, P_current + hP * Pk3))
        P_mid = P_current + hP * (Pk1 + 2 * Pk2 + 2 * Pk3 + Pk4) / 6
        Pk1 = -np.matmul(P_mid.T, np.matmul(T, P_mid))
        Pk2 = -np.matmul(
            (P_mid + 0.5 * hP * Pk1).T, np.matmul(T, P_mid + 0.5 * hP * Pk1)
        )
        Pk3 = -np.matmul(
            (P_mid + 0.5 * hP * Pk2).T, np.matmul(T, P_mid + 0.5 * hP * Pk2)
        )
        Pk4 = -np.matmul((P_mid + hP * Pk3).T, np.matmul(T, P_mid + hP * Pk3))
        P_new = P_mid + hP * (Pk1 + 2 * Pk2 + 2 * Pk3 + Pk4) / 6
        # update q then
        # interpolation of P is required for RK-4 on q
        P_mid_T = np.matmul(P_mid, T)
        P_mid_f = np.matmul(P_mid, f)
        qk1 = -np.matmul(np.matmul(P_current, T), q_current) + np.matmul(P_current, f)
        qk2 = -np.matmul(P_mid_T, q_current + 0.5 * hq * qk1) + P_mid_f
        qk3 = -np.matmul(P_mid_T, q_current + 0.5 * hq * qk2) + P_mid_f
        qk4 = -np.matmul(np.matmul(P_new, T), q_current + hq * qk3) + np.matmul(
            P_new, f
        )
        q_new = q_current + hq * (qk1 + 2 * qk2 + 2 * qk3 + qk4) / 6

        P_current = P_new
        q_current = q_new
    return P_current, q_current


def update(P0, q0, h, phi, b, lamb=1.0):
    dim = P0.shape[0]
    P_current, q_current = P0, q0
    hP, hq = 0.5 * h, h
    T = phi.T @ phi
    f = phi.T @ b
    for i in range(int(lamb / h)):
        # update P first, two times
        Pk1 = -np.transpose(P_current) @ T @ P_current
        Pk2 = (
            -np.transpose(P_current + 0.5 * hP * Pk1) @ T @ (P_current + 0.5 * hP * Pk1)
        )
        Pk3 = (
            -np.transpose(P_current + 0.5 * hP * Pk2) @ T @ (P_current + 0.5 * hP * Pk2)
        )
        Pk4 = -np.transpose(P_current + hP * Pk3) @ T @ (P_current + hP * Pk3)
        P_mid = P_current + hP * (Pk1 + 2 * Pk2 + 2 * Pk3 + Pk4) / 6

        Pk1 = -np.transpose(P_mid) @ T @ P_mid
        Pk2 = -np.transpose(P_mid + 0.5 * hP * Pk1) @ T @ (P_mid + 0.5 * hP * Pk1)
        Pk3 = -np.transpose(P_mid + 0.5 * hP * Pk2) @ T @ (P_mid + 0.5 * hP * Pk2)
        Pk4 = -np.transpose(P_mid + hP * Pk3) @ T @ (P_mid + hP * Pk3)
        P_new = P_mid + hP * (Pk1 + 2 * Pk2 + 2 * Pk3 + Pk4) / 6
        # update q then
        # interpolation of P is required for RK-4 on q
        P_mid_T = P_mid @ T
        P_mid_f = P_mid @ f
        qk1 = -P_current @ T @ q_current + P_current @ f
        qk2 = -P_mid_T @ (q_current + 0.5 * hq * qk1) + P_mid_f
        qk3 = -P_mid_T @ (q_current + 0.5 * hq * qk2) + P_mid_f
        qk4 = -P_new @ T @ (q_current + hq * qk3) + P_new @ f
        q_new = q_current + hq * (qk1 + 2 * qk2 + 2 * qk3 + qk4) / 6

        P_current = P_new
        q_current = q_new
    return P_current, q_current
