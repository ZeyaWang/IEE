import numpy as np
MACHINE_EPSILON = np.finfo(np.double).eps

def weight_fusion_matrices(P_list, Niters=50, tol=1e-8):
    num = len(P_list)
    omega = np.array([1 / num for _ in range(num)])
    Obj = []
    for iter in range(Niters):
        # Given w, update U
        U = np.zeros_like(P_list[0])
        # Compute the weighted sum
        for m, w in zip(P_list, omega):
            U += w * m

        # Given U, update w
        for i in range(num):
            omega[i] = 1 / np.linalg.norm(U - P_list[i], 'fro')
        omega = omega / np.sum(omega)

        # Calculate the objective function value
        obj = 0
        for i in range(num):
            obj += np.linalg.norm(U - P_list[i], 'fro')
        Obj.append(obj)

        if iter > 1 and abs(Obj[iter - 1] - Obj[iter]) < tol:
            print('iteration stops at', iter)
            break
    return U, omega

