import numpy as np
MACHINE_EPSILON = np.finfo(np.double).eps

def weight_fusion_matrices(P_list):
    Niters = 50
    num = len(P_list)
    omega = np.array([1 / num for _ in range(num)])
    Obj = []
    for iter in range(Niters):
        # Fix w, update U
        U = np.zeros_like(P_list[0])
        # Compute the weighted sum
        for m, w in zip(P_list, omega):
            U += w * m

        # Fix U, update w
        for v in range(num):
            omega[v] = 1 / np.linalg.norm(U - P_list[v], 'fro')
        omega = omega / np.sum(omega)

        # Calculate objive function value
        obj = 0
        for v in range(num):
            obj += np.linalg.norm(U - P_list[v], 'fro')
        Obj.append(obj)

        if iter > 1 and abs(Obj[iter - 1] - Obj[iter]) < 1e-8:
            print('iteration stops at', iter)
            break
    return U, omega

