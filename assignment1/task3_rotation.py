import numpy as np
import scipy
from utils import path_
from task1_qcqp import solveQCQP


def skew(vector: list | np.ndarray) -> np.ndarray:
    vector = list(vector)
    return np.asarray([[0, -vector[2], vector[1]],
                       [vector[2], 0, -vector[0]],
                       [-vector[1], vector[0], 0]])

def solve_align_problem(data: np.ndarray, epsilon=1e-6, max_iter=100):
   
    R = np.diag([1. for _ in range(3)])

    X, Y, eps = data["X"], data["Y"], epsilon

    print(f'initial loss={np.linalg.norm(X-Y)}')

    for _ in range(max_iter):
        # the A as described in the submitted PDF
        A = np.asarray([R @ skew(X[:, i])
                       for i in range(X.shape[-1])]).sum(axis=0)
        # the b as described
        B = np.asarray([R @ X[:, i] - Y[:, i]
                       for i in range(X.shape[-1])]).sum(axis=0)

        d_omega = solveQCQP(A, B, eps)

        # update R by the desired delta_omega
        R = R @ scipy.linalg.expm(skew(d_omega))

        # print the 
        # 1) Sum of euclidean distance of the point pairs
        # 2) Original matrix 2-norm of RX-Y
        # after each iteration
        print(
            f'sum_loss={np.linalg.norm(np.asarray([R @ X[:,i] - Y[:,i] for i in range(X.shape[-1])]).sum(axis=0))}, norm_loss={np.linalg.norm(R @ X - Y)}')
    return R

def main():
    data = np.load(path_("teapots.npz"))
    R = solve_align_problem(data)
    print(R)


if __name__ == "__main__":
    main()
# return =    
# [[-0.92774621  0.21958011  0.30178062]
#  [-0.23771369  0.27569461 -0.93138858]
#  [-0.28771369 -0.93582961 -0.20357744]]
# sum_loss(Sum of euclidean distance of the point pairs)=3.158382966817268, norm_loss(matrix norm of RX-Y)=13.025303338276972