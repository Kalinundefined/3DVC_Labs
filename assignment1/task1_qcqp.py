from utils import path_
import numpy as np
import scipy

def solveQCQP(A, b, eps, ITER_STEPS = 40, return_lambda = False):
    import jax
    import jax.numpy as jnp

    # if lambda == 0
    x = np.linalg.pinv(A) @ b

    if x.T @ x <= eps:
        return x
    # if lambda != 0

    # x = h(\lambda)
    lam2x = lambda lam: jnp.linalg.inv(A.T @ A + 2 * lam * jnp.eye(A.shape[-1])) @ A.T @ b 

    # compute x.T @ x from lambda for Newton's method
    def x_norm2(lam):
        # x = h(lambda)
        x = jnp.linalg.inv(A.T @ A + 2 * lam * jnp.eye(A.shape[-1])) @ A.T @ b 
        return x.T @ x - eps
    def x_norm2_np(lam):
        x = np.linalg.inv(A.T @ A + 2 * lam * np.eye(A.shape[-1])) @ A.T @ b 
        return x.T @ x - eps
    x_norm2_grad = jax.grad(x_norm2)

    # initial lambda value
    lam = 1e-8
    
    # use newton's method to solve lambda
    for _ in range(ITER_STEPS):
        if x_norm2_np(lam) < 1e-11:
            break
        small_delta_times = 0

        delta = x_norm2(lam) / x_norm2_grad(lam)
        # avoid grad explosion
        if np.abs(delta) < 1e-7:
            small_delta_times += 1
            if small_delta_times >= 3:
                delta = np.sign(delta) * 0.1
        if np.abs(delta) > 10e6:
            delta = np.sign(delta) * 10e6
        lam -= delta

        #print(f'lam={lam}, x2={x_norm2(lam)}, grad={x_norm2_grad(lam)}, lam-={x_norm2(lam) / x_norm2_grad(lam)}')
    if return_lambda:
        return np.asarray(lam2x(lam)), np.asarray(lam)
    return np.asarray(lam2x(lam))

def main():
    data = np.load(path_('QCQP.npz'))
    A, b, eps = data["A"], data["b"], data["eps"]

    print(solveQCQP(A, b, eps))

if __name__ == "__main__":
    main()

# my solution = 
# [ 0.09795038 -0.12841614  0.04953709  0.06482819  0.04341127  0.06206457
#  -0.16418658  0.03840064  0.3091618  -0.12387919  0.06729932 -0.0128481
#  -0.03535068 -0.10851554 -0.02132299 -0.12418856  0.18965702 -0.1572289
#  -0.17646319  0.04182666  0.09246276  0.11353753 -0.1029304  -0.0304796
#   0.03294807 -0.2371428  -0.14864613 -0.07861543  0.1591745  -0.22602603]