import numpy as np

def solveQCQP():
    import jax
    import jax.numpy as jnp

    ITER_STEPS = 10

    data = np.load("/home/karin/assignment/3dvc_1/assgnmet1/QCQP.npz")
    A, b, eps = data["A"], data["b"], data["eps"]
    
    # if lambda == 0
    x = np.linalg.inv(A.T @ A) @ A.T @ b
    if x.T @ x <= eps:
        return x
    # if lambda != 0

    # compute x.T @ x from lambda
    def x_norm2(lam):
        # x = h(lambda)
        x = jnp.linalg.inv(A.T @ A + 2 * lam * jnp.eye(A.shape[-1])) @ A.T @ b 
        return x.T @ x - eps
    h_grad = jax.grad(x_norm2)
    lam = 0.
    
    for _ in range(ITER_STEPS):
        f_val = x_norm2(lam)
        lam = lam - f_val/ h_grad(lam)

    return np.asarray(lam)


if __name__ == "__main__":
    print(solveQCQP())