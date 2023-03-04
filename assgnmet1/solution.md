# 3DVC_Assignment1

## QCQP

### 1

$$
\mathcal L =\frac 1 2 (Ax-b)^T(Ax-b) + \lambda (x^Tx-\epsilon)
$$


$$
\mathrm d\mathcal L= \frac 1 2 ((A\mathrm d x)^T(Ax-b)+(Ax-b)^TA\mathrm dx)+\lambda ((\mathrm dx)^Tx+x^T\mathrm dx)\\=((Ax-b)^TA+2\lambda x^T)\mathrm d x
$$
$$
\mathrm d \mathcal L = \mathrm{tr}(\mathrm d\mathcal L)=\mathrm{tr}(((Ax-b)^TA+2\lambda x^T)\mathrm d x)=\mathrm{tr}((A^T(Ax-b)+2\lambda x)^T\mathrm d x)
$$



Using the fact that
$$
\mathrm d\mathcal L=\mathrm {tr}((\nabla_x\mathcal L)^T\mathrm dx)
$$

We can obtain
$$
\nabla_x\mathcal L=A^T(Ax-b)+2\lambda x
$$

### 2

Let $\nabla_x\mathcal L=0$, 
$$
x=A^{-1}b
$$

### 3

#### a.

$$
A^T(Ax-b)+2\lambda x=0
$$

$$
x=h(\lambda)=(A^TA+2\lambda I)^{-1}A^Tb
$$

#### b.

Since $A^TA$ is symmetric ($rank(A)=n$), $A^TA=U\Lambda U^T$， and $A^TA+2\lambda I=U(\Lambda+2\lambda I)U^T$, where $\Lambda$ is diagonal and $U$ is an orthogonal matrix.

So, 
$$
h(\lambda)^Th(\lambda)=((A^TA+2\lambda I)^{-1}A^Tb)^T(A^TA+2\lambda I)^{-1}A^Tb\\=((U(\Lambda+2\lambda I)U^T)^{-1}A^Tb)^T(U(\Lambda+2\lambda I)U^T)^{-1}A^Tb\\=(A^Tb)^T(U^{-1})^T((\Lambda+2\lambda I)^{-1})^TU^{-1}(U^{-1})^T(\Lambda+2\lambda I)^{-1}U^{-1}A^Tb\\
=q^T(\Lambda +2\lambda I)^{-2}q
$$
where $q=U^{-1}A^Tb$, which is irrelevant to $\lambda$.

$q=[q_1,q_2,\dots,q_n]^T$, $\Lambda=\mathrm{diag}(\lambda_1,\lambda_2,\dots,\lambda_n)$, so 
$$
(\Lambda+2\lambda I)^{-2}=\mathrm{diag}({(2\lambda+\lambda_1)^{-2}},{(2\lambda+\lambda_2)^{-2}},\dots,{(2\lambda+\lambda_n)^{-2}})
$$
Thus,
$$
h(\lambda)^Th(\lambda)=\sum_{i=1}^n(\frac{q_i}{2\lambda+\lambda_i})^2
$$
Since $A^TA$ is positive semi-definite, $\forall 0\leq i\leq n,  \lambda_i\geq0$, i.e. $2\lambda + \lambda_i \geq 0$ for all i.

So $\forall \lambda \geq 0$, $h(\lambda)^Th(\lambda)$ is monotonically decreasing.

### 4.



### 3D Geometry Processing

### 1



### 2





### 3







### 4

