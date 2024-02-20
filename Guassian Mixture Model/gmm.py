import numpy as np
import matplotlib.pyplot as plt

X_train = np.loadtxt('./gmm_dataset.csv', delimiter=",")

# Return the determinant of the matrix
def det(S):
    return np.prod(S)

# Implementation of GMM
def EM_GMM(X, K, pi=None, mu=None, S=None, Max_iter=500):
    X = np.array(X)
    n = X.shape[0]
    D = X.shape[1]
    TOL = 1e-5
    # initialization
    if pi is None:
        pi = np.random.rand(K)
        pi = pi/sum(pi)
    if mu is None:
        mu = np.random.rand(K, D)
    if S is None:
        S = np.random.rand(K, D) + 1

    r = np.zeros((n, K))
    r_i = np.zeros(n)
    l_iter_1 = 0
    for iter in range(Max_iter):
        # E step
        for i in range(n):
            for k in range(K):
                A = X[i] - mu[k]
                AT = np.transpose(A)
                ATS = np.array([AT[i]/S[k][i] for i in range(D)])
                r[i][k] = pi[k] * np.power(det(S[k]), -0.5) * np.exp(-0.5 * np.matmul(ATS, A))

        for i in range(n):
            r_i[i] = sum(r[i])
        for i in range(n):
            for k in range(K):
                r[i][k] = r[i][k]/r_i[i]

        # Compute negative log-likelihood
        l_iter = -1 * sum(np.log(r_i) * np.power(2 * np.pi, -0.5))
        if iter > 1 and np.abs(l_iter - l_iter_1) <= TOL * np.abs(l_iter):
            break
        l_iter_1 = l_iter

        # M Step
        r_k = np.zeros(K)
        for k in range(K):
            r_k[k] = sum([r[i][k] for i in range(n)])
            pi[k] = r_k[k]/n
            mu[k] = sum([r[i][k]*X[i]/r_k[k] for i in range(n)])
            for d in range(D):
                S[k][d] = sum([r[i][k]*np.power(X[i][d], 2) for i in range(n)]) / r_k[k] - np.power(mu[k][d], 2)

    return pi, mu, S, l_iter


# Choose optimal k by trial
#L = []
#K = [k+1 for k in range(10)]
#for k in K:
#    pi, mu, S, l = EM_GMM(X_train, k)
#    print(f'Finished k={k}.')
#    L.append(l)

#plt.plot(K, L)
#plt.ylabel("negative log-likelihood")
#plt.xlabel("The value of k")
#plt.show()

# k = 5
pi, mu, S, l = EM_GMM(X_train, 5)
pi_sorted = []
mu_sorted = []
S_sorted = []

pi = list(pi)
mu = list(np.ndarray.round(mu, 3))
S = list(np.ndarray.round(S, 3))

for i in range(len(pi)):
    p = min(pi)
    index = np.where(pi==p)[0][0]
    pi_sorted.append(pi[index])
    mu_sorted.append(mu[index])
    S_sorted.append(S[index])
    pi.pop(index)
    mu.pop(index)
    S.pop(index)

# Reporting parameters, sort the components in increasing order of mixing weights
print([round(p, 4) for p in pi_sorted])
print(np.matrix(mu_sorted))
print(np.matrix(S_sorted))
