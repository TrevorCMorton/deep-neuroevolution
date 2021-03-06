import numpy as np


def euclidean_distance(x, y):
    n, m = len(x), len(y)
    if n > m:
        a = np.linalg.norm(y - x[:m])
        b = np.linalg.norm(y[-1] - x[m:])
    else:
        a = np.linalg.norm(x - y[:n])
        b = np.linalg.norm(x[-1] - y[n:])
    return np.sqrt(a**2 + b**2)


def compute_novelty_vs_archive(archive, novelty_vector, k, normalize=False):
    distances = []
    nov = novelty_vector.astype(np.float)

    max_vector = np.zeros(novelty_vector.shape)

    if normalize:
        for point in archive:
            max_vector = np.maximum(point.astype(np.float), max_vector)

    max_vector += max_vector == 0

    for point in archive:
        distances.append(euclidean_distance(point.astype(np.float) / max_vector, nov / max_vector))

    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indicies = (distances).argsort()[:k]
    top_k = distances[top_k_indicies]
    if top_k.size != 0:
        return top_k.mean()
    else:
        return 0


def levenshtein_distance(x, y):
    n, m = len(x), len(y)
    a = np.zeros(shape=(m, n))
    for i in range(1, m):
        a[i, 0] = i

    for j in range(1, n):
        a[0, j] = j

    for j in range(1, n):
        for i in range(1, m):
            substCost = 0
            if x[j - 1] == y[i-1]:
                substCost = 0
            else:
                substCost = 1
            a[i,j] = min(a[i-1, j] + 1, a[i, j-1] + 1, a[i-1,j-1] + substCost)

    return a[m - 1, n - 1]


def compute_novelty_vs_archive_levenshtein(archive, novelty_vector, k):
    distances = []

    for point in archive:
        distances.append(levenshtein_distance(point, novelty_vector))

    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indicies = (distances).argsort()[:k]
    top_k = distances[top_k_indicies]
    if top_k.size != 0:
        return top_k.mean()
    else:
        return 0

def get_policy_novelty(env, policy, tslimit):
    rew, t, nv = policy.rollout(env, timestep_limit=tslimit)
    return nv


def get_mean_bc(env, policy, tslimit, num_rollouts=1):
    novelty_vector = []
    for n in range(num_rollouts):
        rew, t, nv = policy.rollout(env, timestep_limit=tslimit)
        novelty_vector.append(nv)
    return np.mean(novelty_vector, axis=0)