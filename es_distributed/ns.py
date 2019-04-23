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

def compute_novelty_vs_archive(archive, novelty_vector, k, normalize = False):
    distances = []
    nov = novelty_vector.astype(np.float)

    max_vector = np.zeros(novelty_vector.shape)

    print(novelty_vector)
    if normalize:
        for point in archive:
            print(point)
            max_vector = np.maximum(point.astype(np.float), max_vector)

    max_vector += max_vector == 0

    for point in archive:
        distances.append(euclidean_distance(point.astype(np.float) / max_vector, nov / max_vector))

    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indicies = (distances).argsort()[:k]
    top_k = distances[top_k_indicies]
    return top_k.mean()

def get_mean_bc(env, policy, tslimit, num_rollouts=1):
    novelty_vector = []
    for n in range(num_rollouts):
        rew, t, nv = policy.rollout(env, timestep_limit=tslimit)
        novelty_vector.append(nv)
    return np.mean(novelty_vector, axis=0)