import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import time

def k_means(data: NDArray, k: int, max_rounds: int=50, tol: float=1e-3):
    r"""
    Performs K-means clustering

    Parameters:
    -----------
        data : Array-like
               Source data of shape [m,n]
        k : integer
                Number of K centroids.
        max_rounds: integer
                Maximum number of rounds to iterate
        tol : float
                Maximum tolerance before exiting if nothing happens

    Returns:
    --------
        ks : Array-like
             Centroids, in shape of [k,n]
        idx : Array-like
              Indexes of k to which each data belongs
              in shape of [m,2], where
              [m,0] includes the index of k that is closest
              [m,1] includes the euclidian distance
        err : List
              Historical error calculations
    """
    def generate_random_k(data, k, num_dim):
        r"""
        Generates the first Random K elements
        """
        ks = np.zeros((k, num_dim))
        for dim in range(num_dim):
            min_dim = np.min(data[:,dim])
            max_dim = np.max(data[:,dim])
            ks[:,dim] = np.random.uniform(min_dim, max_dim, k)
        return ks
    
    def find_distance(data, ks):
        diff = data[:, np.newaxis, :] - ks[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return distances

    def find_nearest(dist):
        return np.argmin(dist, 1), np.mean(np.min(dist, 1))

    def update_ks(idx, data):
        unique_groups, inverse = np.unique(idx, return_inverse=True)
        means = np.zeros((len(unique_groups), data.shape[1]))
        for i in range(data.shape[1]):
            means[:, i] = np.bincount(inverse, weights=data[:, i]) / np.bincount(inverse)
        return means

    def plot_diagram(data, ks, i, idx):
        # Simple 2D plot for data and centroids.
        # - color each point by its assigned cluster (idx)
        # - plot centroids with a distinct marker
        plt.clf()
        num_dim = data.shape[1]
        # Use first two dimensions for plotting; if data is 1D, plot against zeros
        if num_dim < 2:
            x = data[:, 0]
            y = np.zeros_like(x)
        else:
            x = data[:, 0]
            y = data[:, 1]

        k = ks.shape[0]
        # Choose a colormap with at least k distinct colors.
        try:
            cmap = plt.cm.get_cmap('tab20', k)
            colors = cmap(np.arange(k))
        except Exception:
            cmap = plt.cm.get_cmap('hsv', k)
            colors = cmap(np.arange(k))

        for cluster in range(k):
            mask = (idx == cluster)
            plt.scatter(x[mask], y[mask], s=20, color=colors[cluster], label=f'cluster {cluster}')

        # Plot centroids
        cent_x = ks[:, 0]
        cent_y = ks[:, 1] if ks.shape[1] > 1 else np.zeros_like(cent_x)
        plt.scatter(cent_x, cent_y, marker='X', s=120, color='k', edgecolor='white', linewidth=1)

        plt.title(f'K-means iteration {i}')
        plt.legend(loc='best', fontsize='small', markerscale=0.7)
        # short pause to allow animation effect when called in a loop
        plt.pause(0.1)


    def time_to_break(ks, prev_ks, tol):
        abs1 = np.abs(ks - prev_ks)
        mean1 = np.mean(abs1)
        return mean1 < tol

    num_data = data.shape[0]
    num_dim = data.shape[1]
    err = []
    ks = generate_random_k(data, k, num_dim)
    for i in range(max_rounds):

        dist = find_distance(data, ks)
        idx, e = find_nearest(dist)
        err.append(e)
        new_ks = update_ks(idx, data)
        if time_to_break(new_ks, ks, tol):
            break
        else:
            ks= new_ks
        time.sleep(1)
        plot_diagram(data, ks, i, idx)
    return ks, idx, err



NUM_POINTS = 1000
NUM_DIM = 2

random_numbers = np.random.uniform(-10,10, NUM_POINTS * NUM_DIM)
random_numbers = random_numbers.reshape(NUM_POINTS,-1)
out = k_means(random_numbers, 3)
print(out)