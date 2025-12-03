import numpy as np
from sklearn.mixture import GaussianMixture
import joblib  # or pickle

gmm1 = joblib.load("models/ubm_gmm.joblib")
gmm2 = joblib.load("models/target_gmm.joblib")


def compute_likelihood(vector, gmms):
    """
    Compute the likelihood of a vector given a list of GMMs.

    Args:
        vector (np.ndarray): Input feature vector of shape (39,)
        gmms (list): List of GaussianMixture objects

    Returns:
        list: Log-likelihoods for each GMM
    """
    vector = np.array(vector).reshape(1, -1)

    likelihoods = []
    for i, gmm in enumerate(gmms, start=1):
        log_likelihood = gmm.score(vector)
        likelihoods.append(log_likelihood)
        print(f"Log-likelihood for GMM{i}: {log_likelihood:.4f}")

    return likelihoods


if __name__ == "__main__":
    custom_vector = np.arange(1 / 8, 1 / 8 * 40, 1 / 8)

    gmms = [gmm1, gmm2]
    compute_likelihood(custom_vector, gmms)
