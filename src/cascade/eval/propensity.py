"""Propensity model for XMTC (Jain et al., 2016), pyxclib convention.

p_l = 1 / (1 + C * exp(-A * ln(N_l + B)))
C = (ln(N) - 1) * (B + 1)^A

where N is the number of training instances and N_l is the number of training
instances carrying label l. Inverse propensity 1/p_l is used as the reward weight
in PSP@k / PS-nDCG@k.
"""

import numpy as np
from scipy.sparse import csr_matrix


def compute_inverse_propensity(Y_train: csr_matrix, A: float = 0.55, B: float = 1.5) -> np.ndarray:
    n_train = Y_train.shape[0]
    label_freq = np.asarray(Y_train.sum(axis=0)).ravel()  # N_l per label

    C = (np.log(n_train) - 1) * (B + 1) ** A
    p_l = 1.0 / (1.0 + C * np.exp(-A * np.log(label_freq + B)))
    return 1.0 / p_l
