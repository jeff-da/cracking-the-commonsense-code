"""
Run LSA on a concept-feature matrix to get "concept vectors."
"""

from collections import defaultdict
import math

import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD


FEATURES = "./all/CONCS_FEATS_concstats_brm.txt"


def load_features_concepts():
    """
    Returns:
        mat: concept-feature matrix
        features: list of strings
        concepts: list of strings
    """

    concepts_features = defaultdict(set)
    features = set()
    with open(FEATURES, "r") as features_f:
        for line in features_f:
            if line.startswith("Concept"):
                continue
            fields = line.strip().split("\t")
            concept_name, feature_name = fields[:2]

            concepts_features[concept_name].add(feature_name)
            features.add(feature_name)

    # Generate feature IDs.
    features = list(sorted(features))
    feature2idx = {feature: idx for idx, feature in enumerate(features)}

    mat = np.zeros((len(concepts_features), len(features)))
    concepts = list(sorted(concepts_features.keys()))
    for i, concept in enumerate(concepts):
        for feature in concepts_features[concept]:
            mat[i, feature2idx[feature]] = 1.0

    return mat, features, concepts


def make_c2c(mat, features, concepts):
    """
    Make a concept-concept co-occurrence matrix using a concept-feature matrix.

    Returns:
        c2c_mat: n_concepts * n_concepts ndarray
    """
    ret = mat @ mat.T
    lengths = np.linalg.norm(mat, axis=1, keepdims=True)
    ret /= lengths @ lengths.T
    return ret


# http://stackoverflow.com/a/36867493/176075
def calc_row_idx(k, n):
    return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))/2

def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j


def report_closest(mat, concepts, sample_concepts, n=50):
    mat = mat[sample_concepts]
    distances = distance.pdist(mat, metric="cosine")
    distances_square = distance.squareform(distances)
    np.fill_diagonal(distances_square, np.inf)

    topn = np.argsort(distances_square.flatten())[:n]
    for idx in topn:
        i, j = np.unravel_index(idx, distances_square.shape)
        print("%20s\t%20s\t%5f" % (concepts[i], concepts[j], distances_square[i, j]))


def main():
    mat, features, concepts = load_features_concepts()

    sample_concepts = np.random.choice(len(concepts), size=50)
    report_closest(mat, concepts, sample_concepts)

    print("\n\n==================================\n\n")

    svd = TruncatedSVD(n_components=50, algorithm="arpack")
    mat_svd = svd.fit_transform(mat)

    print(mat_svd.shape)
    report_closest(mat_svd, concepts, sample_concepts)

    print("\n\n==================================\n\n")

    # Convert to concept-concept matrix.
    c2c_mat = make_c2c(mat, features, concepts)
    c2c_svd = TruncatedSVD(n_components=50)
    c2c_svd = c2c_svd.fit_transform(c2c_mat)

    report_closest(c2c_svd, concepts, sample_concepts)


if __name__ == "__main__":
    main()
