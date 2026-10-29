# =============================================================================
# Shared manifold for Reliable Clustering Evaluation in Deep Learning
# =============================================================================
# File: towards.py

#
# Description:
# This module implements a method to learn a shared manifold for internal vaildation
# in deep clustering evaluation. Our method can be adapted with different manifold 
# learning method. This implementation is based on TSNE, and extends scikit-learn's 
# TSNE implementation and overrides several key methods in TSNE. 
#
# Based on:
# scikit-learn: https://github.com/scikit-learn/scikit-learn
# TSNE source: sklearn.manifold._t_sne.TSNE
#
# Key Features:
# - Inherits from sklearn.manifold.TSNE
# - Overrides `_fit_transform()`
#
# Usage:
# model = IEE(n_components=2, perplexity=30, my_param=0.5)
# embedding = model.fit_transform(X)
#
# Notes:
# Ensure compatibility with the sklearn version 1.5.1
# This code is for research and prototyping purposes.
#
# =============================================================================




import numpy as np
from scipy.spatial.distance import pdist, squareform
MACHINE_EPSILON = np.finfo(np.double).eps
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import check_non_negative
from sklearn.manifold import _utils  # type: ignore
from sklearn.manifold import TSNE
import scipy.stats as stats


from agg import weight_fusion_matrices

def _joint_probabilities_towards(distances, desired_perplexity, verbose):
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose
    )
    return conditional_P


class IEE(TSNE):
    def _fit_prob(self, X):
        if self.learning_rate == "auto":
            self.learning_rate_ = X.shape[0] / self.early_exaggeration / 4
            self.learning_rate_ = np.maximum(self.learning_rate_, 50)
        else:
            self.learning_rate_ = self.learning_rate

        if self.metric == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")
            check_non_negative(
                X,
                (
                    "With metric='precomputed', X "
                    "should contain positive distances."
                ),
            )

        if self.method == "exact":
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("IEE computing pairwise distances...")
                if self.metric == "euclidean":
                    distances = pairwise_distances(X, metric=self.metric, squared=True)
                else:
                    raise ValueError("Other metrics are not supported in this version")

            if np.any(distances < 0):
                raise ValueError(
                    "All distances should be positive, the metric given is not correct"
                )

            Pt = _joint_probabilities_towards(distances, self.perplexity, self.verbose)
        else:
            raise ValueError("method must be exact in this version")
        return Pt

    def _fit_fusion(self, X_list):
        n_samples = X_list[0].shape[0]
        random_state = check_random_state(self.random_state)
        P_list = []

        for X in X_list:
            Pt = self._fit_prob(X)
            P_list.append(Pt)

        if self.method == "exact":
            P, omega = weight_fusion_matrices(P_list)
            P = P + P.T
            sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
            P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), "All probabilities should be less or then equal to one"
        else:
            #### placeholder for future development of other methods like bh ###
            raise ValueError("method must be exact at this version")

        X_embedded = 1e-4 * random_state.standard_normal(
            size=(n_samples, self.n_components)
        ).astype(np.float32)
        degrees_of_freedom = max(self.n_components - 1, 1)
        embedding = self._tsne(
            P,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
            skip_num_points=0,
        )
        return embedding, P, omega

    def fit_transform(self, X_list):
        """get the unified manifold embedding
        Parameters
        ----------
        X_list: a list of X; X has {array-like, sparse matrix} of shape (n_samples, n_features)
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self._max_iter = 1000
        for X in X_list:
            self._check_params_vs_input(X)
        embedding, P, omega = self._fit_fusion(X_list)
        self.embedding_ = embedding
        return self.embedding_, P, omega


def clustering_accuracy(true_labels, predicted_labels):
    true_labels = np.array(true_labels, dtype='int64')
    predicted_labels = np.array(predicted_labels, dtype='int64')
    contingency_matrix = []
    true_classes = np.unique(true_labels)
    num_predicted_classes = np.amax(predicted_labels) + 1
    for class_id in np.arange(len(true_classes)):
      contingency_matrix.append(
          np.bincount(
              predicted_labels[true_labels == true_classes[class_id]],
              minlength=num_predicted_classes,
          )
      )
    contingency_matrix = np.asarray(contingency_matrix).T
    assigned_rows, assigned_columns = linear_sum_assignment(
        np.max(contingency_matrix) - contingency_matrix
    )
    return float(
        contingency_matrix[assigned_rows, assigned_columns].sum()
    ) / len(true_labels)


def calinski_harabasz_score(x,y):
    if len(np.unique(y)) > 1:
        return metrics.calinski_harabasz_score(x,y)
    else:
        return - np.inf

def davies_bouldin_score(x,y):
    if len(np.unique(y)) > 1:
        return - metrics.davies_bouldin_score(x,y)
    else:
        return - np.inf

def silhouette_score(x,y, metric):
    if len(np.unique(y)) > 1:
        return metrics.silhouette_score(x, y, metric=metric)
    else:
        return - np.inf


def clustering_score(x,y, metric):
    if metric == 'dav':
        return davies_bouldin_score(x,y)
    elif metric == 'ch':
        return calinski_harabasz_score(x,y)
    else:
        return silhouette_score(x, y, metric=metric)

def get_files_with_substring_and_suffix(directory, substring, suffix):
    all_files = os.listdir(directory)
    files = [file for file in all_files if substring in file and file.endswith(suffix)]
    return files

def kendalltau(score1, score2):
    '''
    calculate kendal tau
    '''
    stat, pval = stats.kendalltau(score1, score2)
    return np.round(stat, 3), pval

def spearmanr(score1, score2):
    '''
    calculate spearman correlation
    '''
    stat, pval = stats.spearmanr(score1, score2)
    return np.round(stat, 3), pval

if __name__ == '__main__':
    import numpy as np
    from sklearn import metrics
    import os, h5py
    import pickle as pk
    from collections import defaultdict
    import argparse
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.cluster import normalized_mutual_info_score


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='FRGC')
    parser.add_argument('--task', default='DEPICT_hyper')
    parser.add_argument('--n_components', type=int, default=2, help='number of components')
    parser.add_argument('--perplexity', type=float, default=30.0, help='perplexity')

    args = parser.parse_args()
    eval_data = args.dataset
    task_name = args.task
    tpath = os.path.join(task_name, 'iee_{}_{}'.format(args.n_components, args.perplexity))
    os.makedirs(tpath, exist_ok=True)


    file_list_path = os.path.join('file_list', task_name, '{}.txt'.format(eval_data))
    with open(file_list_path, encoding='utf-8') as file:
        modelFiles = [line.strip() for line in file if line.strip()]

    labels = {}
    features = {}
    scored = defaultdict(dict)

    tfname = os.path.join('datasets', eval_data, 'data4torch.h5')
    dataset_file = h5py.File(tfname, 'r')
    truth = np.squeeze(np.array(dataset_file['labels']))
    data = np.array(dataset_file['data'])
    data = np.stack([data[i].flatten() for i in range(data.shape[0])], axis=0)

    output_dir = os.path.join(task_name, 'deep_clustering_outputs')
    if 'JULE' in task_name:
        for m in modelFiles:
            ffname = os.path.join(output_dir, 'feature{}.h5'.format(m))
            lfname = os.path.join(output_dir, 'label{}.h5'.format(m))
            features[m] = np.array(h5py.File(ffname, 'r')['feature'])
            labels[m] = np.squeeze(np.array(h5py.File(lfname, 'r')['label']))
    else:
        for m in modelFiles:
            files = np.load(os.path.join(output_dir, m))
            features[m] = np.array(files['y_features'])
            labels[m] = np.squeeze(np.array(files['y_pred']))

    x_list = []
    for m in modelFiles:
        x = features[m]
        x_list.append(x)

    to_manifold = IEE(n_components=args.n_components, random_state=1, method='exact', perplexity=args.perplexity)
    X_towards, common_P, omega = to_manifold.fit_transform(x_list)
    kl = to_manifold.kl_divergence_

    print(task_name, eval_data, data.shape)
    scored['kl'] = kl
    scored['omega'] = dict(zip(modelFiles, omega.tolist()))
    scored['common_P'] = common_P
    scored['X_towards'] = X_towards

    davs, chs, euclideans, nmis, accs = [], [], [], [], []
    for i, key in enumerate(modelFiles):
        y = labels[key]
        for metric in ['dav', 'ch', 'euclidean']:
            scored[metric][key] = clustering_score(X_towards, y, metric=metric)
        scored['nmi'][key] = normalized_mutual_info_score(truth, y)
        scored['acc'][key] = clustering_accuracy(truth, y)
        accs.append(scored['acc'][key])
        nmis.append(scored['nmi'][key])
        davs.append(scored['dav'][key])
        chs.append(scored['ch'][key])
        euclideans.append(scored['euclidean'][key])


    print('kendall tau in terms of nmi')
    print(kendalltau(euclideans, nmis))
    print(kendalltau(davs, nmis))
    print(kendalltau(chs, nmis))
    print('kendall tau in terms of acc')
    print(kendalltau(euclideans, accs))
    print(kendalltau(davs, accs))
    print(kendalltau(chs, accs))
    print('spearman rank correlation in terms of nmi')
    print(spearmanr(euclideans, nmis))
    print(spearmanr(davs, nmis))
    print(spearmanr(chs, nmis))
    print('spearman rank correlation in terms of acc')
    print(spearmanr(euclideans, accs))
    print(spearmanr(davs, accs))
    print(spearmanr(chs, accs))


    with open(os.path.join(tpath,'{}_{}_score.pkl'.format(task_name,eval_data)), 'wb') as file:
        pk.dump(scored, file)
