import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


def standardise_dataset(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    return X_train, X_test


def extract_mean(channel_features):
    mean = np.mean(channel_features, axis=0)
    return mean


def extract_variance(channel_features):
    var = np.var(channel_features, axis=0)
    return var


def extract_poly_fit(channel_features):
    poly_coeff = np.polyfit(range(len(channel_features)), channel_features, 1)
    return poly_coeff.flatten()


def extract_skewness(channel_features):
    skewness = skew(channel_features, axis=0)
    return skewness


def extract_average_amplitude_change(channel_features):
    amplitude_changes = []
    for i in range(0, len(channel_features)-1):
        amplitude_changes.append(np.abs(channel_features[i+1]-channel_features[i]))
    return np.mean(amplitude_changes, axis=0)