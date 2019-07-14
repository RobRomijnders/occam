import numpy as np


def f_true(data):
    # The true labeling function. We will add noise later
    return data**3 - 3 * data ** 2 + 2 * data + 1/3


x_domain = [-0.5, 2.5]  # Domain of the data
x_range = np.linspace(x_domain[0], x_domain[1], num=500)  # Equi-spaced sample of the domain
y_true = f_true(x_range)  # labels for testing generalization
epsilon = 0.2  # Generalization error after which we consider a failure

# Uncomment to plot the true labeling function
# f = plt.figure()
# plt.scatter(x_range, y_true)
# plt.show()


def construct_featured_samples(data, feats, indices):
    # For every feature, apply it to the data and append as a column
    feature_columns = [feats[i](data) for i in indices]
    # Then stack these feature columns over the -1 dimension
    return np.stack(feature_columns, axis=-1)


# A list of all possible features on the
features = [lambda x: np.power(x, 0),
            lambda x: np.power(x, 1),
            lambda x: np.power(x, 2),
            lambda x: np.power(x, 3),
            lambda x: np.power(x, 4),
            lambda x: np.power(x, 5),
            lambda x: np.sin(2 * np.pi * x),
            lambda x: np.cos(2 * np.pi * x),
            lambda x: np.exp(x)]


def generate_data(num_samples):
    # Sample points uniformly from the domain, get the labels from the true function plus some added noise
    x_points = -1/2 + 3 * np.random.rand(num_samples)
    return x_points, f_true(x_points) + 0.1 * np.random.randn(num_samples)
