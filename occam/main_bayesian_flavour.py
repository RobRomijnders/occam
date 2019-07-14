import numpy as np
from occam.shared import features, generate_data, construct_featured_samples, x_range, y_true, x_domain
from scipy.stats import multivariate_normal

std_prior = 1.0  # Prior standard deviation over the parameters of the Linear model
std_likeli = 1.0  # Standard deviation for observing the data (likelihood)

num_data_samples = 30 # How many samples to take from the data distribution

num_mc = 10000  # How many monte carlo samples to estimate statistics on the Generalization gaps
for num_features in range(3, len(features)):
    # Note: in principle we should also MC estimate over the choice of features and sampling of data, but
    # for simplicity, we omit that step

    prior = multivariate_normal(cov=std_prior**2 * np.eye(num_features))

    X_train, y_train = generate_data(num_data_samples)
    X_feature = construct_featured_samples(X_train, features, list(range(num_features)))

    evidence = np.mean([np.sum(multivariate_normal(y_train, cov=std_likeli * np.eye(num_data_samples)).logpdf(X_feature @ prior.rvs())) for _ in range(num_mc)])

    print(f'At num features {num_features:5} we have evidence {evidence:12.2f} nats')
