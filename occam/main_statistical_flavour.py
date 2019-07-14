import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from occam.shared import features, generate_data, construct_featured_samples, x_range, y_true, x_domain

# Not; generalization gap is slang for "out-of-sample error" minus "within-sample error"
num_mc = 1000  # How many monte carlo samples to estimate statistics on the Generalization gaps
for num_features in range(3, len(features)):
    # print(f'Start with {num_mc} mc samples on {num_features} features')
    generalization_gaps = []
    for _ in range(num_mc):  # Do many many MC samples to estimate the statistics
        X_train, y_train = generate_data(10)  # Bound should hold on any sample, so sample them

        # Our hypothesis being that num_features could be chosen from a set of features. This has finite VC dimension
        # Now our ERM algorithm consists of enumerating all these combinations of hypotheses and
        # choose (one of) the hypothesis with lowest in-sample error
        hypotheses = []
        for idxs in combinations(range(len(features)), num_features):
            X_feature = construct_featured_samples(X_train, features, idxs)

            beta, _, _, _ = np.linalg.lstsq(X_feature, y_train)

            error = np.mean(np.power(y_train - X_feature @ beta, 2))
            hypotheses.append((error, beta, idxs))

        # Our ERM algorithm being the arming of in-sample errors of hypotheses
        hypothesis = min(hypotheses, key=lambda x: x[0])
        error, beta, idxs = hypothesis

        # Calculate the generalization error
        # Note that these theorems hold for any loss function. Here we pick the Mean Squared Error loss
        y_pred = construct_featured_samples(x_range, features, idxs) @ beta
        error_generalization = np.mean(np.power(y_true - y_pred, 2))

        generalization_gaps.append(error_generalization - error)

    # Finally print our statistics and behold
    print(f'Num feats {num_features}: generalization gap, ave: {np.mean(generalization_gaps):15.2f}, '
          f'max: {np.max(generalization_gaps):15.2f}, '
          f'+1std {np.mean(generalization_gaps)+np.std(generalization_gaps):15.2f}')
    plt.hist(generalization_gaps, bins=np.linspace(0, 1, num=100), label=f'Generalization gaps for num_features={num_features}')
plt.legend()
plt.show()
