# Occam's razor two ways: statistical and Bayesian

This project discusses Occam's razor. We will take two views on this principle: the statistical learning view and the Bayesian learning view. The project contains minimal code, but explores a fundamental concept in Machine learning (ML). Whatever view you take on ML.

Occam's razor has been told in many versions. Among others

  * _Entities are not to be multiplied without necessity_
  * _Accept the simplest explanation that fits the data_
  * _We are to admit no more causes of natural things than such as are both true and sufficient to explain their appearances._
  
Occam's razor helps us compare models when we have multiple models explaining the data. For example when regression data can be modelled with 5 or 20 order polynomials; when scatter points can be explained with Gaussian Mixture Models or Normalizing flows; or when a classification data is modelled with a Naive Bayes Classifier or a huge Neural Network. Even when these terms ring no bell to you, everyone has a situation where he needed to choose between two models. Then Occam's razor tells us to find the simplest model.

# How to use?

English quotes are all fine, but how to use this Occam's razor. We should choose the simplest model, but how is simplicity defined. And what if some model is a little more complex, but explains the data also better. Then what model to choose?

This dilemma is where our two views kick in. Statistical learning theory has its view on model comparison and Bayesian learning has its views on model comparison. In short:

  * In Bayesian learning, we marginalise over unknown variables. The marginalisation for complex models inherently embodies a penalty
  * In statistical learning, we aim to bound generalization errors. The bound between in-sample error and out-sample error grows for more complex models.
  
Let's discuss these views one by one in the following subsections.

## Statistical Learning view

Statistical learing aims to bound the generalization error and this bound grows with complexity. Complexity here bases on a description language. Any model requiring a longer description we deem more complex. This description language complies with human conversations. If one model requires you a two hour presentation to explain, but another model can be explained on the back of an envelope, the latter is probably simpler. One can also think of describing each parameter separately. So that a longer description is really more parameters being described. 

Using this description language, we bound the out-sample error as `out_sample_error = in_sample_error + sqrt((description + log(2 / delta)) / (2 * num_samples))`. In words: as our description grows, the generalization gap grows via square root. 

More formally, we repeat theorem 7.7 from Understanding Machine Learning (see further reading section below).

![theorem](www.linktoimage.com)

### Implementation

The script `main_statistical_flavour.py` implements an experiment to show this bound. The script trains a regression problem with increasing number of features. It prints the generalization gaps averaged over many repeats of the experiment.

A run of the script will result in numbers like these
```bash
Num feats 3: generalization gap, ave:            0.33, max:          255.79, +1std            8.42
Num feats 4: generalization gap, ave:            3.38, max:         1655.49, +1std           65.69
Num feats 5: generalization gap, ave:          197.98, max:        81081.70, +1std         3594.80
Num feats 6: generalization gap, ave:          276.86, max:       129575.35, +1std         4975.92
Num feats 7: generalization gap, ave:         6960.57, max:      2215182.10, +1std        93943.58
Num feats 8: generalization gap, ave:      9390682.62, max:   8659856654.92, +1std    283450505.52
```

When increasing the number of features, we increase the complexity of the model. Statistical learning theory tells us that we can bound the generalization gap less. Confirming this, we see the max gap increase.


## Bayesian view

Bayesian learning aims to marginalise any unknown parameter that we are uncertain about. Models are then compared by the _evidence_ we have for a model. The evidence for a model is the probability that a model assigns to the data, after any source of uncertainty is marginalised out. This marginalisation embodies Occam's razor, more parameters have lower probability. Therefore a marginalisation results in lower evidence in the face of equal likelihood. 

Interlude: it took me personally many months to figure out this intuition. Statistical learning clearly states a formula that increases with increasing complexity. In calculating the evidence, the penalty is more subtle. Let's take some time to explore the intuition.

The following formula calculates the evidence for model comparison

<img alt="$p(D|H) = \int_\theta p(D,\theta|H)d\theta= \int_\theta p(D|H,\theta)p(\theta|H)d\theta$" src="https://github.com/RobRomijnders/occam/blob/master/svgs/d15df27da7c656d3e1e4d034107543d0.svg" align="middle" width="356.079405pt" height="26.48448pt"/>

Components
  * <img alt="$D$" src="https://github.com/RobRomijnders/occam/blob/master/svgs/78ec2b7008296ce0561cf83393cb746d.svg" align="middle" width="14.06625pt" height="22.46574pt"/> is the data.
  * <img alt="$H$" src="https://github.com/RobRomijnders/occam/blob/master/svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg" align="middle" width="14.999985pt" height="22.46574pt"/> is the model. Any assumptions on the model are condensed in this variable.
  * <img alt="$\theta$" src="https://github.com/RobRomijnders/occam/blob/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg" align="middle" width="8.1735885pt" height="22.83138pt"/> are all the unknown parameters in our model.
  * We name <img alt="$p(D|H,\theta)$" src="https://github.com/RobRomijnders/occam/blob/master/svgs/2bc2efb539de5a25c5fe6c2445db7a37.svg" align="middle" width="69.254625pt" height="24.6576pt"/> the likelihood and <img alt="$p(\theta|H)$" src="https://github.com/RobRomijnders/occam/blob/master/svgs/97a39984a4c3acb147ca6c9a2b780592.svg" align="middle" width="48.79578pt" height="24.6576pt"/> the prior.

In words, the evidence is the _probability a model assigns to the data, marginalised out all unknown parameters of the model_.

Now we have two intuitions why this would embody Occam's razor. The first is due to E.T. Jaynes, the second is due to David Mackay. (sources stated in the further reading section)

__Intuition 1__ (Due to E.T. Jaynes)

The evidence integrates out the parameters, from the multiplication of likelihood and prior. Let's pick two models that assign equal likelihood to the data. So <img alt="$p(D|H_1,\theta)$" src="https://github.com/RobRomijnders/occam/blob/master/svgs/f727783e8cf64e2e05831dcb76ab3d99.svg" align="middle" width="76.206735pt" height="24.6576pt"/> and <img alt="$p(D|H_2,\theta)$" src="https://github.com/RobRomijnders/occam/blob/master/svgs/08fcc685026c66e9f4c975b2c674920a.svg" align="middle" width="76.206735pt" height="24.6576pt"/> are equal. However, model 2 uses more parameters. With more probability, the prior is spread over more parameters. Therefore, for the likely parameters, the probability density is lower. Integrating the multiplication of prior and likelihood will thus result in a lower value for model 2.

__Intuition 2__ (Due to David Mackay)

The integral for the evidence can be approximated by Laplace's method. Then the evidence is really a multiplication of `evidence = best_fit_likelihood * prior * posterior_width`. Assuming a flat prior around the best fit for the parameters. Our Laplace approximation is `evidence = best_fit_likelihood * 1 / prior_width * posterior_width`. For models assigning equal likelihood to the data, the comparison build on an Occam factor of `Occam_factor = posterior_width / prior_width`. A complex model has more parameters to fit, so will be narrower in the posterior. But the prior for a complex model needs to spread probability mass over more parameters, so has a wider prior. Therefore, a complex model has a way larger Occam factor. In total, we approximate evidence is larger for more complex models.

### Implementation

Even if the previous intuitions made no sense to you, the experiments will. The script `main_bayesian_flavour.py` calculates the evidence integral using Monte Carlo approximation. We use the same problem, data and features as the experiments for the statistical flavours: we increase the number of features fit to regression data. 

A run of the script will result in numbers like these:

```bash
At num features     3 we have evidence      -124.91 nats
At num features     4 we have evidence      -795.19 nats
At num features     5 we have evidence     -6753.77 nats
At num features     6 we have evidence     -8015.36 nats
At num features     7 we have evidence     -8561.80 nats
At num features     8 we have evidence    -18610.57 nats
```

When increasing the number of features, we increase the complexity of the model. Bayesian learning tells us that we the evidence for a model decreases. Confirming this, we see the evidence decrease.

# Conclusion

Occam's razor instructs us to use simple models. Two running interpretations of simplicity and performance of models are the Statistical and the Bayesian view. This post discusses both views, provides intuition and implements both views on a regression problem. 

# Further reading

  * [Wikipedia on Occam's razor](https://en.wikipedia.org/wiki/Occam%27s_razor)
  * [David Mackay, Information theory, inference and learning algorithms, chapter 28](https://www.inference.org.uk/itprnn/book.pdf)
  * [Ben-David, Ben-Schwartz, Understanding Machine learning, chapter 7, non-uniform learning](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
  * [E.T. Jaynes, The logic of science, chapter 20](https://bayes.wustl.edu/etj/prob/book.pdf)
    * I particularly found section 20.3 useful.  