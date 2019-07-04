# adversarial examples

- reference: https://arxiv.org/abs/1902.10660
- doesn't make much sense for categorical variables
- can do adversarial training on smth about dist to splits



# misc

- do rule lists apply to continuous variables?
- rule lists needs a more easily usable implementation - sk8er
- sgd to optimize over splits to be short
  - loss - want most points to use fewest rules
  - optimizing an upper bound on the tree error using stochastic gradient descent (Norouzi et al. 2015)
- stable rule lists?
  - iterative rule lists?
- optimal classification tree?
  - solving rule lists via mixed integer optimization like in bertsimas paper?
- how slow is optimizing over multi level decision tree?