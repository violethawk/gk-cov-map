# Why the prototype uses penalized splines

A Gaussian process and a penalized spline can both represent a continuous field over the goal plane. The prototype uses a tensor-product cubic B-spline because it makes four requirements unusually transparent:

1. **Exact symmetry decomposition.** The design matrix is split into `B(|x|,y)` and `sign(x)B(|x|,y)`, so the fitted field is explicitly `S + A` rather than approximately symmetric through data augmentation.
2. **Censored likelihood inspection.** Exact contacts and right-censored non-contacts enter one compact likelihood whose gradient can be checked directly.
3. **Fast repeated validation.** The acceptance gate requires 20 fits on 200 simulated shots. A low-rank spline with a fixed penalty completes this locally without approximate GP machinery.
4. **Prior-dominance diagnostics.** The ratio of posterior to prior mean-function standard deviation is available analytically and can be rendered as boundary hatching.

The cost is that smoothing hyperparameters are fixed rather than fully integrated over, and the posterior intervals are Laplace approximations conditional on those choices. The report states that limitation next to the outputs. A later research version could replace the fixed penalties with hierarchical priors or a sparse GP while keeping the same censoring and symmetry interfaces.
