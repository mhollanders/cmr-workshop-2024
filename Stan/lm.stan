data {
  int<lower=0> I;  // number of observations
  vector[I] y, x;  // outcome vector and covariate
}

parameters {
  real alpha, beta;  // intercept and slope
  real<lower=0> sigma;  // SD
}

model {
  // priors
  alpha ~ normal(0, 5);
  beta ~ normal(0, 1);
  sigma ~ exponential(1);
  
  // likelihood
  y ~ normal(alpha + beta * x, sigma);
}

generated quantities {
  // pointwise log-likelihood values
  vector[I] log_lik;
  for (i in 1:I) {
    log_lik[i] = normal_lpdf(y[i] | alpha + beta * x[i], sigma);
  }
  
  // posterior predictive distribution
  array[I] real yrep = normal_rng(alpha + beta * x, sigma);
}
