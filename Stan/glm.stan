data {
  int<lower=1> I, G;  // number of observations and groups
  array[I] int<lower=0, upper=1> y;  // binary outcome array
  vector[I] x;  // continuous covariate
  array[I] int<lower=1, upper=G> g;  // group identifier
}

parameters {
  vector[G] alpha;  // group-level intercepts
  real beta;  // slope
}

model {
  // priors
  alpha ~ logistic(0, 1);
  beta ~ normal(0, 1);
  
  // likelihood
  y ~ bernoulli_logit(alpha[g] + beta * x);
  // y ~ bernoulli(inv_logit(alpha[g] + beta * x));
}

generated quantities {
  vector[I] log_lik;
  for (i in 1:I) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | alpha[g[i]] + beta * x[i]);
  }
  array[I] int yrep = bernoulli_logit_rng(alpha[g] + beta * x);
}
