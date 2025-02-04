data {
  int<lower=0> I;  // number of observations
  vector[I] y;  // outcome vector
  array[I] int<lower=1, upper=2> g;  // group identifier
}

parameters {
  vector[2] mu;  // means
  vector<lower=0>[2] sigma;  // SDs
}

model {
  // priors
  mu ~ normal(0, 5);
  sigma ~ exponential(1);
  
  // likelihood
  y ~ normal(mu[g], sigma[g]);
}

generated quantities {
  // difference in means and SDs
  real delta_mu = mu[2] - mu[1];
  real delta_sigma = sigma[2] - sigma[1];
}
