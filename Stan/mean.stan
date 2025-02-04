data {
  int<lower=0> I;  // number of observations
  vector[I] y;  // outcome vector
}

parameters {
  real mu;  // mean
  real<lower=0> sigma;  // SD
}

model {
  // priors
  mu ~ normal(0, 5);
  sigma ~ exponential(1);
  
  // likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  real entropy = 0.5 * log(2 * pi() * e() * square(sigma));
}
