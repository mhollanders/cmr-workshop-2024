data {
  int<lower=1> I, J;  // number of individuals and surveys
  array[I] int<lower=1, upper=J> f;  // first capture
  array[I, J] int<lower=1, upper=2> y;   // detection history
}

parameters {
  real<lower=0, upper=1> phi, p;
}

transformed parameters {
  // ecological TPM (left stochastic)
  matrix[2, 2] P_z = [[ 1, 0 ], 
                      [ 1 - phi, phi ]]';
                          
  // observation TPM (right stochastic)
  matrix[2, 2] P_y = [[ 1, 0 ], 
                      [ 1 - p, p ]];
}

model {
  // priors
  target += beta_lupdf(phi | 1, 1); // phi ~ beta(1, 1)
  target += beta_lupdf(p | 1, 1);  // p ~ beta(1, 1)
  
  // likelihood
  matrix[2, J] Omega;  // marginal state-specific probabilities
  for (i in 1:I) {
    Omega[:, f[i]] = [ 0, 1 ]';
    for (j in (f[i] + 1):J) {
      Omega[:, j] = P_z * Omega[:, j - 1] .* P_y[:, y[i, j]];
    }
    target += log(sum(Omega[:, J]));
  }
}

generated quantities {
  vector[I] log_lik;
  {
    matrix[2, J] Omega;
    for (i in 1:I) {
      Omega[:, f[i]] = [ 0, 1 ]';
      for (j in (f[i] + 1):J) {
        Omega[:, j] = P_z * Omega[:, j - 1] .* P_y[:, y[i, j]];
      }
      log_lik[i] = log(sum(Omega[:, J]));
    }
  }
}