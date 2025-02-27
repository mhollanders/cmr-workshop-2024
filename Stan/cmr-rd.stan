functions {
  /** 
   * Return the natural logarithm of the product of the elementwise 
   * exponentiation of a matrix and vector (log matrix multiplication)
   *
   * @param A  Matrix
   * @param B  Vector
   *
   * @return   log(exp(A) * exp(B))
   */
  vector log_prod_exp(matrix A, vector B) {
    int I = rows(A);
    int J = cols(A);
    matrix[J, I] A_tr = A';
    vector[I] C;
    for (i in 1:I) {
      C[i] = log_sum_exp(A_tr[:, i] + B);
    }
    return C;
  }
  real log_prod_exp(row_vector A, vector B) {
    real C = log_sum_exp(A' + B);
    return C;
  }
  
  /** 
   * Normalise probabilities on log scale
   *
   * @param A  Vector
   * @param B  Vector
   *
   * @return   A - log(sum(exp))
   */
  vector normalise_log(vector A) {
    return A - log_sum_exp(A);
  }
}

data {
  int<lower=1> I, J, K_max;  // number of individuals, primaries, and max secondaries
  array[I] int<lower=1, upper=J> f;  // primary of first capture
  array[I] int<lower=f, upper=J> l;  // primary last capture
  array[J] int<lower=1, upper=K_max> K;  // number of secondaries per primary
  vector<lower=0>[J] tau;  // time intervals between primaries
  array[I, J, K_max] int<lower=1, upper=2> y;   // detection history
  int<lower=0> I_aug;  // number of augmented individuals
}

transformed data {
  int JS = I_aug > 0;  // Jolly-Seber indicator
  array[I] int f_k = zeros_int_array(I);  // secondary of first capture
  for (i in 1:I) {
    for (k in 1:K[f[i]]) {
      if (y[i, f[i], k] == 2) {
        while (f_k[i] == 0) {
          f_k[i] = k;
        }
      }
    }
  }
  tuple(real, vector[J]) gamma_prior;  // beta prior shapes for removal entry
  gamma_prior.1 = 1.0 / J;
  gamma_prior.2 = 2 - linspaced_vector(J, 1, J) / J;
}

parameters {
  real l_phi_a, l_p_a;  // survival and detection log odds intercepts
  vector<lower=0, upper=1>[JS * J] gamma;  // removal entry probabilities if Jolly-Seber
}

transformed parameters {
  real phi_a = inv_logit(l_phi_a), p_a = inv_logit(l_p_a);  // survival and detection probability intercepts
}

model {
  // priors
  target += logistic_lupdf(l_phi_a | 0, 1);
  target += logistic_lupdf(l_p_a | 0, 1);
  if (JS) {
    target += beta_lupdf(gamma | gamma_prior.1, gamma_prior.2);  // Dorazio 2020 prior
  }
  
  // pre-compute log probabilities
  vector[J] phi = pow(phi_a, tau);
  vector[J] log_phi = log(phi), log1m_phi = log1m(phi);
  real log_p = log(p_a), log1m_p = log1m(p_a);
  vector[J] log_gamma, log1m_gamma;
  if (JS) {
    log_gamma = log(gamma);
    log1m_gamma = log1m(gamma);
  }
  
  // state-specific log probabilities in forward algorithm
  vector[3] Omega;
  
  // for each individual 
  for (i in 1:(I + I_aug)) {
    
    // log TPMs
    array[J] matrix[3, 3] log_P_z;
    array[J, K_max] matrix[3, 2] log_P_y;
    for (j in 1:J) {
      log_P_z[j] = [[ log1m_gamma[j], log_gamma[j], negative_infinity() ],
                    [ negative_infinity(), log_phi[j], log1m_phi[j] ],
                    [ negative_infinity(), negative_infinity(), 0 ]];
      for (k in 1:K[j]) {
        log_P_y[j, k] = [[ 0, negative_infinity() ],
                         [ log1m_p, log_p ],
                         [ 0, negative_infinity() ]];
      }
    }
    
    // initialise for Jolly-Seber
    Omega = [ 0, negative_infinity(), negative_infinity() ]';
    
    // forward algorithm for observed individuals
    if (i <= I) {
      
      // only increment entry process if JS
      if (JS) {
        for (j in 1:f[i]) {
          Omega[1:2] = log_prod_exp(log_P_z[j][1:2, 1:2]', Omega[1:2]);
          for (k in 1:K[j]) {
            Omega[1:2] += log_P_y[j, k][1:2, y[i, j, k]];
          }
        }
        // increment detection probabilities of first primary if CJS
      } else {
        Omega[2] = 0;
        for (k in 1:K[f[i]]) {
          if (k != f_k[i]) {
            Omega[2] += log_P_y[f[i], k][2, y[i, f[i], k]];
          }
        }
      }
      
      // alive from first to last detection
      for (j in (f[i] + 1):l[i]) {
        Omega[2] += log_P_z[j][2, 2];
        for (k in 1:K[j]) {
          Omega[2] += log_P_y[j, k][2, y[i, j, k]];
        }
      }
      
      // alive or dead after last capture
      for (j in (l[i] + 1):J) {
        Omega[2:3] = log_prod_exp(log_P_z[j][2:3, 2:3]', Omega[2:3]);
        for (k in 1:K[j]) {
          Omega[2:3] += log_P_y[j, k][2:3, 1];
        }
      }
      
      // increment log density with alive or dead state
      target += log_sum_exp(Omega[2:3]);
      
      // forward algorithm for augmented individuals
    } else {
      // see https://github.com/stan-dev/math/issues/2494
      Omega[1:2] = log_P_z[1][1, 1:2]';
      for (k in 1:K[1]) {
        Omega[1:2] += log_P_y[1, k][1:2, 1];
      }
      for (j in 2:J) {
        Omega = log_prod_exp(log_P_z[j]', Omega);
        for (k in 1:K[j]) {
          Omega += log_P_y[j, k][:, 1];
        }
      }
      target += log_sum_exp(Omega);
    }
  }
}

generated quantities {
  vector[I] log_lik;  // only for observed individuals
  array[I + I_aug, J] int z;  // latent states (1 = not yet entered, 2 = alive, 3 = dead)
  array[J] int N = zeros_int_array(J);  // population sizes per primary
  int N_super = I;  // superpopulation (initialised as observed individuals)
  {
    // parameters
    vector[J] phi = pow(phi_a, tau);
    vector[J] log_phi = log(phi), log1m_phi = log1m(phi);
    real log_p = log(p_a), log1m_p = log1m(p_a);
    vector[J] log_gamma, log1m_gamma;
    if (JS) {
      log_gamma = log(gamma);
      log1m_gamma = log1m(gamma);
    }
    matrix[3, J + 1] Omega;  // matrix to store intermediate values
    for (i in 1:(I + I_aug)) {
      array[J] matrix[3, 3] log_P_z;
      array[J, K_max] matrix[3, 2] log_P_y;
      for (j in 1:J) {
        log_P_z[j] = [[ log1m_gamma[j], log_gamma[j], negative_infinity() ],
                      [ negative_infinity(), log_phi[j], log1m_phi[j] ],
                      [ negative_infinity(), negative_infinity(), 0 ]];
        for (k in 1:K[j]) {
          log_P_y[j, k] = [[ 0, negative_infinity() ],
                           [ log1m_p, log_p ],
                           [ 0, negative_infinity() ]];
        }
      }
      
      // forward algorithm for log-likelihood (mirrors model block)
      Omega[:, 1] = [ 0, negative_infinity(), negative_infinity() ]';
      if (i <= I) {
        if (JS) {
          for (j in 1:f[i]) {
            Omega[1:2, j + 1] = log_prod_exp(log_P_z[j][1:2, 1:2]', Omega[1:2, j]);
            for (k in 1:K[j]) {
              Omega[1:2, j + 1] += log_P_y[j, k][1:2, y[i, j, k]];
            }
          }
        } else {
          Omega[1:2, f[i] + 1] = [ negative_infinity(), 0 ]';
          for (k in 1:K[f[i]]) {
            if (k != f_k[i]) {
              Omega[2, f[i] + 1] += log_P_y[f[i], k][2, y[i, f[i], k]];
            }
          }
        }
        Omega[3, f[i] + 1] = negative_infinity();
        for (j in (f[i] + 1):J) {
          Omega[2:3, j + 1] = log_prod_exp(log_P_z[j][2:3, 2:3]', Omega[2:3, j]);
          for (k in 1:K[j]) {
            Omega[2:3, j + 1] += log_P_y[j, k][2:3, y[i, j, k]];
          }
        }
        log_lik[i] = log_sum_exp(Omega[2:3, J + 1]);
        
        // backward sampling algorithm for latent states
        z[i, J] = categorical_rng(exp(normalise_log(Omega[2:3, J + 1]))) + 1;
        for (j in (l[i] + 1):(J - 1)) {
          int jj = J + l[i] - j;
          z[i, jj] = categorical_rng(exp(normalise_log(Omega[2:3, jj + 1] + log_P_z[jj + 1][2:3, z[i, jj + 1]]))) + 1;
        }
        z[i, f[i]:l[i]] = rep_array(2, l[i] - f[i] + 1);
        if (JS) {
          for (j in 1:(f[i] - 1)) {
            int jj = f[i] - j;
            z[i, jj] = categorical_rng(exp(normalise_log(Omega[:, jj + 1] + log_P_z[jj + 1][:, z[i, jj + 1]])));
          }
        }
        
        // dummy individuals
      } else {
        for (j in 1:J) {
          Omega[:, j + 1] = log_prod_exp(log_P_z[j]', Omega[:, j]);
          for (k in 1:K[j]) {
            Omega[:, j + 1] += log_P_y[j, k][:, 1];
          }
        }
        z[i, J] = categorical_rng(exp(normalise_log(Omega[:, J + 1])));
        for (j in 1:(J - 1)) {
          int jj = J - j;
          z[i, jj] = categorical_rng(exp(normalise_log(Omega[:, jj + 1] + log_P_z[jj + 1][:, z[i, jj + 1]])));
        }

        // increment superpopulation if ever alive
        N_super += sum(z[i, 1:J]) > J;
      }
      if (JS && N_super == I + I_aug) {
        print("N_super == I + I_aug. Increase I_aug and try again.");
      }
      
      // increment population sizes
      for (j in (JS ? 1 : f[i]):J) {
        N[j] += z[i, j] == 2;
      }
    }
  }
}
