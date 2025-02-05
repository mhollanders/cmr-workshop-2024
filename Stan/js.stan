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
}

data {
  int<lower=1> I, J, K;  // number of individuals, surveys, and augmented individuals
  array[I] int<lower=2, upper=J> f;  // first capture
  array[I] int<lower=f, upper=J> l;  // last capture
  array[I + K, J] int<lower=1, upper=2> y;   // detection history
}

parameters {
  real<lower=0, upper=1> phi, p;
  vector<lower=0, upper=1>[J - 1] gamma;
}

model {
  // priors
  target += beta_lupdf(gamma | 1, 10);
  target += beta_lupdf(phi | 1, 1);
  target += beta_lupdf(p | 1, 1);
  
  // pre-compute log probabilities
  real log_phi = log(phi), log1m_phi = log1m(phi);
  real log_p = log(p), log1m_p = log1m(p);
  vector[J - 1] log_gamma = log(gamma), log1m_gamma = log1m(gamma);
  
  // log TPMs (per individual and survey)
  array[I + K, J - 1] matrix[3, 3] log_P_z;
  array[I + K, J - 1] matrix[3, 2] log_P_y;
  for (i in 1:(I + K)) {
    for (j in 1:(J - 1)) {
      log_P_z[i, j] = [[ log1m_gamma[j], log_gamma[j], negative_infinity() ],
                       [ negative_infinity(), log_phi, log1m_phi ],
                       [ negative_infinity(), negative_infinity(), 0 ]]';
      log_P_y[i, j] = [[ 0, negative_infinity() ], 
                       [ log1m_p, log_p ], 
                       [ 0, negative_infinity() ]];
    }
  }
  
  // initialise as not yet entered
  matrix[3, J] Omega;
  Omega[:, 1] = [ 0, negative_infinity(), negative_infinity() ]';
  
  // forward algorithm
  for (i in 1:I) {
    
    // not yet entered or alive up to first capture
    for (j in 2:f[i]) {
      Omega[1:2, j] = log_prod_exp(log_P_z[i, j - 1][1:2, 1:2], Omega[1:2, j - 1]) 
                      + log_P_y[i, j - 1][1:2, y[i, j]];
    }
    
    // alive up to last capture
    for (j in (f[i] + 1):l[i]) {
      Omega[2, j] = log_P_z[i, j - 1][2, 2] + Omega[2, j - 1] 
                    + log_P_y[i, j - 1][2, y[i, j]];
    }
    Omega[3, l[i]] = negative_infinity();
    
    // alive or dead after last capture
    for (j in (l[i] + 1):J) {
      Omega[2:3, j] = log_prod_exp(log_P_z[i, j - 1][2:3, 2:3], Omega[2:3, j - 1])
                      + log_P_y[i, j - 1][2:3, 1];
    }
    target += log_sum_exp(Omega[2:3, J]);
  }
  
  // dummy individuals are uncertain throughout
  for (i in (I + 1):K) {

    // can't transition from not yet entered to dead
    Omega[1:2, 2] = log_prod_exp(log_P_z[i, 1][1:2, 1:2], Omega[1:2, 1])
                    + log_P_y[i, 1][1:2, 1];

    // subsequent surveys
    for (j in 3:J) {
      Omega[:, j] = log_prod_exp(log_P_z[i, j - 1], Omega[:, j - 1])
                    + log_P_y[i, j - 1][:, 1];
    }
    target += log_sum_exp(Omega[:, J]);
  }
}

generated quantities {
  vector[I] log_lik;  // only for observed individuals
  array[I + K, J] int z;  // latent states (1 = not yet entered, 2 = alive, 3 = dead)
  array[J] int N = zeros_int_array(J);
  {
    // parameters
    real log_phi = log(phi), log1m_phi = log1m(phi);
    real log_p = log(p), log1m_p = log1m(p);
    vector[J - 1] log_gamma = log(gamma), log1m_gamma = log1m(gamma);
    array[I + K, J - 1] matrix[3, 3] log_P_z;
    array[I + K, J - 1] matrix[3, 2] log_P_y;
    for (i in 1:(I + K)) {
      for (j in 1:(J - 1)) {
        log_P_z[i, j] = [[ log1m_gamma[j], log_gamma[j], negative_infinity() ],
                         [ negative_infinity(), log_phi, log1m_phi ],
                         [ negative_infinity(), negative_infinity(), 0 ]]';
        log_P_y[i, j] = [[ 0, negative_infinity() ],
                         [ log1m_p, log_p ],
                         [ 0, negative_infinity() ]];
      }
    }
    
    // log-likelihood (mirrors model block)
    matrix[3, J] Omega;
    Omega[:, 1] = [ 0, negative_infinity(), negative_infinity() ]';
    for (i in 1:I) {
      for (j in 2:f[i]) {
        Omega[1:2, j] = log_prod_exp(log_P_z[i, j - 1][1:2, 1:2], Omega[1:2, j - 1]) 
                        + log_P_y[i, j - 1][1:2, y[i, j]];
      }
      for (j in (f[i] + 1):l[i]) {
        Omega[2, j] = log_P_z[i, j - 1][2, 2] + Omega[2, j - 1] 
                      + log_P_y[i, j - 1][2, y[i, j]];
      }
      Omega[3, l[i]] = negative_infinity();
      for (j in (l[i] + 1):J) {
        Omega[2:3, j] = log_prod_exp(log_P_z[i, j - 1][2:3, 2:3], Omega[2:3, j - 1])
                        + log_P_y[i, j - 1][2:3, y[i, j]];
      }
      log_lik[i] = log_sum_exp(Omega[2:3, J]);
    }

    // Viterbi algorithm for latent states
    vector[3] lp;
    matrix[3, J] best_lp;
    array[3, J] int back_pointer;
    z[:, 1] = ones_int_array(I + K);
    best_lp[:, 1] = Omega[:, 1];
    
    // observed individuals
    for (i in 1:I) {
      for (j in 2:f[i]) {
        for (a in 1:2) {
          lp[1:2] = log_P_z[i, j - 1][a, 1:2]' + best_lp[1:2, j - 1]
                    + log_P_y[i, j - 1][1:2, y[i, j]];
          back_pointer[a, j] = sort_indices_desc(lp[1:2])[1];
          best_lp[a, j] = max(lp[1:2]);
        }
      }
      best_lp[2:3, l[i]] = [0, negative_infinity() ]';
      for (j in (l[i] + 1):J) {
        for (a in 2:3) {
          lp[2:3] = log_P_z[i, j - 1][a, 2:3]' + best_lp[2:3, j - 1]
                    + log_P_y[i, j - 1][2:3, y[i, j]];
          back_pointer[a, j] = sort_indices_desc(lp[2:3])[1] + 1;
          best_lp[a, j] = max(lp[2:3]);
        }
      }
      z[i, J] = sort_indices_desc(best_lp[2:3, J])[1] + 1;
      for (j in (l[i] + 1):(J - 1)) {
        int jj = J + l[i] - j;  // reverse the indexing
        z[i, jj] = back_pointer[z[i, jj + 1], jj + 1];
      }
      z[i, f[i]:l[i]] = rep_array(2, l[i] - f[i] + 1);
      for (j in 2:(f[i] - 1)) {
        int jj = f[i] + 1 - j;
        z[i, jj] = back_pointer[z[i, jj + 1], jj + 1];
      }
    }
    
    // dummy individuals
    for (i in (I + 1):K) {
      for (j in 2:J) {
        for (a in 1:3) {
          lp = log_P_z[i, j - 1][a]' + best_lp[:, j - 1]
               + log_P_y[i, j - 1][:, 1];
          back_pointer[a, j] = sort_indices_desc(lp)[1];
          best_lp[a, j] = max(lp);
        }
      }
      z[i, J] = sort_indices_desc(best_lp[:, J])[1];
      for (j in 2:(J - 1)) {
        int jj = J + 1 - j;
        z[i, jj] = back_pointer[z[i, jj + 1], jj + 1];
      }
    }
  }
  
  // population sizes
  array[I + K] int alive;
  for (j in 2:J) {
    for (i in 1:(I + K)) {
      alive[i] = z[i, j] == 1;
    }
    N[j] = sum(alive);
  }
}
