functions {
  /** 
   * Return the natural logarithm of the product of the 
   * elementwise exponentiation of a matrix and vector
   *
   * @param A  First matrix
   * @param B  Second vector
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
  array[I] int<lower=1, upper=J> f;  // first capture
  array[I] int<lower=f, upper=J> l;  // last capture
  array[I + K, J] int<lower=1, upper=2> y;   // detection history
}

parameters {
  real<lower=0, upper=1> gamma, phi, p;
}

model {
  // priors
  target += beta_lupdf(gamma | 1, 10);
  target += beta_lupdf(phi | 1, 1);
  target += beta_lupdf(p | 1, 1);
  
  // pre-compute log probabilities
  real log_gamma = log(gamma), log1m_gamma = log1m(gamma);
  real log_phi = log(phi), log1m_phi = log1m(phi);
  real log_p = log(p), log1m_p = log1m(p);
  
  // log TPMs (per individual and survey)
  array[I + K, J] matrix[3, 3] P_z;
  array[I + K, J] matrix[3, 2] P_y;
  for (i in 1:(I + K)) {
    for (j in 1:J) {
      P_z[i, j] = [[ 1 - gamma, gamma, 0 ],
                   [ 0, phi, 1 - phi ],
                   [ 0, 0, 1 ]]';
      P_y[i, j] = [[ 1, 0 ], 
                   [ 1 - p, p ], 
                   [ 1, 0 ]];
    }
  }
  
  // likelihood
  matrix[3, J] Omega;
  
  // initialise as not yet entered
  Omega[:, 1] = [ 1, 0, 0 ]';
  
  // observed individuals
  for (i in 1:I) {
    
    // not yet entered or alive up to first capture
    for (j in 2:f[i]) {
      Omega[1:2, j] = P_z[i, j][1:2, 1:2] * Omega[1:2, j - 1] .* P_y[i, j][1:2, y[i, j]];
    }
    
    // alive up to last capture
    for (j in (f[i] + 1):l[i]) {
      Omega[2, j] = P_z[i, j][2, 2] * Omega[2, j - 1] * P_y[i, j][2, y[i, j]];
    }
    Omega[3, l[i]] = 0;
    
    // alive or dead after last capture
    for (j in (l[i] + 1):J) {
      Omega[2:3, j] = P_z[i, j][2:3, 2:3] * Omega[2:3, j - 1] .* P_y[i, j][2:3, y[i, j]];
    }
    
    // increment log density
    target += log(sum(Omega[2:3, J]));
  }
  
  // dummy individuals are uncertain throughout
  for (i in (I + 1):K) {
    for (j in 2:J) {
      Omega[:, j] = P_z[i, j] * Omega[:, j - 1] .* P_y[i, j][:, y[i, j]];
    }
    target += log(sum(Omega[:, J]));
  }
}

// generated quantities {
//   vector[I] log_lik;
//   array[I, J] int z;
//   {
//     // log TPMs
//     array[I, J] matrix[2, 2] log_P_z, log_P_y;
//     for (i in 1:I) {
//       for (j in (f[i] + 1):J) {
//         log_P_z[i, j] = [[ 0, log1m(phi) ], 
//                          [ negative_infinity(), log(phi) ]];
//         log_P_y[i, j] = [[ 0, negative_infinity() ], 
//                          [ log1m(p), log(p) ]];
//       }
//     }
//     
//     // log-likelihood (mirrors model{} block)
//     matrix[2, J] Omega;
//     for (i in 1:I) {
//       Omega[2, f[i]] = 0;
//       for (j in (f[i] + 1):l[i]) {
//         Omega[2, j] = log_P_z[i, j][2, 2] + Omega[2, j - 1]
//                       + log_P_y[i, j][2, y[i, j]];
//       }
//       Omega[1, l[i]] = negative_infinity();
//       for (j in (l[i] + 1):J) {
//         Omega[:, j] = log_prod_exp(log_P_z[i, j], Omega[:, j - 1])
//                       + log_P_y[i, j][:, y[i, j]];
//       }
//       log_lik[i] = log_sum_exp(Omega[:, J]);
//     }
//     
//     // Viterbi algorithm for latent states (0 = dead, 1 = alive)
//     vector[2] lp;
//     matrix[2, J] best_lp;
//     array[2, J] int back_pointer;
//     for (i in 1:I) {
//       z[i, f[i]:l[i]] = ones_int_array(l[i] - f[i] + 1);
//       best_lp[:, l[i]] = [ negative_infinity(), 0 ]';
//       for (j in (l[i] + 1):J) {
//         for (a in 1:2) {
//           lp = log_P_z[i, j][a]' + best_lp[:, j - 1]
//                + log_P_y[i, j][:, 1];
//           back_pointer[a, j] = sort_indices_desc(lp)[1] - 1;
//           best_lp[a, j] = max(lp);
//         }
//       }
//       z[i, J] = sort_indices_desc(best_lp[:, J])[1] - 1;
//       for (j in (l[i] + 1):(J - 1)) {
//         int jj = J + l[i] - j;  // reverse the indexing
//         z[i, jj] = back_pointer[z[i, jj + 1] + 1, jj + 1];
//       }
//     }
//   }
// }
