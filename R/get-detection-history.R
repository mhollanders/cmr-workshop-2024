get_dh <- function(detections, 
                   dates, 
                   date = "date", 
                   id = "id", 
                   jolly_seber = 0, 
                   K = 500, 
                   covariates = NULL) {
  
  stopifnot(
    "date and/or id labels not found in `detections`." = 
      all(c(date, id) %in% colnames(detections))
  )
  
  # create survey column
  dates <- dates |> 
    mutate(survey = factor(!!sym(date)) |> as.numeric())
  
  # join datasets
  dat <- detections |> 
    select(contains(c(date, id, covariates))) |> 
    left_join(dates, by = date) |> 
    drop_na() |> 
    mutate(individual = factor(!!sym(id))) |> 
    arrange(individual)
  
  # get metadata
  I <- nlevels(dat$individual)
  J <- max(dat$survey)
  first_last <- dat |> 
    summarise(first = min(survey), 
              last = max(survey),
              .by = individual)
  f <- first_last$first
  l <- first_last$last
  tau <- dates |> 
    mutate(tau = difftime(date, lag(date), units = "weeks")) |> 
    pull(tau) |> 
    as.numeric()
  tau[1] <- 1  # for JS
  
  # fill detection history
  y <- matrix(0, I, J)
  for (i in 1:I) {
    dat_i <- filter(dat, as.numeric(individual) == i)
    for (j in f[i]:l[i]) {
      y[i, j] <- dat_i |> 
        filter(survey == j) |>
        nrow()
    }
  }
  
  # return Jolly-Seber data
  if (jolly_seber) {
    list(I = I,
         J = J + 1, 
         K = K, 
         f = f + 1, 
         l = l + 1, 
         tau = tau / 52,  # yearly unit by default
         y = cbind(rep(1, I), y + 1) |> 
           rbind(matrix(1, K, J + 1)))
    
    # return Cormack-Jolly-Seber data
  } else {
    list(I = I,
         J = J, 
         f = f, 
         l = l,
         tau = tau[-1] / 52,  # yearly unit by default
         y = y + 1)
  }
}
