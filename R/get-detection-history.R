library(tidyverse)
get_dh <- function(detections, 
                   dates, 
                   date = "date",
                   individual = "individual", 
                   survey = "survey", 
                   covariates = NULL) {
  # create survey column
  dates <- dates |> 
    mutate(factor(date) |> as.numeric())
  
  # join datasets
  dat <- detections |> 
    select(all_of(c(date, individual, covariates))) |> 
    left_join(dates, by = date) |> 
    drop_na() |> 
    mutate(ind = factor(get(individual))) |> 
    arrange(ind)
  
  # get metadata
  I <- nlevels(dat$ind)
  first_last <- dat |> 
    summarise(first = min(survey), 
              last = max(survey),
              .by = ind)
  f <- first_last$first
  l <- first_last$last
  
  # fill detection history
  y <- matrix(0, I, J)
  for (i in 1:I) {
    dat_i <- filter(dat, as.numeric(ind) == i)
    for (j in f[i]:J) {
      y[i, j] <- dat_i |> 
        filter(survey == j) |>
        nrow()
    }
  }
  list(I = I,
       J = J, 
       f = f, 
       l = l,
       y = y)
}
