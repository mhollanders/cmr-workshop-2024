# custom ggplot theme
my_theme <- function(base_size = 10,
                     base_family = "", 
                     base_line_size = base_size / 20 / 2, 
                     base_rect_size = base_size / 20) {
  # set colour and line size
  my_black <- "#333333"
  half_line <- base_size / 2
  # theme
  theme_grey(base_size = base_size,
             base_family = base_family, 
             base_line_size = base_line_size,
             base_rect_size = base_rect_size) %+replace%
    # updates
    theme(axis.line = element_blank(), # element_line(colour = my_black),
          axis.text = element_text(colour = my_black, size = rel(0.9)),
          axis.title = element_text(colour = my_black, size = rel(1)),
          axis.ticks =  element_line(colour = my_black),
          legend.key = element_rect(fill = "white", colour = NA),
          legend.text = element_text(size = rel(0.9)),
          panel.background = element_rect(fill = NA, colour = NA),
          panel.border = element_rect(fill = NA, colour = my_black),
          panel.grid = element_blank(),
          plot.margin = margin(10, 10, 10, 10),
          plot.title = element_text(size = rel(1.1), hjust = 0, vjust = 1, margin = margin(b = half_line)),
          strip.background = element_rect(fill = my_black, colour = my_black, linewidth = base_size / 2), 
          strip.text = element_text(colour = "white", size = rel(0.9), margin = margin(rep(0.8 * half_line, 4))),
          complete = TRUE)
}
