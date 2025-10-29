# plot_directionality.R
# Visualize directional bias movement using ln(SHR) before vs after fine-tuning.
# - Baseline and fine-tuned Excel files must each contain columns:
#   group, occupation, SHR   (case/space-insensitive; order doesn't matter)
# - The plot shows per-occupation movement on ln(SHR) from baseline -> fine-tuned.
#   Lines = baseline; circles = after; colors indicate toward/away from fairness.
#   Red circles mark direction flips (crossing 0) or values clipped at the axis cap.

library(readxl)
library(dplyr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

baseline_file  <- if (length(args) >= 1) args[[1]] else Sys.getenv("BASELINE_XLSX", "stories_chara_metrics.xlsx")
finetuned_file <- if (length(args) >= 2) args[[2]] else Sys.getenv("FINETUNED_XLSX", "stories_chara_ft_metrics.xlsx")

baseline_label <- if (length(args) >= 3) args[[3]] else Sys.getenv("BASELINE_LABEL", "Baseline")
finetuned_label<- if (length(args) >= 4) args[[4]] else Sys.getenv("FINETUNED_LABEL","Fine-tuned")

# --------------------------
# Helper: pick required cols
# --------------------------
pick_col <- function(df){
  nm <- names(df)
  norm <- tolower(gsub("\\s+", "", nm))
  map1 <- function(key) {
    hit <- which(norm == key)[1]
    if (is.na(hit)) stop(sprintf("Required column '%s' not found.", key))
    nm[hit]
  }
  df %>%
    select(
      group      = all_of(map1("group")),
      occupation = all_of(map1("occupation")),
      SHR        = all_of(map1("shr"))
    )
}

# --------------------------
# Load and prepare data
# --------------------------
base_raw <- read_excel(baseline_file)
ft_raw   <- read_excel(finetuned_file)

base <- pick_col(base_raw) %>%
  mutate(b_base = log(SHR)) %>%
  select(group, occupation, b_base)

ft <- pick_col(ft_raw) %>%
  mutate(b_ft = log(SHR)) %>%
  select(group, occupation, b_ft)

df <- base %>%
  mutate(occupation = trimws(occupation)) %>%
  inner_join(ft %>% mutate(occupation = trimws(occupation)),
             by = c("group","occupation")) %>%
  group_by(group) %>% mutate(order_in_group = row_number()) %>% ungroup() %>%
  mutate(
    # y-placement: positive order for M, negative for F (purely to separate halves)
    y = ifelse(group == "M", order_in_group, -order_in_group),
    
    # direction flip across 0 on ln(SHR)
    cross0   = (b_base < 0 & b_ft > 0) | (b_base > 0 & b_ft < 0),
    
    # toward vs away from fairness (0) on magnitude
    to_zero  = abs(b_ft) < abs(b_base),
    seg_col  = ifelse(to_zero, "toward", "away"),
    
    # end marker fill
    end_fill = ifelse(cross0, "red", "white")
  )

# Axis cap using the 98th percentile of absolute ln(SHR), with a minimum span
L <- max(3, quantile(abs(c(df$b_base, df$b_ft)), 0.98, na.rm = TRUE))
df <- df %>%
  mutate(
    b_base_plot = pmax(pmin(b_base,  L), -L),
    b_ft_plot   = pmax(pmin(b_ft,    L), -L),
    clipped_end = abs(b_ft) > L
  )

# Neutral band width (tau) around 0 in ln-space (≈ ±ln(1.2))
tau <- log(1.2)

x_min <- -L; x_max <- L
cols_segments <- c(toward = "#000000", away = "#ff7f0e")

p <- ggplot(df) +
  # Quadrant shading to hint group-preference zones (optional aesthetic)
  annotate("rect", xmin = 0,     xmax = x_max, ymin = 0,     ymax = Inf,  fill = "#cfe8ff", alpha = .18) +
  annotate("rect", xmin = x_min, xmax = 0,     ymin = -Inf,  ymax = 0,    fill = "#cfe8ff", alpha = .18) +
  annotate("rect", xmin = x_min, xmax = 0,     ymin = 0,     ymax = Inf,  fill = "#ffd6e7", alpha = .18) +
  annotate("rect", xmin = 0,     xmax = x_max, ymin = -Inf,  ymax = 0,    fill = "#ffd6e7", alpha = .18) +
  
  # Neutral band around 0
  annotate("rect", xmin = -tau, xmax =  tau, ymin = -Inf, ymax = Inf, fill = "grey70", alpha = .08) +
  
  geom_hline(yintercept = 0, color = "steelblue", linewidth = 1) +
  geom_vline(xintercept = 0, color = "steelblue", linewidth = 1) +
  geom_vline(xintercept = setdiff(seq(floor(x_min), ceiling(x_max), 1), 0),
             linetype = "dotted", linewidth = .3, color = "grey80") +
  
  # Movement segments: baseline -> fine-tuned
  geom_segment(aes(x = b_base_plot, xend = b_ft_plot, y = y, yend = y, color = seg_col),
               linewidth = 1.1, lineend = "round") +
  scale_color_manual(values = cols_segments, name = NULL,
                     breaks = c("toward","away"),
                     labels = c("Towards Fairness ↓",
                                "Away from Fairness ↑")) +
  
  # Endpoints: fine-tuned status
  geom_point(aes(x = b_ft_plot, y = y,
                 fill = ifelse(cross0 | clipped_end, "red", "white")),
             shape = 21, size = 2.8, stroke = 0.9, color = "black") +
  scale_fill_manual(values = c(white = "white", red = "#d62728"),
                    name = NULL,
                    breaks = c("white","red"),
                    labels = c("Non-directional",
                               "Directional / Clipped")) +
  
  # Occupation labels on the right margin
  geom_text(aes(x = x_max + 0.2, y = y, label = occupation), hjust = 0, size = 3.1) +
  coord_cartesian(xlim = c(x_min, x_max), clip = "off") +
  scale_y_continuous(name = "Axis: Character (top = M, bottom = F)", breaks = NULL) +
  
  labs(
    x = "ln(SHR)",
    title = sprintf("%s → %s", baseline_label, finetuned_label),
    subtitle = "Circle = fine-tuned; Line = baseline • Top half=M group, Bottom half=F group\nBlue vertical band: neutral zone around 0 (ln-scale)."
  ) +
  
  theme_minimal(base_size = 11) +
  theme(
    plot.margin = margin(10, 90, 10, 10),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

print(p)
