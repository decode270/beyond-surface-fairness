library(tidyverse)
library(readxl)


# =======================
before_df <- read_excel("basline_metrics_resultxxxx.xlsx") %>%
  select(occupation, group, BiasScore_before = BiasScore)

after_df <- read_excel("finetuned_metrics_resultxxxx.xlsx") %>%
  select(occupation, BiasScore_after = BiasScore)

# Merge data
merged_df <- left_join(before_df, after_df, by = "axes, e.g.,occupation") %>%
  filter(!is.na(BiasScore_before), !is.na(BiasScore_after)) %>%
  filter(occupation != "__MEAN__")


# Sort(Baseline Female)
merged_df_female <- merged_df %>% filter(group == "F") %>% arrange(desc(BiasScore_before))
merged_df_male   <- merged_df %>% filter(group == "M") %>% arrange(desc(BiasScore_before))

# orders in left_f, right_m
ordered_occupations <- unique(c(merged_df_female$occupation, merged_df_male$occupation))

# factor order
merged_df <- merged_df %>%
  mutate(occupation = factor(occupation, levels = ordered_occupations))

# =======================
# plots in 3 lines
# =======================
ggplot() +
  # Baseline Model - Female
  geom_line(
    data = merged_df %>% filter(group == "F"),
    aes(x = occupation, y = BiasScore_before, group = 1, color = "Baseline Model"),
    size = 1
  ) +
  geom_point(
    data = merged_df %>% filter(group == "F"),
    aes(x = occupation, y = BiasScore_before, color = "Baseline Model", shape = "Baseline Model"),
    size = 2.5, stroke = 1
  ) +
  
  # Baseline Model - Male
  geom_line(
    data = merged_df %>% filter(group == "M"),
    aes(x = occupation, y = BiasScore_before, group = 1, color = "Baseline Model"),
    size = 1
  ) +
  geom_point(
    data = merged_df %>% filter(group == "M"),
    aes(x = occupation, y = BiasScore_before, color = "Baseline Model", shape = "Baseline Model"),
    size = 2.5, stroke = 1
  ) +
  
  # Fine-tuned Model - Female
  geom_line(
    data = merged_df %>% filter(group == "F"),
    aes(x = occupation, y = BiasScore_after, group = 1, color = "Fine-tuned Model (Female)"),
    size = 1
  ) +
  geom_point(
    data = merged_df %>% filter(group == "F"),
    aes(x = occupation, y = BiasScore_after, color = "Fine-tuned Model (Female)", shape = "Fine-tuned Model (Female)"),
    size = 2.5, stroke = 1
  ) +
  
  # Fine-tuned Model - Male
  geom_line(
    data = merged_df %>% filter(group == "M"),
    aes(x = occupation, y = BiasScore_after, group = 1, color = "Fine-tuned Model (Male)"),
    size = 1
  ) +
  geom_point(
    data = merged_df %>% filter(group == "M"),
    aes(x = occupation, y = BiasScore_after, color = "Fine-tuned Model (Male)", shape = "Fine-tuned Model (Male)"),
    size = 2.5, stroke = 1
  ) +
  
  # separate line
  geom_vline(xintercept = length(merged_df_female$occupation) + 0.5,
             linetype = "dashed", color = "black") +
  
  # settings
  scale_color_manual(
    values = c(
      "Baseline Model" = "gray30",
      "Fine-tuned Model (Female)" = "deeppink",
      "Fine-tuned Model (Male)" = "dodgerblue"
    )
  ) +
  scale_shape_manual(
    values = c(
      "Baseline Model" = 4,
      "Fine-tuned Model (Female)" = 16,
      "Fine-tuned Model (Male)" = 16
    )
  ) +
  
  labs(
    color = "", shape = "",
    title = "Fairness Score by Character and Gender (Llama3.1-8B)",#(Mistral 7B), #(Llama3.1-8B)
    x = NULL, y = "Fairness Score"
  ) +
  
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(
      angle = 90, hjust = 1,
      color = ifelse(levels(merged_df$occupation) %in% merged_df_female$occupation,
                     "deeppink", "dodgerblue")
    ),
    legend.position = "top"
  )

