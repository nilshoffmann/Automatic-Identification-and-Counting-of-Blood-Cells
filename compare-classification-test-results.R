library(tidyverse)
ground_truth <- read_csv("output/ground_truth.csv")
tf_results <- read_csv("output/rbcs-tf-Testing.csv")
onnx_results <- read_csv("output/rbcs-onnx-Testing.csv")

all_results <- bind_rows(ground_truth, tf_results, onnx_results) |> mutate(file_name = tools::file_path_sans_ext(file_name))

n <- ground_truth |> group_by(file_name) |> summarize(n = n()) |> nrow()

all_summaries <- all_results |> 
  group_by(label, type, file_name) |> 
  summarize(
    count = n(),
    center_x_mean = mean(center_x),
    center_x_sd = sd(center_x),
    center_y_mean = mean(center_y),
    center_y_sd = sd(center_y),
  )

all_summaries |> 
    dplyr::select(label, type, file_name, count) |> 
    pivot_wider(names_from = type, values_from = count) |>
    group_by(label, file_name) |>
    summarize(
        e_tf2 = abs(groundTruth - tf2),
        e_onnx = abs(groundTruth - onnx),
    ) |> na.omit() |> 
    ungroup() |>
    group_by(label) |>
    summarize(
        e_label_tf2 = sum(e_tf2)/n,
        e_label_onnx = sum(e_onnx)/n,
    )

all_summaries |> 
    dplyr::select(label, type, file_name, count) |> 
    group_by(label, type) |>
    summarize(
        totalCount = sum(count),
    ) |> 
    pivot_wider(names_from = type, values_from = totalCount) |>
    write_csv("output/rbcs-classification-numbers.csv")

crosstab <- xtabs(count~type + label + file_name, data = all_summaries)

boxPlot <- ggplot() + 
  geom_boxplot(data = all_summaries, aes(x = type, y = count, color = type)) + 
  labs(x = "Type", y = "Count", title = "RBCS Classification Counts") + 
  theme_bw() +
  facet_wrap(~file_name, ncol = 4) +
  theme(legend.position = "bottom")

scatterPlot <- ggplot() + 
  geom_point(data = all_results, aes(x = center_x, y = center_y, color = type)) + 
  labs(x = "Center X", y = "Center Y", title = "RBCS Classification Centers") + 
  theme_bw() +
  facet_wrap(~label, ncol = 3) +
  theme(legend.position = "bottom")

ggsave("output/rbcs-classification-centers.png", plot = scatterPlot, width = 10, height = 10, units = "in")

dm <- dist(select(all_results, c("center_x","center_y")), method = "euclidean")
hclust <- hclust(na.omit(dm), method = "ward.D2")

xtabs <- xtabs(~type, data =all_results)

plot(hclust, hang = -1)
center_tf <- as.matrix(
        select(tf_results, c("center_x","center_y"))
    )
center_onnx <- as.matrix(
        select(onnx_results, c("center_x","center_y"))
    )
cancor <- cancor(
    center_tf, center_onnx
)
cancor