


library(tensorflow, exclude = c("shape", "set_random_seed"))
library(tfdatasets, exclude = "shape")
library(dplyr, warn.conflicts = FALSE)
library(keras3)
use_backend("jax")


"https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip" |>
  get_file(origin = _) |>
  zip::unzip("jena_climate_2009_2016.csv")


writeLines(readLines("jena_climate_2009_2016.csv", 3))


withr::with_package("readr", {
  full_df <- read_csv(
    "jena_climate_2009_2016.csv",
    locale = locale(tz = "Etc/GMT+1"),
    col_types = cols(
      `Date Time` = col_datetime("%d.%m.%Y %H:%M:%S"),
      .default = col_double()
    )
  )
})


tibble::glimpse(full_df)


plot(`T (degC)` ~ `Date Time`, data = full_df, pch = 20, cex = .3)


plot(`T (degC)` ~ `Date Time`, data = full_df[1:1440, ])


num_train_samples <- round(nrow(full_df) * .5)
num_val_samples <- round(nrow(full_df) * 0.25)
num_test_samples <- nrow(full_df) - num_train_samples - num_val_samples

train_df <- full_df[seq(num_train_samples), ]

val_df <- full_df[seq(from = nrow(train_df) + 1,
                      length.out = num_val_samples), ]

test_df <- full_df[seq(to = nrow(full_df),
                       length.out = num_test_samples), ]

cat("num_train_samples:", nrow(train_df), "\n")
cat("num_val_samples:", nrow(val_df), "\n")
cat("num_test_samples:", nrow(test_df), "\n")


input_data_colnames <- names(full_df) |> setdiff(c("Date Time"))
normalization_values <- train_df[input_data_colnames] |>
  lapply(\(col) list(mean = mean(col), sd = sd(col)))

str(normalization_values)

normalize_input_data <- function(df) {
  purrr::map2(df, normalization_values[names(df)], \(col, nv) {
    (col - nv$mean) / nv$sd
  }) |> as_tibble()
}


int_sequence <- seq(10)
dummy_dataset <- timeseries_dataset_from_array(
  data = head(int_sequence, -3),
  targets = tail(int_sequence, -3),
  sequence_length = 3,
  batch_size = 2
)

dummy_dataset_iterator <- as_array_iterator(dummy_dataset)

repeat {
  batch <- iter_next(dummy_dataset_iterator)
  if (is.null(batch))
    break
  .[inputs, targets] <- batch
  for (r in 1:nrow(inputs))
    cat(sprintf("input: [ %s ]  target: %s\n",
                paste(inputs[r, ], collapse = " "), targets[r]))
  cat(strrep("-", 27), "\n")
}


sampling_rate <- 6
sequence_length <- 120
delay <- sampling_rate * (sequence_length + 24 - 1)
batch_size <- 256

df_to_inputs_and_targets <- function(df) {
  inputs <- df[input_data_colnames] |>
    normalize_input_data() |>
    as.matrix()

  targets <- as.array(df$`T (degC)`)

  list(
    head(inputs, -delay),
    tail(targets, -delay)
  )
}

make_dataset <- function(df) {
  .[inputs, targets] <- df_to_inputs_and_targets(df)

  timeseries_dataset_from_array(
    inputs, targets,
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    shuffle = TRUE,
    batch_size = batch_size
  )
}

train_dataset <- make_dataset(train_df)
val_dataset <- make_dataset(val_df)
test_dataset <- make_dataset(test_df)


.[samples, targets] <- iter_next(as_iterator(train_dataset))
cat("samples shape: ", format(samples$shape), "\n",
    "targets shape: ", format(targets$shape), "\n", sep = "")


evaluate_naive_method <- function(dataset) {

  .[temp_sd = sd, temp_mean = mean] <- normalization_values$`T (degC)`
  unnormalize_temperature <- function(x) {
    (x * temp_sd) + temp_mean
  }

  temp_col_idx <- match("T (degC)", input_data_colnames)

  reduction <- dataset |>
    dataset_unbatch() |>
    dataset_map(function(samples, target) {
      last_temp_in_input <- samples@r[-1, temp_col_idx]                         # <1>
      pred <- unnormalize_temperature(last_temp_in_input)                       # <2>
      abs(pred - target)
    }) |>
    dataset_reduce(
      initial_state = list(total_samples_seen = 0L,
                           total_abs_error = 0),
      reduce_func = function(state, element) {
        `add<-` <- `+`
        add(state$total_samples_seen) <- 1L
        add(state$total_abs_error) <- element
        state
      }
    ) |>
    lapply(as.numeric)                                                          # <3>

  mae <- with(reduction, total_abs_error / total_samples_seen)                  # <4>
  mae
}

sprintf("Validation MAE: %.2f", evaluate_naive_method(val_dataset))
sprintf("Test MAE: %.2f", evaluate_naive_method(test_dataset))


ncol_input_data <- length(input_data_colnames)

inputs <- keras_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs |>
  layer_flatten() |>
  layer_dense(16, activation="relu") |>
  layer_dense(1)

model <- keras_model(inputs, outputs)

callbacks = list(
  callback_model_checkpoint("jena_dense.keras", save_best_only = TRUE)          # <1>
)

model |> compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = "mae"
)

history <- model |> fit(
  train_dataset,
  epochs = 10,
  validation_data = val_dataset,
  callbacks = callbacks
)

model <- load_model("jena_dense.keras")                                         # <2>
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)["mae"])


plot(history, metrics = "mae")


inputs <- keras_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs |>
  layer_conv_1d(8, 24, activation = "relu") |>
  layer_max_pooling_1d(2) |>
  layer_conv_1d(8, 12, activation = "relu") |>
  layer_max_pooling_1d(2) |>
  layer_conv_1d(8, 6, activation = "relu") |>
  layer_global_average_pooling_1d() |>
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks <- list(
  callback_model_checkpoint("jena_conv.keras", save_best_only = TRUE)
)

model |> compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = "mae"
)


history <- model |> fit(
  train_dataset,
  epochs = 10,
  validation_data = val_dataset,
  callbacks = callbacks
)


model <- load_model("jena_conv.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)[["mae"]])


plot(history, metrics = "mae")


inputs <- keras_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs |>
  layer_lstm(16) |>
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks <- list(
  callback_model_checkpoint("jena_lstm.keras", save_best_only = TRUE)
)

compile(model, optimizer = "rmsprop", loss = "mse", metrics = "mae")


history <- model |> fit(
  train_dataset,
  epochs = 10,
  validation_data = val_dataset,
  callbacks = callbacks
)


local({
  p <- plot(history, metrics = "mae")
  p$data %<>% .[.$epoch > 1, ]
  print(p)
})


model <- load_model("jena_dense.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)[["mae"]])


runif_array <- function(dim) array(runif(prod(dim)), dim)

timesteps <- 100                                                                # <1>
input_features <- 32                                                            # <2>
output_features <- 64                                                           # <3>

inputs <- runif_array(c(timesteps, input_features))                             # <4>
state_t <- array(0, dim = output_features)                                      # <5>
W <- runif_array(c(output_features, input_features))                            # <6>
U <- runif_array(c(output_features, output_features))                           # <6>
b <- runif_array(c(output_features, 1))                                         # <6>
outputs <- array(0, dim = c(timesteps, output_features))

for(ts in 1:timesteps) {
  input_t <- inputs[ts, ]                                                       # <7>
  output_t <- tanh( (W %*% input_t) + (U %*% state_t) + b )                     # <8>
  outputs[ts, ] <- state_t <- output_t                                          # <9>
}

final_output_sequence <- outputs                                                # <10>


num_features <- 14
inputs <- keras_input(shape = c(NA, num_features))
outputs <- inputs |> layer_simple_rnn(16)


num_features <- 14
steps <- 120
inputs <- keras_input(shape = c(steps, num_features))
outputs <- inputs |> layer_simple_rnn(16, return_sequences = FALSE)             # <1>
op_shape(outputs)


num_features <- 14
steps <- 120
inputs <- keras_input(shape = c(steps, num_features))
outputs <- inputs |> layer_simple_rnn(16, return_sequences = TRUE)              # <1>
op_shape(outputs)


inputs <- keras_input(shape = c(steps, num_features))
outputs <- inputs |>
  layer_simple_rnn(16, return_sequences = TRUE) |>
  layer_simple_rnn(16, return_sequences = TRUE) |>
  layer_simple_rnn(16)


inputs <- keras_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs |>
  layer_lstm(32, recurrent_dropout = 0.25) |>
  layer_dropout(0.5) |>                                                         # <1>
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks = list(
  callback_model_checkpoint("jena_lstm_dropout.keras", save_best_only = TRUE)
)

compile(model, optimizer = "rmsprop", loss = "mse", metrics = "mae")


history <- model |> fit(
  train_dataset,
  epochs = 50,
  validation_data = val_dataset,
  callbacks = callbacks
)


local({
  p <- plot(history, metrics = "mae")
  p$data %<>% .[.$epoch > 1, ]
  print(p)
})


inputs <- keras_input(shape = c(sequence_length, num_features))                 # <1>
x <- inputs |> layer_lstm(32, recurrent_dropout = 0.2, unroll = TRUE)           # <2>


inputs <- keras_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs |>
  layer_gru(32, recurrent_dropout = 0.5, return_sequences = TRUE) |>
  layer_gru(32, recurrent_dropout = 0.5) |>
  layer_dropout(0.5) |>
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks <- list(
  callback_model_checkpoint(
    "jena_stacked_gru_dropout.keras", save_best_only = TRUE
  )
)

model |> compile(optimizer = "rmsprop", loss = "mse", metrics = "mae")


history <- model |> fit(
  train_dataset,
  epochs = 50,
  validation_data = val_dataset,
  callbacks = callbacks
)


plot(history)


model <- load_model("jena_stacked_gru_dropout.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)[["mae"]])


inputs <- keras_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs |>
  layer_bidirectional(layer_lstm(units = 16)) |>
  layer_dense(1)
model <- keras_model(inputs, outputs)

model |> compile(optimizer = "rmsprop", loss = "mse", metrics = "mae")

history <- model |> fit(
  train_dataset,
  epochs = 10,
  validation_data = val_dataset
)

plot(history)



