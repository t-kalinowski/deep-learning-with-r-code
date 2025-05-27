## ----setup, include=FALSE-------------------------------------------------
library(keras3)
tensorflow::as_tensor(1)


## ---- eval = FALSE--------------------------------------------------------
## url <-
##   "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"
## download.file(url, destfile = basename(url))
## zip::unzip(zipfile = "jena_climate_2009_2016.csv.zip",
##            files = "jena_climate_2009_2016.csv")


## ---- results = 'hide'----------------------------------------------------
full_df <- readr::read_csv("jena_climate_2009_2016.csv")


## ---- results = 'asis', echo = FALSE--------------------------------------
writeLines(format(full_df))


## -------------------------------------------------------------------------
full_df$`Date Time` %<>%
  as.POSIXct(tz = "Etc/GMT+1", format = "%d.%m.%Y %H:%M:%S")


## ---- eval = FALSE--------------------------------------------------------
## full_df$`Date Time` <- full_df$`Date Time` %>%
##   as.POSIXct(tz = "Etc/GMT+1", format = "%d.%m.%Y %H:%M:%S")


## -------------------------------------------------------------------------
plot(`T (degC)` ~ `Date Time`, data = full_df, pch = 20, cex = .3)


## -------------------------------------------------------------------------
plot(`T (degC)` ~ `Date Time`, data = full_df[1:1440, ])


## -------------------------------------------------------------------------
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


## -------------------------------------------------------------------------
input_data_colnames <- names(full_df) %>%
  setdiff(c("Date Time"))

normalization_values <-
  zip_lists(mean = lapply(train_df[input_data_colnames], mean),
            sd = lapply(train_df[input_data_colnames], sd))

str(normalization_values)
normalize_input_data <- function(df) {
  normalize <- function(x, center, scale)
    (x - center) / scale

  for(col_nm in input_data_colnames) {
    col_nv <- normalization_values[[col_nm]]
    df[[col_nm]] %<>% normalize(., col_nv$mean, col_nv$sd)
  }

  df
}


## -------------------------------------------------------------------------
library(keras3)
int_sequence <- seq(10)
dummy_dataset <- timeseries_dataset_from_array(
  data = head(int_sequence, -3),
  targets = tail(int_sequence, -3),
  sequence_length = 3,
  batch_size = 2
)

library(tfdatasets, exclude = "shape")
dummy_dataset_iterator <- as_array_iterator(dummy_dataset)

repeat {
  batch <- iter_next(dummy_dataset_iterator)
  if (is.null(batch))
    break
  c(inputs, targets) %<-% batch
  for (r in 1:nrow(inputs))
    cat(sprintf("input: [ %s ]  target: %s\n",
                paste(inputs[r, ], collapse = " "), targets[r]))
  cat(strrep("-", 27), "\n")
}


## -------------------------------------------------------------------------
sampling_rate <- 6
sequence_length <- 120
delay <- sampling_rate * (sequence_length + 24 - 1)
batch_size <- 256

df_to_inputs_and_targets <- function(df) {
  inputs <- df[input_data_colnames] %>%
    normalize_input_data() %>%
    as.matrix()

  targets <- as.array(df$`T (degC)`)

  list(
    head(inputs, -delay),
    tail(targets, -delay)
  )
}

make_dataset <- function(df) {
  c(inputs, targets) %<-% df_to_inputs_and_targets(df)
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


## -------------------------------------------------------------------------
c(samples, targets) %<-% iter_next(as_iterator(train_dataset))
cat("samples shape: ", format(samples$shape), "\n",
    "targets shape: ", format(targets$shape), "\n", sep = "")


## ---- eval = FALSE--------------------------------------------------------
## mean(abs(preds - targets))


## -------------------------------------------------------------------------
evaluate_naive_method <- function(dataset) {

  unnormalize_temperature <- function(x) {
    nv <- normalization_values$`T (degC)`
    (x * nv$sd) + nv$mean
  }

  temp_col_idx <- match("T (degC)", input_data_colnames)

  reduction <- dataset %>%
    dataset_unbatch() %>%
    dataset_map(function(samples, target) {
      last_temp_in_input <- samples[-1, temp_col_idx]
      pred <- unnormalize_temperature(last_temp_in_input)
      abs(pred - target)
    }) %>%
    dataset_reduce(
      initial_state = list(total_samples_seen = 0L,
                           total_abs_error = 0),
      reduce_func = function(state, element) {
        state$total_samples_seen %<>% `+`(1L)
        state$total_abs_error %<>% `+`(element)
        state
      }
    ) %>%
    lapply(as.numeric)

  mae <- with(reduction,
              total_abs_error / total_samples_seen)
  mae
}

sprintf("Validation MAE: %.2f", evaluate_naive_method(val_dataset))
sprintf("Test MAE: %.2f", evaluate_naive_method(test_dataset))


## -------------------------------------------------------------------------
ncol_input_data <- length(input_data_colnames)

inputs <- layer_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs %>%
  layer_flatten() %>%
  layer_dense(16, activation="relu") %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks = list(
    callback_model_checkpoint("jena_dense.keras",
                              save_best_only=TRUE)
)

model %>%
  compile(optimizer = "rmsprop",
          loss = "mse",
          metrics = "mae")


## -------------------------------------------------------------------------
history <- model %>%
  fit(train_dataset,
      epochs = 10,
      validation_data = val_dataset,
      callbacks = callbacks)


## -------------------------------------------------------------------------
model <- load_model("jena_dense.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)["mae"])


## -------------------------------------------------------------------------
plot(history, metrics = "mae")


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs %>%
  layer_conv_1d(8, 24, activation = "relu") %>%
  layer_max_pooling_1d(2) %>%
  layer_conv_1d(8, 12, activation = "relu") %>%
  layer_max_pooling_1d(2) %>%
  layer_conv_1d(8, 6, activation = "relu") %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks <- list(callback_model_checkpoint("jena_conv.keras",
                                            save_best_only = TRUE))

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")


## -------------------------------------------------------------------------
history <- model %>% fit(
  train_dataset,
  epochs = 10,
  validation_data = val_dataset,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
model <- load_model("jena_conv.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)["mae"])


## -------------------------------------------------------------------------
plot(history, metrics = "mae")


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs %>%
  layer_lstm(16) %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks <- list(callback_model_checkpoint("jena_lstm.keras",
                                            save_best_only = TRUE))

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")


## -------------------------------------------------------------------------
history <- model %>% fit(
  train_dataset,
  epochs = 10,
  validation_data = val_dataset,
  callbacks = callbacks
)


## -------------------------------------------------------------------------


## ---- echo = FALSE--------------------------------------------------------
local({
  p <- plot(history, metrics = "mae")
  p$data %<>% .[.$epoch > 1, ]
  print(p)
})


## -------------------------------------------------------------------------
model <- load_model("jena_lstm.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)["mae"])


## ---- eval = FALSE--------------------------------------------------------
## state_t <- 0
## for (input_t in input_sequence) {
##   output_t <- f(input_t, state_t)
##   state_t <- output_t
## }


## ---- eval = FALSE--------------------------------------------------------
## state_t <- 0
## for (input_t in input_sequence) {
##   output_t <- activation(dot(W, input_t) + dot(U, state_t) + b)
##   state_t <- output_t
## }


## -------------------------------------------------------------------------
random_array <- function(dim)
  array(runif(prod(dim)), dim)

timesteps <- 100
input_features <- 32
output_features <- 64

inputs <- random_array(c(timesteps, input_features))
state_t <- array(0, dim = output_features)

W <- random_array(c(output_features, input_features))
U <- random_array(c(output_features, output_features))
b <- random_array(c(output_features, 1))
successive_outputs <- array(0, dim = c(timesteps, output_features))

for(ts in 1:timesteps) {
  input_t <- inputs[ts, ]
  output_t <- tanh((W %*% input_t) + (U %*% state_t) + b)
  ## W %*% input_t, U %*% input_t, and b all have the same shape: (output_features, 1)
  successive_outputs[ts, ] <- output_t
  state_t <- output_t
}
final_output_sequence <- successive_outputs


## ---- eval = FALSE--------------------------------------------------------
## output_t <- tanh((W %*% input_t) + (U %*% state_t) + b)


## -------------------------------------------------------------------------
num_features <- 14
inputs <- layer_input(shape = c(NA, num_features))
outputs <- inputs %>% layer_simple_rnn(16)


## -------------------------------------------------------------------------
num_features <- 14
steps <- 120
inputs <- layer_input(shape = c(steps, num_features))
outputs <- inputs %>% layer_simple_rnn(16, return_sequences = FALSE)
outputs$shape


## -------------------------------------------------------------------------
num_features <- 14
steps <- 120
inputs <- layer_input(shape = c(steps, num_features))
outputs <- inputs %>% layer_simple_rnn(16, return_sequences = TRUE)
outputs$shape


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(steps, num_features))
outputs <- inputs %>%
  layer_simple_rnn(16, return_sequences = TRUE) %>%
  layer_simple_rnn(16, return_sequences = TRUE) %>%
  layer_simple_rnn(16)


## ---- eval = FALSE--------------------------------------------------------
## y <- activation((state_t %*% U) + (input_t %*% W) + b)


## ---- eval = FALSE--------------------------------------------------------
## output_t <-
##        activation((state_t %*% Uo) + (input_t %*% Wo) + (c_t %*% Vo) + bo)
## i_t <- activation((state_t %*% Ui) + (input_t %*% Wi) + bi)
## f_t <- activation((state_t %*% Uf) + (input_t %*% Wf) + bf)
## k_t <- activation((state_t %*% Uk) + (input_t %*% Wk) + bk)


## ---- eval = FALSE--------------------------------------------------------
## c_t+1 = i_t * k_t + c_t * f_t


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs %>%
  layer_lstm(32, recurrent_dropout = 0.25) %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks = list(callback_model_checkpoint("jena_lstm_dropout.keras",
                                           save_best_only = TRUE))

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")


## -------------------------------------------------------------------------
history <- model %>% fit(
  train_dataset,
  epochs = 50,
  validation_data = val_dataset,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
plot(history)


## ---- echo = FALSE--------------------------------------------------------
local({
  p <- plot(history, metrics = "mae")
  p$data %<>% .[.$epoch > 1, ]
  print(p)
})


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(sequence_length, num_features))
x <- inputs %>% layer_lstm(32, recurrent_dropout = 0.2, unroll = TRUE)


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs %>%
  layer_gru(32, recurrent_dropout = 0.5, return_sequences = TRUE) %>%
  layer_gru(32, recurrent_dropout = 0.5) %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks <- list(
  callback_model_checkpoint("jena_stacked_gru_dropout.keras",
                            save_best_only = TRUE)
)

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")


## -------------------------------------------------------------------------
history <- model %>% fit(
  train_dataset,
  epochs = 50,
  validation_data = val_dataset,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
plot(history)


## -------------------------------------------------------------------------
model <- load_model("jena_stacked_gru_dropout.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)["mae"])


## ---- eval = FALSE--------------------------------------------------------
## dataset_map(function(samples, targets) {
##   list(samples[, NA:NA:-1, ], targets)
## })


## ---- include = FALSE-----------------------------------------------------
inputs <- layer_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs %>%
  layer_lstm(16) %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks <- list(callback_model_checkpoint("jena_lstm_reversed.keras",
                                            save_best_only = TRUE))

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")


## ---- include=FALSE-------------------------------------------------------
dataset_reverse_time <- function(ds) {
  dataset_map(ds, function(samples, targets)
    list(samples[, NA:NA:-1, ], targets))
}
history <- model %>% fit(
  train_dataset %>% dataset_reverse_time(),
  epochs = 10,
  validation_data = val_dataset %>% dataset_reverse_time(),
  callbacks = callbacks
)

## ---- echo = FALSE--------------------------------------------------------
local({
  p <- plot(history, metrics = "mae")
  p$data %<>% .[.$epoch > 1, ]
  print(p)
})


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(sequence_length, ncol_input_data))
outputs <- inputs %>%
  bidirectional(layer_lstm(units = 16)) %>%
  layer_dense(1)

model <- keras_model(inputs, outputs)

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")


## -------------------------------------------------------------------------
history <- model %>%
  fit(train_dataset,
      epochs = 10,
      validation_data = val_dataset)
