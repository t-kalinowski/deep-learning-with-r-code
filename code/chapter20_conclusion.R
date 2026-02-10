# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "envir")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
Sys.setenv("CUDA_VISIBLE_DEVICES" = "")
library(keras3)
use_backend("jax", FALSE)
py_require("keras-hub")

num_input_features <- 10
src_seq_length <- 20
num_classes <- 10
num_values <- 10
num_features <- 10
height <- width <- 512
channels <- 3
vocab_size <- 200L
num_timesteps <- 100
sequence_length <- 100
embed_dim <- 10L
dst_seq_length <- 20
seq_length <- 20L


# ----------------------------------------------------------------------
inputs <- keras_input(shape = c(num_input_features))
outputs <- inputs |>
  layer_dense(32, activation = "relu") |>
  layer_dense(32, activation = "relu") |>
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model |> compile(optimizer = "rmsprop", loss = "binary_crossentropy")


# ----------------------------------------------------------------------
inputs <- keras_input(shape = c(num_input_features))
outputs <- inputs |>
  layer_dense(32, activation = "relu") |>
  layer_dense(32, activation = "relu") |>
  layer_dense(num_classes, activation = "softmax")
model <- keras_model(inputs, outputs)
model |> compile(optimizer = "rmsprop", loss = "categorical_crossentropy")


# ----------------------------------------------------------------------
inputs <- keras_input(shape = c(num_input_features))
outputs <- inputs |>
  layer_dense(32, activation = "relu") |>
  layer_dense(32, activation = "relu") |>
  layer_dense(num_classes, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model |> compile(optimizer = "rmsprop", loss = "binary_crossentropy")


# ----------------------------------------------------------------------
inputs <- keras_input(shape = c(num_input_features))
outputs <- inputs |>
  layer_dense(32, activation = "relu") |>
  layer_dense(32, activation = "relu") |>
  layer_dense(num_values)
model <- keras_model(inputs, outputs)
model |> compile(optimizer = "rmsprop", loss = "mse")


# ----------------------------------------------------------------------
inputs <- keras_input(shape = c(height, width, channels))
outputs <- inputs |>
  layer_separable_conv_2d(32, 3, activation = "relu") |>
  layer_separable_conv_2d(64, 3, activation = "relu") |>
  layer_max_pooling_2d(2) |>
  layer_separable_conv_2d(64, 3, activation = "relu") |>
  layer_separable_conv_2d(128, 3, activation = "relu") |>
  layer_max_pooling_2d(2) |>
  layer_separable_conv_2d(64, 3, activation = "relu") |>
  layer_separable_conv_2d(128, 3, activation = "relu") |>
  layer_global_average_pooling_2d() |>
  layer_dense(32, activation = "relu") |>
  layer_dense(num_classes, activation = "softmax")
model <- keras_model(inputs, outputs)
model |> compile(optimizer = "rmsprop", loss = "categorical_crossentropy")


# ----------------------------------------------------------------------
reticulate::py_require("keras-hub")
keras_hub <- reticulate::import("keras_hub")
envir::import_from(
  keras_hub$layers,
  TokenAndPositionEmbedding,
  TransformerDecoder,
  TransformerEncoder
)

encoder_inputs <- keras_input(c(src_seq_length), dtype="int64")                 # <1>

encoder_outputs <- encoder_inputs |>
  (\(x) {
    TokenAndPositionEmbedding(vocab_size, src_seq_length, embed_dim)(x)
  })() |>
  (\(x) TransformerEncoder(intermediate_dim = 256L, num_heads = 8L)(x))()

decoder_inputs <- keras_input(c(dst_seq_length), dtype = "int64")               # <2>
decoder_outputs <- decoder_inputs |>
  (\(x) {
    TokenAndPositionEmbedding(vocab_size, dst_seq_length, embed_dim)(x)
  })() |>
  (\(x) {
    layer <- TransformerDecoder(intermediate_dim = 256L, num_heads = 8L)
    layer(x, encoder_outputs)
  })() |>
  layer_dense(vocab_size, activation = "softmax")                               # <3>

transformer <- keras_model(
  c(encoder_inputs, decoder_inputs),
  decoder_outputs
)
transformer |> compile(
  optimizer = "adamw",
  loss = "categorical_crossentropy"
)


# ----------------------------------------------------------------------
inputs <- x <- keras_input(c(seq_length), dtype = "int64")
x <- TokenAndPositionEmbedding(vocab_size, seq_length, embed_dim)(x)
x <- TransformerEncoder(intermediate_dim = 256L, num_heads = 8L)(x)
x <- layer_global_max_pooling_1d()(x)
outputs <- layer_dense(units = 1, activation = "sigmoid")(x)
model <- keras_model(inputs, outputs)
model |> compile(optimizer = "adamw", loss = "binary_crossentropy")


# ----------------------------------------------------------------------
inputs <- keras_input(shape = c(num_timesteps, num_features))
outputs <- inputs |>
  layer_lstm(32) |>
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model |> compile(optimizer = "rmsprop", loss = "binary_crossentropy")


# ----------------------------------------------------------------------
inputs <- keras_input(shape = c(num_timesteps, num_features))
outputs <- inputs |>
  layer_lstm(32, return_sequences = TRUE) |>
  layer_lstm(32, return_sequences = TRUE) |>
  layer_lstm(32) |>
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model |> compile(optimizer = "rmsprop", loss = "binary_crossentropy")


