# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)


# ----------------------------------------------------------------------
#| lst-cap: Adding white-noise channels or all-zeros channels to MNIST
library(keras3)

.[.[train_images, train_labels], .] <- dataset_mnist()
train_images <- array_reshape(train_images / 255, c(60000, 28 * 28))

runif_array <- \(dim) array(runif(prod(dim)), dim)

noise_channels <- runif_array(dim(train_images))
train_images_with_noise_channels <- cbind(train_images, noise_channels)

zeros_channels <- array(0, dim(train_images))
train_images_with_zeros_channels <- cbind(train_images, zeros_channels)


# ----------------------------------------------------------------------
#| lst-cap: Training a model on MNIST data with noise/all-zero channels
get_model <- function() {
  model <- keras_model_sequential() |>
    layer_dense(512, activation = "relu") |>
    layer_dense(10, activation = "softmax")

  model |> compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = c("accuracy")
  )

  model
}

model <- get_model()
history_noise <- model |> fit(
  train_images_with_noise_channels, train_labels,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

model <- get_model()
history_zeros <- model |> fit(
  train_images_with_zeros_channels, train_labels,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)


# ----------------------------------------------------------------------
#| lst-cap: Plotting a validation accuracy comparison
#| fig-cap: Effect of noise channels on validation accuracy
plot(NULL,
     main = "Effect of Noise Channels on Validation Accuracy",
     xlab = "Epochs", xlim = c(1, history_noise$params$epochs),
     ylab = "Validation Accuracy", ylim = c(0.9, 0.98), las = 1)
lines(history_zeros$metrics$val_accuracy, lty = 1, type = "o")
lines(history_noise$metrics$val_accuracy, lty = 2, type = "o")
legend("bottomright", lty = 1:2,
       legend = c("Validation accuracy with zeros channels",
                  "Validation accuracy with noise channels"))


# ----------------------------------------------------------------------
#| lst-cap: Fitting an MNIST model with randomly shuffled labels
.[.[train_images, train_labels], .] <- dataset_mnist()
train_images <- array_reshape(train_images / 255,
                              c(60000, 28 * 28))

random_train_labels <- sample(train_labels)                                     # <1>

model <- keras_model_sequential() |>
  layer_dense(512, activation = "relu") |>
  layer_dense(10, activation = "softmax")

model |> compile(optimizer = "rmsprop",
                 loss = "sparse_categorical_crossentropy",
                 metrics = "accuracy")

history <- model |> fit(
  train_images, random_train_labels,
  epochs = 100, batch_size = 128,
  validation_split = 0.2
)


# ----------------------------------------------------------------------
#| eval: false
#| lst-cap: Hold-out validation (labels omitted for simplicity)
# num_validation_samples <- 10000
# val_indices <- sample.int(nrow(data), num_validation_samples)                   # <1>
# validation_data <- data[val_indices, ]                                          # <2>
# training_data <- data[-val_indices, ]                                           # <3>
# model <- get_model()                                                            # <4>
# fit(model, training_data, ...)                                                  # <4>
# validation_score <- evaluate(model, validation_data, ...)                       # <4>
# 
# ...                                                                             # <5>
# 
# model <- get_model()                                                            # <6>
# fit(model, data, ...)                                                           # <6>
# test_score <- evaluate(model, test_data, ...)                                   # <6>


# ----------------------------------------------------------------------
#| eval: false
#| lst-cap: K-fold cross-validation (labels omitted for simplicity)
# k <- 3
# fold_id <- sample(rep(1:k, length.out = nrow(data)))
# validation_scores <- numeric(k)
# 
# for (fold in seq_len(k)) {
#   validation_idx <- which(fold_id == fold)                                      # <1>
# 
#   validation_data <- data[validation_idx, ]                                     # <1>
#   training_data <- data[-validation_idx, ]                                      # <2>
#   model <- get_model()                                                          # <3>
#   fit(model, training_data, ...)
#   validation_score <- evaluate(model, validation_data, ...)
#   validation_scores[[fold]] <- validation_score
# }
# 
# validation_score <- mean(validation_scores)                                     # <4>
# model <- get_model()                                                            # <5>
# fit(model, data, ...)                                                           # <5>
# test_score <- evaluate(model, test_data, ...)                                   # <5>


# ----------------------------------------------------------------------
#| lst-cap: Training MNIST with an incorrectly high learning rate
.[.[train_images, train_labels], .] <- dataset_mnist()
train_images <- array_reshape(train_images / 255,
                              c(60000, 28 * 28))

model <- keras_model_sequential() |>
  layer_dense(512, activation = "relu") |>
  layer_dense(10, activation = "softmax")

model |> compile(
  optimizer = optimizer_rmsprop(1),
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history <- model |> fit(
  train_images, train_labels,
  epochs = 10, batch_size = 128,
  validation_split = 0.2
)


# ----------------------------------------------------------------------
#| lst-cap: The same model with a more appropriate learning rate
model <- keras_model_sequential() |>
  layer_dense(512, activation = "relu") |>
  layer_dense(10, activation = "softmax")

model |> compile(
  optimizer = optimizer_rmsprop(1e-2),
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history <- model |> fit(
  train_images, train_labels,
  epochs = 10, batch_size = 128,
  validation_split = 0.2
)


# ----------------------------------------------------------------------
#| lst-cap: A simple logistic regression on MNIST
model <- keras_model_sequential() |>
  layer_dense(10, activation = "softmax")

model |> compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history_small_model <- model |> fit(
  train_images, train_labels,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2
)


# ----------------------------------------------------------------------
#| fig-cap: Effect of insufficient model capacity on loss curves
plot(history_small_model$metrics$val_loss, type = 'o',
     main = "Effect of Insufficient Model Capacity on Validation Loss",
     xlab = "Epochs", ylab = "Validation Loss")


# ----------------------------------------------------------------------
model <- keras_model_sequential() |>
    layer_dense(128, activation="relu") |>
    layer_dense(128, activation="relu") |>
    layer_dense(10, activation="softmax")

model |> compile(
  optimizer="rmsprop",
  loss="sparse_categorical_crossentropy",
  metrics="accuracy"
)

history_large_model <- model |> fit(
  train_images, train_labels,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2
)


# ----------------------------------------------------------------------
#| fig-cap: Validation loss for a model with appropriate capacity
plot(history_large_model$metrics$val_loss, type = 'o',
     main = "Validation Loss for a Model with Appropriate Capacity",
     xlab = "Epochs", ylab = "Validation Loss")


# ----------------------------------------------------------------------
model <- keras_model_sequential() |>
  layer_dense(2048, activation = "relu") |>
  layer_dense(2048, activation = "relu") |>
  layer_dense(2048, activation = "relu") |>
  layer_dense(10, activation = "softmax")

model |> compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history_very_large_model <- model |> fit(
  train_images, train_labels,
  epochs = 20,
  batch_size = 32,                                                              # <1>
  validation_split = 0.2
)


# ----------------------------------------------------------------------
#| fig-cap: Effect of excessive model capacity on validation loss
plot(history_very_large_model$metrics$val_loss, type = 'o',
     main = "Validation Loss for a Model with Too Much Capacity",
     xlab = "Epochs", ylab = "Validation Loss")


# ----------------------------------------------------------------------
#| lst-cap: Original model
.[.[train_data, train_labels], .] <- dataset_imdb(num_words = 10000)

vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in seq_along(sequences)) {
    idx <- sequences[[i]] + 1L
    idx <- idx[idx <= dimension]
    results[i, idx] <- 1
  }
  results
}

train_data <- vectorize_sequences(train_data)

model <- keras_model_sequential() |>
    layer_dense(16, activation="relu") |>
    layer_dense(16, activation="relu") |>
    layer_dense(1, activation="sigmoid")

model |> compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

history_original <- model |> fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512, validation_split = 0.4
)


# ----------------------------------------------------------------------
#| lst-cap: Version of the model with lower capacity
model <- keras_model_sequential() |>
  layer_dense(4, activation = "relu") |>
  layer_dense(4, activation = "relu") |>
  layer_dense(1, activation = "sigmoid")

model |> compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

history_smaller_model <- model |> fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512, validation_split = 0.4
)


# ----------------------------------------------------------------------
#| fig-cap: Original model vs. smaller model on IMDb review classification
plot(
  NULL,
  main = "Original Model vs. Smaller Model on IMDB Review Classification",
  xlab = "Epochs",
  xlim = c(1, history_original$params$epochs),
  ylab = "Validation Loss",
  ylim = extendrange(c(history_original$metrics$val_loss,
                       history_smaller_model$metrics$val_loss)),
  panel.first = abline(v = 1:history_original$params$epochs,
                       lty = "dotted", col = "lightgrey")
)

lines(history_original     $metrics$val_loss, lty = 2)
lines(history_smaller_model$metrics$val_loss, lty = 1)
legend("topleft", lty = 2:1,
       legend = c("Validation loss of original model",
                  "Validation loss of smaller model"))


# ----------------------------------------------------------------------
#| lst-cap: Version of the model with higher capacity
model <- keras_model_sequential() |>
  layer_dense(512, activation = "relu") |>
  layer_dense(512, activation = "relu") |>
  layer_dense(1, activation = "sigmoid")

model |> compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

history_larger_model <- model |> fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512, validation_split = 0.4
)


# ----------------------------------------------------------------------
#| fig-cap: Original model vs. much larger model on IMDb review classification
plot(
  NULL,
  main = "Original Model vs. Much Larger Model on IMDB Review Classification",
  xlab = "Epochs", xlim = c(1, history_original$params$epochs),
  ylab = "Validation Loss",
  ylim = range(c(history_original$metrics$val_loss,
                 history_larger_model$metrics$val_loss)),
  panel.first = abline(v = 1:history_original$params$epochs,
                       lty = "dotted", col = "lightgrey")
)
lines(history_original    $metrics$val_loss, lty = 2)
lines(history_larger_model$metrics$val_loss, lty = 1)
legend("topleft", lty = 2:1,
       legend = c("Validation loss of original model",
                  "Validation loss of larger model"))


# ----------------------------------------------------------------------
#| lst-cap: Adding L2 weight regularization to the model
model <- keras_model_sequential() |>
  layer_dense(16, activation = "relu",
              kernel_regularizer = regularizer_l2(0.002)) |>
  layer_dense(16, activation = "relu",
              kernel_regularizer = regularizer_l2(0.002)) |>
  layer_dense(1, activation = "sigmoid")

model |> compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

history_l2_reg <- model |> fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512, validation_split = 0.4
)


# ----------------------------------------------------------------------
#| fig-cap: Effect of L2 weight regularization on validation loss
plot(NULL,
     main = "Effect of L2 Weight Regularization on Validation Loss",
     xlab = "Epochs", xlim = c(1, history_original$params$epochs),
     ylab = "Validation Loss",
     ylim = range(c(history_original$metrics$val_loss,
                    history_l2_reg  $metrics$val_loss)),
     panel.first = abline(v = 1:history_original$params$epochs,
                          lty = "dotted", col = "lightgrey"))
lines(history_original$metrics$val_loss, lty = 2)
lines(history_l2_reg  $metrics$val_loss, lty = 1)
legend("topleft", lty = 2:1,
       legend = c("Validation loss of original model",
                  "Validation loss of L2-regularized model"))


# ----------------------------------------------------------------------
#| results: hide
#| eval: false
#| lst-cap: Weight regularizers available in Keras
# regularizer_l1(0.001)                                                           # <1>
# regularizer_l1_l2(l1 = 0.001, l2 = 0.001)                                       # <2>


# ----------------------------------------------------------------------
#| eval: false
# zero_out <- runif_array(dim(layer_output)) < .5                                 # <1>
# layer_output[zero_out] <- 0                                                     # <1>


# ----------------------------------------------------------------------
#| eval: false
# layer_output <- layer_output * .5                                               # <1>


# ----------------------------------------------------------------------
#| eval: false
# layer_output[runif_array(dim(layer_output)) < dropout_rate] <- 0                # <1>
# layer_output <- layer_output / (1 - dropout_rate)                               # <2>


# ----------------------------------------------------------------------
#| lst-cap: Adding dropout to the IMDb model
model <- keras_model_sequential() |>
  layer_dense(16, activation = "relu") |>
  layer_dropout(0.5) |>
  layer_dense(16, activation = "relu") |>
  layer_dropout(0.5) |>
  layer_dense(1, activation = "sigmoid")

model |> compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

history_dropout <- model |> fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512,
  validation_split = 0.4
)


# ----------------------------------------------------------------------
#| fig-cap: Effect of dropout on validation loss
plot(NULL,
     main = "Effect of Dropout on Validation Loss",
     xlab = "Epochs", xlim = c(1, history_original$params$epochs),
     ylab = "Validation Loss",
     ylim = range(c(history_original$metrics$val_loss,
                    history_dropout $metrics$val_loss)),
     panel.first = abline(v = 1:history_original$params$epochs,
                          lty = "dotted", col = "lightgrey"))
lines(history_original$metrics$val_loss, lty = 2)
lines(history_dropout $metrics$val_loss, lty = 1)
legend("topleft", lty = 2:1,
       legend = c("Validation loss of original model",
                  "Validation loss of dropout-regularized model"))


