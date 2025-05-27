## ----setup, include = FALSE-----------------------------------------------
tensorflow::as_tensor(1)
library(keras3)



## -------------------------------------------------------------------------
library(keras3)

mnist <- dataset_mnist()
train_labels <- mnist$train$y
train_images <- array_reshape(mnist$train$x / 255,
                              c(60000, 28 * 28))

random_array <- function(dim) array(runif(prod(dim)), dim)

noise_channels <- random_array(dim(train_images))
train_images_with_noise_channels <- cbind(train_images, noise_channels)

zeros_channels <- array(0, dim(train_images))
train_images_with_zeros_channels <- cbind(train_images, zeros_channels)


## -------------------------------------------------------------------------
get_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(10, activation = "softmax")

  model %>% compile(
    optimizer = "rmsprop",
    loss = "sparse_categorical_crossentropy",
    metrics = "accuracy")

  model
}


## -------------------------------------------------------------------------
model <- get_model()
history_noise <- model %>% fit(
  train_images_with_noise_channels, train_labels,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2)

model <- get_model()
history_zeros <- model %>% fit(
  train_images_with_zeros_channels, train_labels,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2)


## -------------------------------------------------------------------------
plot(NULL,
     main = "Effect of Noise Channels on Validation Accuracy",
     xlab = "Epochs", xlim = c(1, history_noise$params$epochs),
     ylab = "Validation Accuracy", ylim = c(0.9, 1), las = 1)
lines(history_zeros$metrics$val_accuracy, lty = 1, type = "o")
lines(history_noise$metrics$val_accuracy, lty = 2, type = "o")
legend("bottomright", lty = 1:2,
       legend = c("Validation accuracy with zeros channels",
                  "Validation accuracy with noise channels"))


## -------------------------------------------------------------------------
c(c(train_images, train_labels), .) %<-% dataset_mnist()

train_images <- array_reshape(train_images / 255,
                              c(60000, 28 * 28))

random_train_labels <- sample(train_labels)

model <- keras_model_sequential() %>%
  layer_dense(512, activation = "relu") %>%
  layer_dense(10, activation = "softmax")

model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
history <- model %>% fit(train_images, random_train_labels,
                         epochs = 100,
                         batch_size = 128,
                         validation_split = 0.2)

## -------------------------------------------------------------------------
plot(history)


## ---- eval = FALSE--------------------------------------------------------
## num_validation_samples <- 10000
## val_indices <- sample.int(num_validation_samples, nrow(data))
## validation_data <- data[val_indices, ]
## training_data <- data[-val_indices, ]
## model <- get_model()
## fit(model, training_data, ...)
## validation_score <- evaluate(model, validation_data, ...)
##
## ...
##
## model <- get_model()
## fit(model, data, ...)
## test_score <- evaluate(model, test_data, ...)


## ---- eval = FALSE--------------------------------------------------------
## k <- 3
## fold_id <- sample(rep(1:k, length.out = nrow(data)))
## validation_scores <- numeric()
##
## for (fold in seq_len(k)) {
##   validation_idx <- which(fold_id == fold)
##
##   validation_data <- data[validation_idx, ]
##   training_data <- data[-validation_idx, ]
##   model <- get_model()
##   fit(model, training_data, ...)
##   validation_score <- evaluate(model, validation_data, ...)
##   validation_scores[[fold]] <- validation_score
## }
##
## validation_score <- mean(validation_scores)
## model <- get_model()
## fit(model, data, ...)
## test_score <- evaluate(model, test_data, ...)


## -------------------------------------------------------------------------
c(c(train_images, train_labels), .) %<-% dataset_mnist()
train_images <- array_reshape(train_images / 255,
                              c(60000, 28 * 28))

model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(optimizer = optimizer_rmsprop(1),
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
history <- model %>% fit(train_images, train_labels,
                         epochs = 10, batch_size = 128,
                         validation_split = 0.2)


## -------------------------------------------------------------------------
plot(history)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(optimizer = optimizer_rmsprop(1e-2),
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
model %>%
  fit(train_images, train_labels,
      epochs = 10, batch_size = 128,
      validation_split = 0.2) ->
  history


## -------------------------------------------------------------------------
plot(history)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(10, activation = "softmax")

model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")

history_small_model <- model %>%
  fit(train_images, train_labels,
      epochs = 20,
      batch_size = 128,
      validation_split = 0.2)


## -------------------------------------------------------------------------
plot(history_small_model$metrics$val_loss, type = 'o',
     main = "Effect of Insufficient Model Capacity on Validation Loss",
     xlab = "Epochs", ylab = "Validation Loss")


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
    layer_dense(96, activation="relu") %>%
    layer_dense(96, activation="relu") %>%
    layer_dense(10, activation="softmax")

model %>% compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")


## -------------------------------------------------------------------------
history_large_model <- model %>%
  fit(train_images, train_labels,
      epochs = 20,
      batch_size = 128,
      validation_split = 0.2)


## -------------------------------------------------------------------------
plot(history_large_model$metrics$val_loss, type = 'o',
     main = "Validation Loss for a Model with Appropriate Capacity",
     xlab = "Epochs", ylab = "Validation Loss")


## -------------------------------------------------------------------------
c(c(train_data, train_labels), .) %<-% dataset_imdb(num_words = 10000)

vectorize_sequences <- function(sequences, dimension=10000) {
    results <- matrix(0, nrow = length(sequences), ncol = dimension)
    for(i in seq_along(sequences))
        results[i, sequences[[i]]] <- 1
    results
}

train_data <- vectorize_sequences(train_data)

model <- keras_model_sequential() %>%
    layer_dense(16, activation="relu") %>%
    layer_dense(16, activation="relu") %>%
    layer_dense(1, activation="sigmoid")

model %>% compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics="accuracy")


## -------------------------------------------------------------------------
history_original <- model %>%
  fit(train_data, train_labels,
      epochs = 20, batch_size = 512, validation_split = 0.4)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(4, activation = "relu") %>%
  layer_dense(4, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
history_smaller_model <- model %>%
  fit(train_data, train_labels,
      epochs = 20, batch_size = 512, validation_split = 0.4)


## -------------------------------------------------------------------------
plot(
  NULL,
  main = "Original Model vs. Smaller Model on IMDB Review Classification",
  xlab = "Epochs",
  xlim = c(1, history_original$params$epochs),
  ylab = "Validation Loss",
  ylim = extendrange(history_original$metrics$val_loss),
  panel.first = abline(v = 1:history_original$params$epochs,
                       lty = "dotted", col = "lightgrey")
)

lines(history_original     $metrics$val_loss, lty = 2)
lines(history_smaller_model$metrics$val_loss, lty = 1)
legend("topleft", lty = 2:1,
       legend = c("Validation loss of original model",
                  "Validation loss of smaller model"))


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
    layer_dense(512, activation="relu") %>%
    layer_dense(512, activation="relu") %>%
    layer_dense(1, activation="sigmoid")

model %>% compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics="accuracy")


## -------------------------------------------------------------------------
history_larger_model <- model %>%
  fit(train_data, train_labels,
      epochs = 20, batch_size = 512, validation_split = 0.4)


## -------------------------------------------------------------------------
plot(
  NULL,
  main =
    "Original Model vs. Much Larger Model on IMDB Review Classification",
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


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu",
              kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dense(16, activation = "relu",
              kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics="accuracy")


## -------------------------------------------------------------------------
history_l2_reg <- model %>% fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512, validation_split = 0.4)

## -------------------------------------------------------------------------
plot(history_l2_reg)


## -------------------------------------------------------------------------
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


## -------------------------------------------------------------------------
regularizer_l1(0.001)
regularizer_l1_l2(l1 = 0.001, l2 = 0.001)


## ---- eval = FALSE--------------------------------------------------------
## zero_out <- random_array(dim(layer_output)) < .5
## layer_output[zero_out] <- 0


## ---- eval = FALSE--------------------------------------------------------
## layer_output <- layer_output * .5


## ---- eval = FALSE--------------------------------------------------------
## layer_output[random_array(dim(layer_output)) < dropout_rate] <- 0
## layer_output <- layer_output / .5


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(16, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
history_dropout <- model %>% fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512,
  validation_split = 0.4
)


## -------------------------------------------------------------------------
plot(history_dropout)


## -------------------------------------------------------------------------
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
legend("topleft", lty = 1:2,
       legend = c("Validation loss of dropout-regularized model",
                  "Validation loss of original model"))
