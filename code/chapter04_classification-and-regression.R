# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "envir", "ggplot2", "withr")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
#| lst-cap: Loading the IMDb dataset
library(keras3)

.[.[train_data, train_labels], .[test_data, test_labels]] <-
  dataset_imdb(num_words = 10000)


# ----------------------------------------------------------------------
#| eval: false
# imdb <- dataset_imdb(num_words = 10000)
# train_data <- imdb$train$x
# train_labels <- imdb$train$y
# test_data <- imdb$test$x
# test_labels <- imdb$test$y


# ----------------------------------------------------------------------
str(train_data)


# ----------------------------------------------------------------------
max(sapply(train_data, max))


# ----------------------------------------------------------------------
str(train_labels)


# ----------------------------------------------------------------------
word_index <- dataset_imdb_word_index()                                         # <1>
str(word_index)
max(unlist(word_index))
stopifnot(all(
  1:max(unlist(word_index)) == sort(unlist(word_index))
))


# ----------------------------------------------------------------------
#| lst-cap: Decoding reviews back to text
imdb_token_id_to_word <- c(                                                     # <1>
  "<padding>", "<start-of-sequence>", "<unknown>", "<unused>",                  # <2>
  names(sort(unlist(word_index)))                                               # <3>
)

decode_imdb_words <- function(token_ids) {
  paste0(imdb_token_id_to_word[token_ids + 1L],                                 # <4>
         collapse = " ")
}


# ----------------------------------------------------------------------
decode_imdb_words(head(train_data[[1]], 32))  |>
  strwrap() |> cat(sep = "\n")


# ----------------------------------------------------------------------
#| lst-cap: Encoding the integer sequences via multi-hot encoding
multi_hot_encode <- function(sequences, num_classes) {
  results <- matrix(0, nrow = length(sequences), ncol = num_classes)            # <1>
  for (i in seq_along(sequences)) {
    results[i, sequences[[i]] + 1] <- 1                                         # <2>
  }
  results
}
x_train <- multi_hot_encode(train_data, num_classes = 10000)                    # <3>
x_test <- multi_hot_encode(test_data, num_classes = 10000)                      # <4>


# ----------------------------------------------------------------------
str(x_train)


# ----------------------------------------------------------------------
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


# ----------------------------------------------------------------------
#| lst-cap: Model definition
model <- keras_model_sequential() |>
  layer_dense(16, activation = "relu") |>
  layer_dense(16, activation = "relu") |>
  layer_dense(1, activation = "sigmoid")


# ----------------------------------------------------------------------
#| fig-cap: The sigmoid function
sigmoid <- function(x) 1 / (1 + exp(-1 * x))
withr::with_par(list(pty = "s", las = 1), {
  plot(sigmoid, -4, 4,
    main = "Sigmoid",
    ylim = c(-1, 2),
    ylab = ~ sigmoid(x), xlab = ~ x,
    panel.first = grid())
})


# ----------------------------------------------------------------------
#| fig-cap: The rectified linear unit function
relu <- function(x) pmax(0, x)
withr::with_par(list(pty = "s", las = 1), {
  plot(relu, -4, 4,
    main = "ReLU",
    ylim = c(-1, 2),
    ylab = ~ relu(x), xlab = ~ x,
    panel.first = grid())
})


# ----------------------------------------------------------------------
#| eval: false
# output <- dot(input, W) + b


# ----------------------------------------------------------------------
#| lst-cap: Compiling the model
model |> compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)


# ----------------------------------------------------------------------
#| lst-cap: Setting aside a validation set
val_indices <- 1:10000

x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]


# ----------------------------------------------------------------------
#| lst-cap: Training the model
history <- model |> fit(
  partial_x_train, partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)


# ----------------------------------------------------------------------
#| eval: false
# history <- model |> fit(
#   x_train, y_train,
#   epochs = 20,
#   batch_size = 512,
#   validation_split = 0.2
# )
# 


# ----------------------------------------------------------------------
str(history$metrics)


# ----------------------------------------------------------------------
#| lst-cap: Plotting the training and validation loss and accuracy
#| fig-cap: IMDb training and validation metrics
library(ggplot2)
plot(history) + ggtitle("[IMDb] Training history")


# ----------------------------------------------------------------------
#| lst-cap: Retraining a model from scratch
model <- keras_model_sequential() |>
  layer_dense(16, activation = "relu") |>
  layer_dense(16, activation = "relu") |>
  layer_dense(1, activation = "sigmoid")

model |> compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

model |> fit(x_train, y_train, epochs = 4, batch_size = 512)

results <- model |> evaluate(x_test, y_test)


# ----------------------------------------------------------------------
str(results)                                                                    # <1>


# ----------------------------------------------------------------------
preds <- model |> predict(x_test)
str(preds)


# ----------------------------------------------------------------------
#| fig-cap: Histogram of predicted probabilities (IMDb)
hist(preds)


# ----------------------------------------------------------------------
#| lst-cap: Loading the Reuters dataset
.[.[train_data, train_labels], .[test_data, test_labels]] <-
  dataset_reuters(num_words = 10000)


# ----------------------------------------------------------------------
str(train_data)


# ----------------------------------------------------------------------
str(test_data)


# ----------------------------------------------------------------------
#| lst-cap: Decoding newswires back to text
word_index <- dataset_reuters_word_index()                                      # <1>
reuters_token_id_to_word <- c(                                                  # <2>
  "<padding>", "<start-of-sequence>", "<unknown>", "<unused>",                  # <3>
  names(sort(unlist(word_index)))                                               # <4>
)
decode_reuters_words <- function(token_ids) {
  paste0(reuters_token_id_to_word[token_ids + 1L],                              # <5>
         collapse = " ")
}


# ----------------------------------------------------------------------
str(train_labels)


# ----------------------------------------------------------------------
#| lst-cap: Encoding the input data
x_train <- multi_hot_encode(train_data, num_classes = 10000)                    # <1>
x_test <- multi_hot_encode(test_data, num_classes = 10000)                      # <2>


# ----------------------------------------------------------------------
#| lst-cap: Encoding the labels
one_hot_encode <- function(labels, num_classes = 46) {
  results <- matrix(0, nrow = length(labels), ncol = num_classes)
  for (i in seq_along(labels)) {
    label_position <- labels[[i]] + 1                                           # <1>
    results[i, label_position] <- 1
  }
  results
}

y_train <- one_hot_encode(train_labels)                                         # <2>
y_test <- one_hot_encode(test_labels)                                           # <3>


# ----------------------------------------------------------------------
y_train <- to_categorical(train_labels)
y_test <- to_categorical(test_labels)


# ----------------------------------------------------------------------
#| lst-cap: Model definition
model <- keras_model_sequential() |>
  layer_dense(64, activation = "relu") |>
  layer_dense(64, activation = "relu") |>
  layer_dense(46, activation = "softmax")


# ----------------------------------------------------------------------
#| lst-cap: Compiling the model
model |> compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c(
    "accuracy",
    metric_top_k_categorical_accuracy(k = 3, name = "top_3_accuracy")
  )
)


# ----------------------------------------------------------------------
#| lst-cap: Setting aside a validation set
val_indices <- 1:1000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- y_train[val_indices,]
partial_y_train <- y_train[-val_indices,]


# ----------------------------------------------------------------------
#| lst-cap: Training the model
history <- model |> fit(
  partial_x_train, partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)


# ----------------------------------------------------------------------
#| lst-cap: "Plotting the training and validation loss, accuracy, and top-3 accuracy"
#| fig-cap: Training and validation metrics (Reuters)
plot(history) + ggtitle("Training and validation metrics")


# ----------------------------------------------------------------------
#| lst-cap: Retraining a model from scratch
model <- keras_model_sequential() |>
  layer_dense(64, activation = "relu") |>
  layer_dense(64, activation = "relu") |>
  layer_dense(46, activation = "softmax")

model |> compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

model |> fit(x_train, y_train, epochs = 9, batch_size = 512)

results <- model |> evaluate(x_test, y_test)


# ----------------------------------------------------------------------
str(results)


# ----------------------------------------------------------------------
mean(test_labels == sample(test_labels))


# ----------------------------------------------------------------------
predictions <- model |> predict(x_test)


# ----------------------------------------------------------------------
str(predictions)


# ----------------------------------------------------------------------
sum(predictions[1, ])


# ----------------------------------------------------------------------
envir::import_from(dplyr, near)
all(near(1, rowSums(predictions), tol = 1e-6))


# ----------------------------------------------------------------------
which.max(predictions[1, ])


# ----------------------------------------------------------------------
y_train <- train_labels
y_test <- test_labels


# ----------------------------------------------------------------------
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)


# ----------------------------------------------------------------------
#| lst-cap: A model with an information bottleneck
model <- keras_model_sequential() |>
  layer_dense(64, activation = "relu") |>
  layer_dense(4, activation = "relu") |>
  layer_dense(46, activation = "softmax")

model |> compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)
history <- model |> fit(
  partial_x_train, partial_y_train,
  epochs = 20,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)


# ----------------------------------------------------------------------
#| fig-cap: Training history of a model with an information bottleneck
plot(history)


# ----------------------------------------------------------------------
#| lst-cap: Loading the California housing dataset
.[.[train_data, train_targets], .[test_data, test_targets]] <-
  dataset_california_housing(version = "small")                                 # <1>


# ----------------------------------------------------------------------
str(train_data)


# ----------------------------------------------------------------------
str(test_data)


# ----------------------------------------------------------------------
str(train_targets)


# ----------------------------------------------------------------------
#| lst-cap: Normalizing the data
train_mean <- apply(train_data, 2, mean)
train_sd <- apply(train_data, 2, sd)
x_train <- scale(train_data, center = train_mean, scale = train_sd)
x_test <- scale(test_data, center = train_mean, scale = train_sd)


# ----------------------------------------------------------------------
#| lst-cap: Scaling the targets
y_train <- train_targets / 100000
y_test <- test_targets / 100000


# ----------------------------------------------------------------------
#| lst-cap: Model definition
get_model <- function() {                                                       # <1>
  model <- keras_model_sequential() |>
    layer_dense(64, activation = "relu") |>
    layer_dense(64, activation = "relu") |>
    layer_dense(1)
  model |> compile(
    optimizer = "adam",
    loss = "mean_squared_error",
    metrics = "mean_absolute_error"
  )
  model
}


# ----------------------------------------------------------------------
#| lst-cap: K-fold validation
k <- 4
fold_id <- sample(rep(1:k, length.out = nrow(train_data)))
num_epochs <- 50
all_scores <- numeric(k)

for (i in 1:k) {
  cat(sprintf("Processing fold #%i\n", i))

  fold_val_indices <- which(fold_id == i)
  fold_x_val <- x_train[fold_val_indices, ]                                     # <1>
  fold_y_val <- y_train[fold_val_indices]                                       # <1>
  fold_x_train <- x_train[-fold_val_indices, ]                                  # <2>
  fold_y_train <- y_train[-fold_val_indices]                                    # <2>

  model <- get_model()                                                          # <3>
  model |> fit(                                                                 # <4>
    fold_x_train, fold_y_train,
    epochs = num_epochs, batch_size = 16, verbose = 0
  )
  results <- model |> evaluate(fold_x_val, fold_y_val, verbose = 0)             # <5>
  all_scores[i] <- results$mean_absolute_error
}


# ----------------------------------------------------------------------
round(all_scores, 3)


# ----------------------------------------------------------------------
mean(all_scores)


# ----------------------------------------------------------------------
#| lst-cap: Saving the validation logs at each fold
k <- 4
num_epochs <- 200
all_mae_histories <- list()

for (i in 1:k) {
  cat(sprintf("Processing fold #%i\n", i))

  fold_val_indices <- which(fold_id == i)                                       # <1>
  fold_x_val <- x_train[fold_val_indices, ]                                     # <1>
  fold_y_val <- y_train[fold_val_indices]                                       # <1>
  fold_x_train <- x_train[-fold_val_indices, ]                                  # <2>
  fold_y_train <- y_train[-fold_val_indices]                                    # <2>

  model <- get_model()                                                          # <3>
  history <- model |> fit(                                                      # <4>
    fold_x_train, fold_y_train,
    validation_data = list(fold_x_val, fold_y_val),
    epochs = num_epochs, batch_size = 16, verbose = 0
  )
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories[[i]] <- mae_history
}

all_mae_histories <- do.call(cbind, all_mae_histories)


# ----------------------------------------------------------------------
#| lst-cap: Building the history of successive mean K-fold validation scores
average_mae_history <- rowMeans(all_mae_histories)


# ----------------------------------------------------------------------
#| lst-cap: Plotting validation scores
#| fig-cap: Validation MAE by epoch
plot(average_mae_history, ylab = "Validation MAE", xlab = "Epoch", type = 'l')


# ----------------------------------------------------------------------
#| lst-cap: "Plotting validation scores, excluding the first 10 data points"
#| fig-cap: "Validation MAE by epoch, excluding the first 10 data points"
truncated_mae_history <- average_mae_history[-(1:10)]
plot(average_mae_history, type = 'l',
     ylab = "Validation MAE", xlab = "Epoch",
     ylim = range(truncated_mae_history))


# ----------------------------------------------------------------------
#| lst-cap: Training the final model
model <- get_model()                                                            # <1>
model |> fit(x_train, y_train,                                                  # <2>
             epochs = 130, batch_size = 16, verbose = 0)
.[test_mean_squared_error, test_mean_absolute_error] <-
  model |> evaluate(x_test, y_test)


# ----------------------------------------------------------------------
test_mean_absolute_error


# ----------------------------------------------------------------------
predictions <- model |> predict(x_test)
predictions[1, ]


