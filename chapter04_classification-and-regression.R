library(keras3)


.[.[train_data, train_labels], .[test_data, test_labels]] <-
  dataset_imdb(num_words = 10000)


str(train_data)


str(train_labels)

max(sapply(train_data, max))


word_index <- dataset_imdb_word_index()                                         # <1>

reverse_word_index <- names(word_index)                                         # <2>
names(reverse_word_index) <- as.character(word_index)                           # <2>

decoded_words <- train_data[[1]] |>
  sapply(function(i) {
    if (i > 3) reverse_word_index[[as.character(i - 3)]]                        # <3>
    else "?"
  })
decoded_review <- paste0(decoded_words, collapse = " ")


decoded_review |> substr(1, 200) |> strwrap(70) |> cat(sep = "\n")


multi_hot_encode <- function(sequences, num_classes) {
  results <- matrix(0, nrow = length(sequences), ncol = num_classes)            # <1>
  for (i in seq_along(sequences)) {
    results[i, sequences[[i]]] <- 1                                             # <2>
  }
  results
}
x_train <- multi_hot_encode(train_data, num_classes = 10000)                    # <3>
x_test <- multi_hot_encode(test_data, num_classes = 10000)                      # <4>


x_train[1, ] |> str()


str(x_train)

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


model <- keras_model_sequential() |>
  layer_dense(16, activation = "relu") |>
  layer_dense(16, activation = "relu") |>
  layer_dense(1, activation = "sigmoid")


sigmoid <- function(x) 1 / (1 + exp(-1 * x))
withr::with_par(list(pty = "s"), {
  plot(sigmoid, -4, 4,
    main = "sigmoid",
    ylim = c(-1, 2),
    ylab = ~ sigmoid(x), xlab = ~ x,
    panel.first = grid())
})


relu <- function(x) pmax(0, x)
withr::with_par(list(pty = "s"), {
  plot(relu, -4, 4,
    main = "relu",
    ylim = c(-1, 2),
    ylab = ~ relu(x), xlab = ~ x,
    panel.first = grid())
})


model |> compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)


val_indices <- 1:1000

x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]


history <- model |> fit(
  partial_x_train, partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)


history <- model |> fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 512,
  validation_split = 0.2
)


str(history$metrics)


library(ggplot2)
plot(history) + ggtitle("[IMDB] Training history")


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


results


preds <- model |> predict(x_test)
str(preds)


.[.[train_data, train_labels], .[test_data, test_labels]] <-
  dataset_reuters(num_words = 10000)


str(train_data)


str(test_data)


train_data[[11]] |> str()


word_index <- dataset_reuters_word_index()
reverse_word_index <- setNames(object = names(word_index),
                               nm = unlist(word_index) - 3L)                    # <1>
decoded_newswire <- train_data[[1]] |>
  sapply(function(i) {
    if (i > 3) reverse_word_index[[as.character(i)]]
    else "?"
  }) |>
  paste0(collapse = " ")
decoded_newswire


train_labels[[11]]


x_train <- multi_hot_encode(train_data, num_classes = 10000)                    # <1>
x_test <- multi_hot_encode(test_data, num_classes = 10000)                      # <2>


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


y_train <- to_categorical(train_labels)
y_test <- to_categorical(test_labels)


model <- keras_model_sequential() |>
  layer_dense(64, activation = "relu") |>
  layer_dense(64, activation = "relu") |>
  layer_dense(46, activation = "softmax")


model |> compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy", metric_top_k_categorical_accuracy(k = 3))
)


val_indices <- 1:1000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- y_train[val_indices,]
partial_y_train <- y_train[-val_indices,]


history <- model |> fit(
  partial_x_train, partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)


plot(history) + ggtitle("Training and validation metrics")


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


results


mean(test_labels == sample(test_labels))


predictions <- model |> predict(x_test)


str(predictions)


sum(predictions[1, ])


which.max(predictions[1,])


y_train <- train_labels
y_test <- test_labels


model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)


model <- keras_model_sequential() |>
  layer_dense(64, activation = "relu") |>
  layer_dense(4, activation = "relu") |>
  layer_dense(46, activation = "softmax")

model |> compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)
model |> fit(
  partial_x_train, partial_y_train,
  epochs = 20,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)


.[.[train_data, train_targets], .[test_data, test_targets]] <-
  dataset_california_housing(version = "small")                                 # <1>


str(train_data)


str(test_data)


train_mean <- apply(train_data, 2, mean)
train_sd <- apply(train_data, 2, sd)
x_train <- scale(train_data, center = train_mean, scale = train_sd)
x_test <- scale(test_data, center = train_mean, scale = train_sd)


y_train <- train_targets / 100000
y_test <- test_targets / 100000


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


round(all_scores, 3)

mean(all_scores)


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


average_mae_history <- rowMeans(all_mae_histories)


plot(average_mae_history, ylab = "Validation MAE", xlab = "Epoch", type = 'l')


truncated_mae_history <- average_mae_history[-(1:10)]
plot(average_mae_history, xlab = "epoch", type = 'l',
     ylim = range(truncated_mae_history))


model <- get_model()                                                            # <1>
model |> fit(x_train, y_train,                                                  # <2>
             epochs = 130, batch_size = 16, verbose = 0)
.[test_mean_squared_error, test_mean_absolute_error] <-
  model |> evaluate(x_test, y_test)


test_mean_absolute_error


predictions <- model |> predict(test_data)
predictions[1, ]

