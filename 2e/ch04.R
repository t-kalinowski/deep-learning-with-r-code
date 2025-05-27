## ----setup, include = FALSE-----------------------------------------------
tensorflow::as_tensor(1)


## -------------------------------------------------------------------------
library(keras3)

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb


## -------------------------------------------------------------------------
str(train_data)
str(train_labels)


## -------------------------------------------------------------------------
max(sapply(train_data, max))


## -------------------------------------------------------------------------
word_index <- dataset_imdb_word_index()

reverse_word_index <- names(word_index)
names(reverse_word_index) <- as.character(word_index)

decoded_words <- train_data[[1]] %>%
  sapply(function(i) {
    if (i > 3) reverse_word_index[[as.character(i - 3)]]
    else "?"
    })
decoded_review <- paste0(decoded_words, collapse = " ")
decoded_review


## -------------------------------------------------------------------------
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- array(0, dim = c(length(sequences), dimension))
  for (i in seq_along(sequences)) {
    sequence <- sequences[[i]]
    for (j in sequence)
      results[i, j] <- 1
  }
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)


## -------------------------------------------------------------------------
str(x_train)


## -------------------------------------------------------------------------
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")


## ---- eval = FALSE--------------------------------------------------------
## output <- relu(dot(input, W) + b)


## ---- eval = FALSE--------------------------------------------------------
## output <- dot(input, W) + b


## -------------------------------------------------------------------------
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
x_val <- x_train[seq(10000), ]
partial_x_train <- x_train[-seq(10000), ]
y_val <- y_train[seq(10000)]
partial_y_train <- y_train[-seq(10000)]


## -------------------------------------------------------------------------
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)


## -------------------------------------------------------------------------
str(history$metrics)


## ---- message=FALSE-------------------------------------------------------
plot(history)


## -------------------------------------------------------------------------
history_df <- as.data.frame(history)
str(history_df)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")

model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)


## -------------------------------------------------------------------------
results


## ---- include = FALSE-----------------------------------------------------
model %>% predict(x_test)


## -------------------------------------------------------------------------
reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters


## -------------------------------------------------------------------------
length(train_data)
length(test_data)


## -------------------------------------------------------------------------
str(train_data)


## -------------------------------------------------------------------------
word_index <- dataset_reuters_word_index()

reverse_word_index <- names(word_index)
names(reverse_word_index) <- as.character(word_index)

decoded_words <- train_data[[1]] %>%
  sapply(function(i) {
    if (i > 3) reverse_word_index[[as.character(i - 3)]]
    else "?"
    })
decoded_review <- paste0(decoded_words, collapse = " ")
decoded_review


## -------------------------------------------------------------------------
str(train_labels)


## -------------------------------------------------------------------------
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in seq_along(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)


## -------------------------------------------------------------------------
to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  labels <- labels + 1
  for(i in seq_along(labels)) {
    j <- labels[[i]]
    results[i, j] <- 1
  }
  results
}
y_train <- to_one_hot(train_labels)
y_test <- to_one_hot(test_labels)


## -------------------------------------------------------------------------
y_train <- to_categorical(train_labels)
y_test <- to_categorical(test_labels)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(46, activation = "softmax")


## -------------------------------------------------------------------------
model %>% compile(optimizer = "rmsprop",
                  loss = "categorical_crossentropy",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
val_indices <- 1:1000

x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]

y_val <- y_train[val_indices, ]
partial_y_train <- y_train[-val_indices, ]


## -------------------------------------------------------------------------
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)


## -------------------------------------------------------------------------
plot(history)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(46, activation = "softmax")

model %>% compile(optimizer = "rmsprop",
                  loss = "categorical_crossentropy",
                  metrics = "accuracy")

model %>% fit(x_train, y_train, epochs = 9, batch_size = 512)

results <- model %>% evaluate(x_test, y_test)


## -------------------------------------------------------------------------
results


## -------------------------------------------------------------------------
mean(test_labels == sample(test_labels))


## -------------------------------------------------------------------------
predictions <- model %>% predict(x_test)


## -------------------------------------------------------------------------
str(predictions)


## -------------------------------------------------------------------------
sum(predictions[1, ])


## -------------------------------------------------------------------------
which.max(predictions[1, ])


## -------------------------------------------------------------------------
y_train <- train_labels
y_test <- test_labels


## -------------------------------------------------------------------------
model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy")


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(4, activation = "relu") %>%
  layer_dense(46, activation = "softmax")

model %>% compile(optimizer = "rmsprop",
                  loss = "categorical_crossentropy",
                  metrics = "accuracy")

model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)


## -------------------------------------------------------------------------
boston <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% boston


## -------------------------------------------------------------------------
str(train_data)
str(test_data)


## -------------------------------------------------------------------------
str(train_targets)


## -------------------------------------------------------------------------
mean <- apply(train_data, 2, mean)
sd <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = sd)
test_data <- scale(test_data, center = mean, scale = sd)


## -------------------------------------------------------------------------
build_model <- function() {

  model <- keras_model_sequential() %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(1)

  model %>% compile(optimizer = "rmsprop",
                    loss = "mse",
                    metrics = "mae")
  model
}


## -------------------------------------------------------------------------
k <- 4
fold_id <- sample(rep(1:k, length.out = nrow(train_data)))
num_epochs <- 100
all_scores <- numeric()

for (i in 1:k) {
  cat("Processing fold #", i, "\n")

  val_indices <- which(fold_id == i)

  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]

  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]

  model <- build_model()

  model %>% fit(
    partial_train_data,
    partial_train_targets,
    epochs = num_epochs,
    batch_size = 16,
    verbose = 0
  )

  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores[[i]] <- results[['mae']]
}


## -------------------------------------------------------------------------
all_scores
mean(all_scores)


## -------------------------------------------------------------------------
num_epochs <- 500
all_mae_histories <- list()
for (i in 1:k) {
  cat("Processing fold #", i, "\n")

  val_indices <- which(fold_id == i)
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]

  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]

  model <- build_model()
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 16, verbose = 0
  )
  mae_history <- history$metrics$val_mae
  all_mae_histories[[i]] <- mae_history
}

all_mae_histories <- do.call(cbind, all_mae_histories)


## -------------------------------------------------------------------------
average_mae_history <- rowMeans(all_mae_histories)


## -------------------------------------------------------------------------
plot(average_mae_history, xlab = "epoch", type = 'l')


## -------------------------------------------------------------------------
truncated_mae_history <- average_mae_history[-(1:10)]
plot(average_mae_history, xlab = "epoch", type = 'l',
     ylim = range(truncated_mae_history))


## -------------------------------------------------------------------------
model <- build_model()
model %>% fit(train_data, train_targets,
              epochs = 130, batch_size = 16, verbose = 0)
result <- model %>% evaluate(test_data, test_targets)


## -------------------------------------------------------------------------
result$mae


## -------------------------------------------------------------------------
predictions <- model %>% predict(test_data)
predictions[1, ]
