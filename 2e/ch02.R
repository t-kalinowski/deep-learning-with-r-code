## ----setup, include=FALSE-------------------------------------------------
tensorflow::as_tensor(1)


## -------------------------------------------------------------------------
library(tensorflow)
library(keras3)
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y


## -------------------------------------------------------------------------
str(train_images)
str(train_labels)


## -------------------------------------------------------------------------
str(test_images)
str(test_labels)


## -------------------------------------------------------------------------
model <- keras_model_sequential(list(
  layer_dense(units = 512, activation = "relu"),
  layer_dense(units = 10, activation = "softmax")
))


## -------------------------------------------------------------------------
compile(model,
        optimizer = "rmsprop",
        loss = "sparse_categorical_crossentropy",
        metrics = "accuracy")


## -------------------------------------------------------------------------
train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255


## -------------------------------------------------------------------------
fit(model, train_images, train_labels, epochs = 5, batch_size = 128)


## -------------------------------------------------------------------------
test_digits <- test_images[1:10, ]
predictions <- predict(model, test_digits)
str(predictions)
predictions[1, ]


## -------------------------------------------------------------------------
which.max(predictions[1, ])
predictions[1, 8]


## -------------------------------------------------------------------------
test_labels[1]


## -------------------------------------------------------------------------
metrics <- evaluate(model, test_images, test_labels)
metrics["accuracy"]


## -------------------------------------------------------------------------
x <- as.array(c(12, 3, 6, 14, 7))
str(x)
length(dim(x))


## -------------------------------------------------------------------------
x <- array(seq(3 * 5), dim = c(3, 5))
x
dim(x)


## -------------------------------------------------------------------------
x <- array(seq(2 * 3 * 4), dim = c(2, 3, 4))
str(x)
length(dim(x))


## -------------------------------------------------------------------------
library(keras3)
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y


## -------------------------------------------------------------------------
length(dim(train_images))


## -------------------------------------------------------------------------
dim(train_images)


## -------------------------------------------------------------------------
typeof(train_images)


## -------------------------------------------------------------------------
digit <- train_images[5, , ]
par(mar = c(0, 0, 0, 0))
plot(as.raster(abs(255 - digit), max = 255))


## -------------------------------------------------------------------------
train_labels[5]


## -------------------------------------------------------------------------
my_slice <- train_images[10:99, , ]
dim(my_slice)


## -------------------------------------------------------------------------
my_slice <- train_images[, 15:28, 15:28]
dim(my_slice)


## -------------------------------------------------------------------------
batch <- train_images[1:128, , ]


## -------------------------------------------------------------------------
batch <- train_images[129:256, , ]


## -------------------------------------------------------------------------
n <- 3
batch <- train_images[seq(to = 128 * n, length.out = 128), , ]


## -------------------------------------------------------------------------
layer_dense(units = 512, activation = "relu")


## ---- eval = FALSE--------------------------------------------------------
## output <- relu(dot(W, input) + b)


## -------------------------------------------------------------------------
naive_relu <- function(x) {
  stopifnot(length(dim(x)) == 2)
  for (i in 1:nrow(x))
    for (j in 1:ncol(x))
      x[i, j] <- max(x[i, j], 0)
  x
}


## -------------------------------------------------------------------------
naive_add <- function(x, y) {
  stopifnot(length(dim(x)) == 2, dim(x) == dim(y))
  for (i in 1:nrow(x))
    for (j in 1:ncol(x))
      x[i, j]  <- x[i, j] + y[i, j]
  x
}


## ---- eval = FALSE--------------------------------------------------------
## z <- x + y
## z[z < 0] <- 0


## -------------------------------------------------------------------------
random_array <- function(dim, min = 0, max = 1)
  array(runif(prod(dim), min, max),
        dim)

x <- random_array(c(20, 100))
y <- random_array(c(20, 100))

system.time({
  for (i in seq_len(1000)) {
    z <- x + y
    z[z < 0] <- 0
  }
})[["elapsed"]]


## -------------------------------------------------------------------------
system.time({
  for (i in seq_len(1000)) {
    z <- naive_add(x, y)
    z <- naive_relu(z)
  }
})[["elapsed"]]


## -------------------------------------------------------------------------
X <- random_array(c(64, 3, 32, 10))
y <- random_array(c(10))


## -------------------------------------------------------------------------
dim(y) <- c(1, 10)
str(y)


## -------------------------------------------------------------------------
Y <- y[rep(1, 32), ]
str(Y)


## -------------------------------------------------------------------------
naive_add_matrix_and_vector <- function(x, y) {
  stopifnot(length(dim(x)) == 2,
            length(dim(y)) == 1,
            ncol(x) == dim(y))
  for (i in seq(dim(x)[1]))
    for (j in seq(dim(x)[2]))
      x[i, j] <- x[i, j] + y[j]
  x
}


## -------------------------------------------------------------------------
x <- random_array(c(32))
y <- random_array(c(32))
z <- x %*% y


## -------------------------------------------------------------------------
naive_vector_dot <- function(x, y) {
  stopifnot(length(dim(x)) == 1,
            length(dim(y)) == 1,
            dim(x) == dim(y))
  z <- 0
  for (i in seq_along(x))
    z <- z + x[i] * y[i]
  z
}


## -------------------------------------------------------------------------
naive_matrix_vector_dot <- function(x, y) {
  stopifnot(length(dim(x)) == 2,
            length(dim(y)) == 1,
            nrow(x) == dim(y))
  z <- array(0, dim = nrow(x))
  for (i in 1:nrow(x))
    for (j in 1:ncol(x))
      z[i] <- z[i] + x[i, j] * y[j]
  z
}


## -------------------------------------------------------------------------
naive_matrix_vector_dot <- function(x, y) {
  z <- array(0, dim = c(nrow(x)))
  for (i in 1:nrow(x))
    z[i] <- naive_vector_dot(x[i, ], y)
  z
}


## -------------------------------------------------------------------------
naive_matrix_dot <- function(x, y) {
  stopifnot(length(dim(x)) == 2,
            length(dim(y)) == 2,
            ncol(x) == nrow(y))
  z <- array(0, dim = c(nrow(x), ncol(y)))
  for (i in 1:nrow(x))
    for (j in 1:ncol(y)) {
      row_x <- x[i, ]
      column_y <- y[, j]
      z[i, j] <- naive_vector_dot(row_x, column_y)
    }
  z
}


## -------------------------------------------------------------------------
train_images <- array_reshape(train_images, c(60000, 28 * 28))


## -------------------------------------------------------------------------
x <- array(1:6)
x
array_reshape(x, dim = c(3, 2))
array_reshape(x, dim = c(2, 3))


## -------------------------------------------------------------------------
x <- array(1:6, dim = c(3, 2))
x
t(x)


## ---- eval = FALSE--------------------------------------------------------
## output <- relu(dot(input, W) + b)


## ---- eval = FALSE--------------------------------------------------------
## f(x + epsilon_x) = y + a * epsilon_x


## ---- eval = FALSE--------------------------------------------------------
## y_pred = dot(W, x)
## loss_value = loss_fn(y_pred, y_true)


## ---- eval = FALSE--------------------------------------------------------
## loss_value = f(W)


## ---- eval = FALSE--------------------------------------------------------
## past_velocity <- 0
## momentum <- 0.1
## repeat {
##   p <- get_current_parameters()
##
##   if (p$loss <= 0.01)
##     break
##
##   velocity <- past_velocity * momentum + learning_rate * p$gradient
##   w <- p$w + momentum * velocity - learning_rate * p$gradient
##
##   past_velocity <- velocity
##   update_parameter(w)
## }


## ---- eval = FALSE--------------------------------------------------------
## loss_value <- loss(y_true,
##                    softmax(dot(relu(dot(inputs, W1) + b1), W2) + b2))


## -------------------------------------------------------------------------
fg <- function(x) {
  x1 <- g(x)
  y <- f(x1)
  y
}


## ---- eval = FALSE--------------------------------------------------------
## fghj <- function(x) {
##     x1 <- j(x)
##     x2 <- h(x1)
##     x3 <- g(x2)
##     y <- f(x3)
##     y
## }
##
## grad(y, x) == (grad(y, x3) * grad(x3, x2) * grad(x2, x1) * grad(x1, x))


## -------------------------------------------------------------------------
library(tensorflow)
x <- tf$Variable(0)
with(tf$GradientTape() %as% tape, {
  y <- 2 * x + 3
})
grad_of_y_wrt_x <- tape$gradient(y, x)


## -------------------------------------------------------------------------
x <- tf$Variable(array(0, dim = c(2, 2)))
with(tf$GradientTape() %as% tape, {
  y <- 2 * x + 3
})
grad_of_y_wrt_x <- as.array(tape$gradient(y, x))


## -------------------------------------------------------------------------
W <- tf$Variable(random_array(c(2, 2)))
b <- tf$Variable(array(0, dim = c(2)))

x <- random_array(c(2, 2))
with(tf$GradientTape() %as% tape, {
    y <- tf$matmul(x, W) + b
})
grad_of_y_wrt_W_and_b <- tape$gradient(y, list(W, b))
str(grad_of_y_wrt_W_and_b)


## -------------------------------------------------------------------------
library(keras3)
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255

test_images <- mnist$test$x
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

train_labels <- mnist$train$y
test_labels <- mnist$test$y


## -------------------------------------------------------------------------
model <- keras_model_sequential(list(
  layer_dense(units = 512, activation = "relu"),
  layer_dense(units = 10, activation = "softmax")
))


## -------------------------------------------------------------------------
compile(model,
        optimizer = "rmsprop",
        loss = "sparse_categorical_crossentropy",
        metrics = c("accuracy"))


## -------------------------------------------------------------------------
fit(model, train_images, train_labels, epochs = 5, batch_size = 128)


## ---- eval = FALSE--------------------------------------------------------
## output <- activation(dot(W, input) + b)


## -------------------------------------------------------------------------
layer_naive_dense <- function(input_size, output_size, activation) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveDense"

  self$activation <- activation

  w_shape <- c(input_size, output_size)
  w_initial_value <- random_array(w_shape, min = 0, max = 1e-1)
  self$W <- tf$Variable(w_initial_value)

  b_shape <- c(output_size)
  b_initial_value <- array(0, b_shape)
  self$b <- tf$Variable(b_initial_value)

  self$weights <- list(self$W, self$b)

  self$call <- function(inputs) {
    self$activation(tf$matmul(inputs, self$W) + self$b)
  }

  self
}


## -------------------------------------------------------------------------
naive_model_sequential <- function(layers) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveSequential"

  self$layers <- layers

  weights <- lapply(layers, function(layer) layer$weights)
  self$weights <- do.call(c, weights)

  self$call <- function(inputs) {
    x <- inputs
    for (layer in self$layers)
      x <- layer$call(x)
    x
  }

  self
}


## -------------------------------------------------------------------------
model <- naive_model_sequential(list(
  layer_naive_dense(input_size = 28 * 28, output_size = 512,
                    activation = tf$nn$relu),
  layer_naive_dense(input_size = 512, output_size = 10,
                    activation = tf$nn$softmax)
))
stopifnot(length(model$weights) == 4)


## -------------------------------------------------------------------------
new_batch_generator <- function(images, labels, batch_size = 128) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "BatchGenerator"

  stopifnot(nrow(images) == nrow(labels))
  self$index <- 1
  self$images <- images
  self$labels <- labels
  self$batch_size <- batch_size
  self$num_batches <- ceiling(nrow(images) / batch_size)

  self$get_next_batch <- function() {
    start <- self$index
    if(start > nrow(images))
      return(NULL)

    end <- start + self$batch_size - 1
    if(end > nrow(images))
      end <- nrow(images)

    self$index <- end + 1
    indices <- start:end
    list(images = self$images[indices, ],
         labels = self$labels[indices])
  }

  self
}


## -------------------------------------------------------------------------
one_training_step <- function(model, images_batch, labels_batch) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model$call(images_batch)
    per_sample_losses <-
      loss_sparse_categorical_crossentropy(labels_batch, predictions)
    average_loss <- mean(per_sample_losses)
  })
  gradients <- tape$gradient(average_loss, model$weights)
  update_weights(gradients, model$weights)
  average_loss
}


## -------------------------------------------------------------------------
learning_rate <- 1e-3

update_weights <- function(gradients, weights) {
  stopifnot(length(gradients) == length(weights))
  for (i in seq_along(weights))
    weights[[i]]$assign_sub(gradients[[i]] * learning_rate)
}


## -------------------------------------------------------------------------
optimizer <- optimizer_sgd(learning_rate=1e-3)

update_weights <- function(gradients, weights)
    optimizer$apply_gradients(zip_lists(gradients, weights))


## -------------------------------------------------------------------------
str(zip_lists(gradients = list("grad_for_wt_1", "grad_for_wt_2", "grad_for_wt_3"),
              weights = list("weight_1", "weight_2", "weight_3")))


## -------------------------------------------------------------------------
fit <- function(model, images, labels, epochs, batch_size = 128) {
  for (epoch_counter in seq_len(epochs)) {
    cat("Epoch ", epoch_counter, "\n")
    batch_generator <- new_batch_generator(images, labels)
    for (batch_counter in seq_len(batch_generator$num_batches)) {
      batch <- batch_generator$get_next_batch()
      loss <- one_training_step(model, batch$images, batch$labels)
      if (batch_counter %% 100 == 0)
        cat(sprintf("loss at batch %s: %.2f\n", batch_counter, loss))
    }
  }
}


## -------------------------------------------------------------------------
mnist <- dataset_mnist()
train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28)) / 255
test_images <- array_reshape(mnist$test$x, c(10000, 28 * 28)) / 255
test_labels <- mnist$test$y
train_labels <- mnist$train$y

fit(model, train_images, train_labels, epochs = 10, batch_size = 128)


## -------------------------------------------------------------------------
predictions <- model$call(test_images)
predictions <- as.array(predictions)
predicted_labels <- max.col(predictions) - 1
matches <- predicted_labels == test_labels
cat(sprintf("accuracy: %.2f\n", mean(matches)))
