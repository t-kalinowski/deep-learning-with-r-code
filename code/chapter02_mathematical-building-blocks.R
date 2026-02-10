# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "scales")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
keras3::use_backend("tensorflow")
# TF specific tf$Gradient() used in this chapter


# ----------------------------------------------------------------------
#| lst-cap: Loading the MNIST dataset in Keras
library(keras3)
.[.[train_images, train_labels], .[test_images, test_labels]] <-
  dataset_mnist()


# ----------------------------------------------------------------------
str(train_images)
str(train_labels)


# ----------------------------------------------------------------------
str(test_images)
str(test_labels)


# ----------------------------------------------------------------------
#| lst-cap: The network architecture
model <- keras_model_sequential() |>
  layer_dense(512, activation = "relu") |>
  layer_dense(10, activation = "softmax")


# ----------------------------------------------------------------------
#| lst-cap: The compilation step
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)


# ----------------------------------------------------------------------
#| lst-cap: Preparing the image data
train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255


# ----------------------------------------------------------------------
#| lst-cap: _Fitting_ the model
fit(model, train_images, train_labels, epochs = 5, batch_size = 128)


# ----------------------------------------------------------------------
#| lst-cap: Using the model to make predictions
test_digits <- test_images[1:10, ]
predictions <- predict(model, test_digits)
str(predictions)


# ----------------------------------------------------------------------
predictions[1, ]


# ----------------------------------------------------------------------
which.max(predictions[1,])
predictions[1, 8]


# ----------------------------------------------------------------------
test_labels[1]


# ----------------------------------------------------------------------
#| lst-cap: Evaluating the model on new data
metrics <- evaluate(model, test_images, test_labels)
metrics$accuracy


# ----------------------------------------------------------------------
np <- import("numpy")
np$sum(1:3)


# ----------------------------------------------------------------------
np <- import("numpy", convert = FALSE)
np$sum(1:3)
np$sum(1:3) |> np$square() |> py_to_r()


# ----------------------------------------------------------------------
x <- np$array(12)
x
x$ndim


# ----------------------------------------------------------------------
x <- as.array(c(12, 3, 6, 14, 7))
str(x)
dim(x)


# ----------------------------------------------------------------------
x <- np_array(x)
x
x$shape
x$ndim


# ----------------------------------------------------------------------
x <- array(seq(2 * 3), dim = c(2, 3))
x
length(dim(x))


# ----------------------------------------------------------------------
x <- np_array(x)
x
x$ndim


# ----------------------------------------------------------------------
x <- array(seq(2 * 3 * 4), dim = c(4, 3, 2))
str(x)
length(dim(x))
x


# ----------------------------------------------------------------------
x <- np_array(x)
str(x)
x$ndim
x$shape
x


# ----------------------------------------------------------------------
.[.[train_images, train_labels], .[test_images, test_labels]] <-
  dataset_mnist(convert = FALSE)                                                # <1>


# ----------------------------------------------------------------------
op_ndim(train_images)


# ----------------------------------------------------------------------
op_shape(train_images)


# ----------------------------------------------------------------------
op_dtype(train_images)


# ----------------------------------------------------------------------
train_labels@r[5]


# ----------------------------------------------------------------------
#| no_mar: false
#| lst-cap: Displaying the fifth digit
digit <- train_images@r[5, , ] |> py_to_r() |>
  scales::rescale(to = c(0, 1), from = c(255, 0))
par(pty = "s", mar = c(1, 1, 1, 1))
plot(as.raster(digit), interpolate = FALSE)
box()


# ----------------------------------------------------------------------
x <- np_array(1:10)
x@r[1]
x@py[1]


# ----------------------------------------------------------------------
x@r[1:3]
x@py[1:3]


# ----------------------------------------------------------------------
x@r[NA:5]
x@r[6:NA]


# ----------------------------------------------------------------------
my_slice <- train_images@r[10:99]                                               # <1>
my_slice <- train_images@r[10:99, , ]                                           # <2>
my_slice <- train_images@r[10:99, NA:NA, NA:NA]                                 # <3>
my_slice <- train_images@r[10:99, 1:28, 1:28]                                   # <4>
my_slice$shape


# ----------------------------------------------------------------------
my_slice <- train_images@r[, 15:NA, 15:NA]
my_slice$shape


# ----------------------------------------------------------------------
train_images$shape
train_images@r[newaxis] |> _$shape
train_images@r[, newaxis] |> _$shape
train_images@r[.., newaxis] |> _$shape
train_images@r[.., newaxis, ] |> _$shape


# ----------------------------------------------------------------------
x <- np_array(1:10)
x@r[-1]
x@r[-3:NA]


# ----------------------------------------------------------------------
my_slice <- train_images@r[, 8:-8, 8:-8]
my_slice$shape


# ----------------------------------------------------------------------
batch <- train_images@r[1:128, , ]                                              # <1>


# ----------------------------------------------------------------------
batch <- train_images@r[129:256, , ]


# ----------------------------------------------------------------------
n <- 3
ids <- seq(to = 128 * n, length.out = 128)
batch <- train_images@r[ids, , ]


# ----------------------------------------------------------------------
#| eval: false
# layer_dense(units = 512, activation="relu")


# ----------------------------------------------------------------------
#| eval: false
# output <- relu(matmul(input, W) + b)


# ----------------------------------------------------------------------
naive_relu <- function(x) {
  stopifnot(is.array(x), length(dim(x)) == 2)                                   # <1>
  for (i in 1:nrow(x))
    for (j in 1:ncol(x))
      x[i, j] <- max(x[i, j], 0)                                                # <2>
  x
}


# ----------------------------------------------------------------------
naive_add <- function(x, y) {
  stopifnot(is.array(x), is.array(y),
            length(dim(x)) == 2, dim(x) == dim(y))                              # <1>
  for (i in 1:nrow(x))
    for (j in 1:ncol(x))
      x[i, j]  <- x[i, j] + y[i, j]
  x
}


# ----------------------------------------------------------------------
#| eval: false
# z <- x + y                                                                      # <1>
# z <- pmax(z, 0.)                                                                # <2>


# ----------------------------------------------------------------------
runif_array <- function(dim) {
  array(runif(prod(dim)), dim)
}


# ----------------------------------------------------------------------
x <- runif_array(c(20, 100))
y <- runif_array(c(20, 100))

system.time({
  for (i in seq_len(10000)) {
    z <- x + y
    z <- pmax(z, 0)
  }
})[["elapsed"]]


# ----------------------------------------------------------------------
system.time({
  for (i in seq_len(10000)) {
    z <- naive_add(x, y)
    z <- naive_relu(z)
  }
})[["elapsed"]]


# ----------------------------------------------------------------------
X <- np$random$random(shape(32, 10))                                            # <1>
y <- np$random$random(shape(10))                                                # <2>


# ----------------------------------------------------------------------
y <- y@r[newaxis, ..]
str(y)                                                                          # <1>


# ----------------------------------------------------------------------
Y <- np$tile(y, shape(32, 1))                                                   # <1>
Y$shape


# ----------------------------------------------------------------------
#| eval: false
# y <- runif_array(c(10))
# Y <- local({
#   dim(y) <- c(1, 10)
#   y[rep(1, 32), ]                                                               # <1>
# })


# ----------------------------------------------------------------------
naive_add_matrix_and_vector <- function(x, y) {
  stopifnot(length(dim(x)) == 2,                                                # <1>
            length(dim(y)) == 1,                                                # <2>
            ncol(x) == dim(y))
  for (i in seq_len(dim(x)[1]))
    for (j in seq_len(dim(x)[2]))
      x[i, j] <- x[i, j] + y[j]
  x
}


# ----------------------------------------------------------------------
x <- np_array(runif_array(c(64, 3, 32, 10)))                                    # <1>
y <- np_array(runif_array(c(32, 10)))                                           # <2>
z <- x + y                                                                      # <3>


# ----------------------------------------------------------------------
z_explicit <- x + y[newaxis, newaxis, ..]
all(py_to_r(z == z_explicit))


# ----------------------------------------------------------------------
x <- np_array(runif_array(32))
y <- np_array(runif_array(32))

z <- x %*% y                                                                    # <1>

np <- import("numpy", convert = FALSE)
z <- np$matmul(x, y)                                                            # <2>


# ----------------------------------------------------------------------
naive_vector_dot <- function(x, y) {
  stopifnot(length(dim(x)) == 1,                                                # <1>
            length(dim(y)) == 1,                                                # <1>
            dim(x) == dim(y))                                                   # <1>
  z <- 0
  for (i in seq_along(x))
    z <- z + x[i] * y[i]
  z
}


# ----------------------------------------------------------------------
naive_matrix_vector_dot <- function(x, y) {
  stopifnot(length(dim(x)) == 2,                                                # <1>
            length(dim(y)) == 1,                                                # <2>
            nrow(x) == dim(y))                                                  # <3>
  z <- array(0, dim = nrow(x))                                                  # <4>
  for (i in 1:nrow(x))
    for (j in 1:ncol(x))
      z[i] <- z[i] + x[i, j] * y[j]
  z
}


# ----------------------------------------------------------------------
naive_matrix_vector_dot <- function(x, y) {
  z <- array(0, dim = c(nrow(x)))
  for (i in 1:nrow(x))
    z[i] <- naive_vector_dot(x[i, ], y)
  z
}


# ----------------------------------------------------------------------
naive_matrix_dot <- function(x, y) {
  stopifnot(length(dim(x)) == 2,                                                # <1>
            length(dim(y)) == 2,                                                # <1>
            ncol(x) == nrow(y))                                                 # <2>
  z <- array(0, dim = c(nrow(x), ncol(y)))                                      # <3>
  for (i in 1:nrow(x))                                                          # <4>
    for (j in 1:ncol(y)) {                                                      # <5>
      row_x <- x[i, ]
      column_y <- y[, j]
      z[i, j] <- naive_vector_dot(row_x, column_y)
    }
  z
}


# ----------------------------------------------------------------------
train_images <- array_reshape(train_images, c(60000, 28 * 28))


# ----------------------------------------------------------------------
x <- array(1:6)
x
array_reshape(x, dim = c(3, 2))
array_reshape(x, dim = c(2, 3))


# ----------------------------------------------------------------------
x <- array(1:6, dim = c(3, 2))
x
t(x)


# ----------------------------------------------------------------------
#| eval: false
# past_velocity <- 0
# momentum <- 0.1                                                                 # <1>
# while (loss > 0.01) {                                                           # <2>
#   .[w, loss, gradient] <- get_current_parameters()
#   velocity <- past_velocity * momentum - learning_rate * gradient
#   w <- w + momentum * velocity - learning_rate * gradient
#   past_velocity <- velocity
#   update_parameter(w)
# }


# ----------------------------------------------------------------------
#| eval: false
# loss_value <- loss(
#   y_true,
#   softmax(matmul(relu(matmul(inputs, W1) + b1), W2) + b2)
# )


# ----------------------------------------------------------------------
#| eval: false
# fg <- function(x) {
#   x1 <- g(x)
#   y <- f(x1)
#   y
# }


# ----------------------------------------------------------------------
#| eval: false
# fghj <- function(x) {
#   x1 <- j(x)
#   x2 <- h(x1)
#   x3 <- g(x2)
#   y <- f(x3)
#   y
# }
# 
# grad(y, x) == (grad(y, x3) * grad(x3, x2) * grad(x2, x1) * grad(x1, x))


# ----------------------------------------------------------------------
.[.[train_images, train_labels], .[test_images, test_labels]] <-
  dataset_mnist(convert = FALSE)

preprocess_images <- function(images) {
  images <- images$
    reshape(shape(nrow(images), 28 * 28))$
    astype("float32")
  images / 255
}

train_images <- train_images |> preprocess_images()
test_images <- test_images |> preprocess_images()
str(named_list(train_images,  test_images))


# ----------------------------------------------------------------------
model <- keras_model_sequential() |>
  layer_dense(units = 512, activation = "relu") |>
  layer_dense(units = 10, activation = "softmax")


# ----------------------------------------------------------------------
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)


# ----------------------------------------------------------------------
fit(model, train_images, train_labels, epochs = 5, batch_size = 128)


# ----------------------------------------------------------------------
#| eval: false
# output <- activation(matmul(input, W) + b)


# ----------------------------------------------------------------------
layer_naive_dense <- function(input_size, output_size, activation = NULL) {
  self <- new.env(parent = emptyenv())                                          # <1>
  attr(self, "class") <- "NaiveDense"

  self$activation <- activation

  self$W <- keras_variable(shape = shape(input_size, output_size),              # <2>
                           initializer = "uniform", dtype = "float32")          # <2>

  self$b <- keras_variable(shape = shape(output_size),                          # <3>
                           initializer = "zeros", dtype = "float32")            # <3>

  self$weights <- list(self$W, self$b)                                          # <4>

  self$call <- function(inputs) {                                               # <5>
    x <- (inputs %*% self$W) + self$b                                           # <5>
    if (is.function(self$activation))                                           # <5>
      x <- self$activation(x)                                                   # <5>
    x                                                                           # <5>
  }                                                                             # <5>

  self
}


# ----------------------------------------------------------------------
naive_sequential_model <- function(layers) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveSequential"

  self$layers <- layers

  self$weights <- unlist(lapply(layers, `[[`, "weights"))

  self$call <- function(inputs) {
    x <- inputs
    for (layer in self$layers)
      x <- layer$call(x)
    x
  }

  self
}


# ----------------------------------------------------------------------
model <- naive_sequential_model(list(
  layer_naive_dense(input_size = 28 * 28, output_size = 512,
                    activation = op_relu),
  layer_naive_dense(input_size = 512, output_size = 10,
                    activation = op_softmax)
))
stopifnot(length(model$weights) == 4)


# ----------------------------------------------------------------------
new_batch_generator <- function(images, labels, batch_size = 128) {
  stopifnot(nrow(images) == nrow(labels))
  next_start <- 1

  function(exhausted = NULL) {                                                  # <1>
    start <- next_start
    if (start > nrow(images))
      return(exhausted)                                                         # <2>

    end <- start + batch_size - 1
    end <- min(end, nrow(images))                                               # <3>

    next_start <<- end + 1
    list(images = images@r[start:end, ..],
         labels = labels@r[start:end, ..])
  }
}


# ----------------------------------------------------------------------
#| eval: false
#| lst-cap: A single step of training
# one_training_step <- function(model, images_batch, labels_batch) {
#   predictions <- model$call(images_batch)                                       # <1>
#   loss <- op_sparse_categorical_crossentropy(labels_batch, predictions)         # <1>
#   average_loss <- op_mean(loss)
#   gradients <- get_gradients_of_loss_wrt_weights(loss, model$weights)           # <2>
#   update_weights(gradients, model$weights)                                      # <3>
#   loss
# }


# ----------------------------------------------------------------------
learning_rate <- 1e-3

update_weights <- function(gradients, weights) {
  Map(\(w, g) w$assign(w - g * learning_rate),                                  # <1>
      weights, gradients)
}


# ----------------------------------------------------------------------
optimizer <- optimizer_sgd(learning_rate = 1e-3)

update_weights <- function(gradients, weights) {
  optimizer$apply(gradients, weights)
}


# ----------------------------------------------------------------------
#| message: false
library(tensorflow)

x <- tf$zeros(shape = shape())                                                  # <1>
with(tf$GradientTape() %as% tape, {                                             # <2>
  y <- 2 * x + 3                                                                # <3>
})
grad_of_y_wrt_x <- tape$gradient(y, x)                                          # <4>


# ----------------------------------------------------------------------
one_training_step <- function(model, images_batch, labels_batch) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model$call(images_batch)
    per_sample_losses <-
      op_sparse_categorical_crossentropy(labels_batch, predictions)
    average_loss <- op_mean(per_sample_losses)
  })
  gradients <- tape$gradient(average_loss, model$weights)
  update_weights(gradients, model$weights)
  average_loss
}


# ----------------------------------------------------------------------
fit <- function(model, images, labels, epochs, batch_size = 128) {
  for (epoch_counter in seq_len(epochs)) {
    cat("Epoch ", epoch_counter, "\n")
    batch_generator <- new_batch_generator(images, labels, batch_size)
    batch_counter <- 0
    repeat {
      batch <- batch_generator() %||% break
      batch_counter <- batch_counter + 1
      loss <- one_training_step(model, batch$images, batch$labels)
      if (batch_counter %% 100 == 0)
        cat(sprintf("loss at batch %s: %.2f\n", batch_counter, loss))
    }
  }
}


# ----------------------------------------------------------------------
mnist <- dataset_mnist(convert = FALSE)
train_images <- mnist$train$x$
  reshape(shape(60000, 28 * 28))$astype("float32") / 255
test_images <- mnist$test$x$
  reshape(shape(10000, 28 * 28))$astype("float32") / 255
train_labels <- mnist$train$y
test_labels <- mnist$test$y

fit(model, train_images, train_labels, epochs = 10, batch_size = 128)


# ----------------------------------------------------------------------
predictions <- model$call(test_images) |> as.array()                            # <1>
predicted_labels <- max.col(predictions) - 1L                                   # <2>
matches <- predicted_labels == as.array(test_labels)
cat(sprintf("accuracy: %.2f\n", mean(matches)))


