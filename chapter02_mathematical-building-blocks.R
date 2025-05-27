keras3::use_backend("tensorflow")


library(keras3)
.[.[train_images, train_labels], .[test_images, test_labels]] <- dataset_mnist()


str(train_images)
str(train_labels)


str(test_images)
str(test_labels)


model <- keras_model_sequential(layers = list(
  layer_dense(units = 512, activation = "relu"),
  layer_dense(units = 10, activation = "softmax")
))


model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)


train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

test_labels <- matrix(test_labels, ncol = 1)
train_labels <- matrix(train_labels, ncol = 1)


fit(model, train_images, train_labels, epochs = 5, batch_size = 128)


test_digits <- test_images[1:10, ]
predictions <- predict(model, test_digits)
str(predictions)
predictions[1, ]


which.max(predictions[1,])
predictions[1, 8]


test_labels[1]


metrics <- evaluate(model, test_images, test_labels)
metrics$accuracy


x <- as.array(c(12, 3, 6, 14, 7))
str(x)
dim(x)


x <- np_array(x)
x
x$shape
x$ndim


x <- array(seq(3 * 5), dim = c(3, 5))
x
dim(x)

np_array(x)


x <- array(seq(2 * 3 * 4), dim = c(2, 3, 4))
str(x)
dim(x)
length(dim(x))


x
np_array(x)


.[.[train_images, train_labels], .[test_images, test_labels]] <-
  dataset_mnist()                                                               # <1>


length(dim(train_images))


dim(train_images)


typeof(train_images)


knitr::knit_hooks$set(no_mar = function(before, ...) {
  if (before) par(mar = c(0, 0, 0, 0))
})


digit <- train_images[5, , ]
plot(as.raster(abs(255 - digit), max = 255))


train_labels[5]


x <- np_array(1:10)
x@r[1]
x@py[1]


x@r[1:3]
x@py[1:3]


x@r[NA:5]
x@r[6:NA]


images <- np_array(train_images, "float32")
my_slice <- images@r[10:99]                                                     # <1>
my_slice <- images@r[10:99, , ]                                                 # <2>
my_slice <- images@r[10:99, NA:NA, NA:NA]                                       # <3>
my_slice <- images@r[10:99, 1:28, 1:28]                                         # <4>
my_slice$shape


my_slice <- images@r[, 15:NA, 15:NA]
my_slice$shape


images$shape                                                                    # <1>
images@r[newaxis] |> _$shape                                                    # <2>
images@r[, newaxis] |> _$shape                                                  # <3>
images@r[.., newaxis] |> _$shape                                                # <4>
images@r[.., newaxis, ] |> _$shape                                              # <5>


x <- np_array(1:10)
x@r[-1]
x@r[-3:-1]
x@r[-3:NA]


my_slice <- images@r[, 8:-8, 8:-8]
my_slice$shape


batch <- train_images[1:128, , ]                                                # <1>


batch <- train_images[129:256, , ]


n <- 3
ids <- seq(to = 128 * n, length.out = 128)
batch <- train_images[ids, , ]


naive_relu <- function(x) {
  stopifnot(is.array(x), length(dim(x)) == 2)                                   # <1>
  for (i in 1:nrow(x))
    for (j in 1:ncol(x))
      x[i, j] <- max(x[i, j], 0)                                                # <2>
  x
}


naive_add <- function(x, y) {
  stopifnot(is.array(x), is.array(y),
            length(dim(x)) == 2, dim(x) == dim(y))                              # <1>
  for (i in 1:nrow(x))
    for (j in 1:ncol(x))
      x[i, j]  <- x[i, j] + y[i, j]
  x
}


runif_array <- function(dim) {
  array(runif(prod(dim)), dim)
}

x <- runif_array(c(20, 100))
y <- runif_array(c(20, 100))

system.time({
  for (i in seq_len(1000)) {
    z <- x + y
    z <- pmax(z, 0)
  }
})[["elapsed"]]


x <- runif_array(c(20, 100))
y <- runif_array(c(20, 100))

system.time({
  for (i in seq_len(1000)) {
    z <- naive_add(x, y)
    z <- naive_relu(z)
  }
})[["elapsed"]]


X <- runif_array(c(32, 10))                                                     # <1>
y <- runif_array(c(10))                                                         # <2>


dim(y) <- c(1, 10)
str(y)                                                                          # <1>


Y <- y[rep(1, 32), ]
dim(Y)                                                                          # <1>


naive_add_matrix_and_vector <- function(x, y) {
  stopifnot(length(dim(x)) == 2,                                                # <1>
            length(dim(y)) == 1,                                                # <2>
            ncol(x) == dim(y))
  for (i in seq_len(dim(x)[1]))
    for (j in seq_len(dim(x)[2]))
      x[i, j] <- x[i, j] + y[j]
  x
}


x <- np_array(runif_array(c(64, 3, 32, 10)))                                    # <1>
y <- np_array(runif_array(c(32, 10)))                                           # <2>
z <- x + y                                                                      # <3>


x <- np_array(runif_array(32))
y <- np_array(runif_array(32))

z <- x %*% y                                                                    # <1>

np <- reticulate::import("numpy", convert = FALSE)
z <- np$matmul(x, y)                                                            # <2>


naive_vector_dot <- function(x, y) {
  stopifnot(length(dim(x)) == 1,                                                # <1>
            length(dim(y)) == 1,                                                # <1>
            dim(x) == dim(y)) #                                                 # <1>
  z <- 0
  for (i in seq_along(x))
    z <- z + x[i] * y[i]
  z
}


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


naive_matrix_vector_dot <- function(x, y) {
  z <- array(0, dim = c(nrow(x)))
  for (i in 1:nrow(x))
    z[i] <- naive_vector_dot(x[i, ], y)
  z
}


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


train_images <- array_reshape(train_images, c(60000, 28 * 28))


x <- array(1:6)
x
array_reshape(x, dim = c(3, 2))
array_reshape(x, dim = c(2, 3))


x <- array(1:6, dim = c(3, 2))
x
t(x)


.[.[train_images, train_labels], .[test_images, test_labels]] <-
  dataset_mnist()

train_images <- array_reshape(train_images, c(60000, 28 * 28)) / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28)) / 255

test_labels <- matrix(test_labels, ncol = 1)
train_labels <- matrix(train_labels, ncol = 1)


model <- keras_model_sequential() |>
  layer_dense(units = 512, activation = "relu") |>
  layer_dense(units = 10, activation = "softmax")


compile(model,
        optimizer = "rmsprop",
        loss = "sparse_categorical_crossentropy",
        metrics = c("accuracy"))


fit(model, train_images, train_labels, epochs = 5, batch_size = 128)


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


naive_sequential_model <- function(layers) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveSequential"

  self$layers <- layers

  self$weights <- lapply(layers, \(layer) layer$weights) |> unlist()

  self$call <- function(inputs) {
    x <- inputs
    for (layer in self$layers)
      x <- layer$call(x)
    x
  }

  self
}


model <- naive_sequential_model(list(
  layer_naive_dense(input_size = 28 * 28, output_size = 512,
                    activation = op_relu),
  layer_naive_dense(input_size = 512, output_size = 10,
                    activation = op_softmax)
))
stopifnot(length(model$weights) == 4)


new_batch_generator <- function(images, labels, batch_size = 128) {

  stopifnot(nrow(images) == nrow(labels))
  index <- 1

  function() {                                                                  # <1>
    start <- index
    if(start > nrow(images))
      return(NULL)                                                              # <2>

    end <- start + batch_size - 1
    if(end > nrow(images))
      end <- nrow(images)                                                       # <3>

    index <<- end + 1
    list(images = images[start:end, , drop = FALSE],
         labels = labels[start:end, , drop = FALSE])
  }
}


learning_rate <- 1e-3

update_weights <- function(gradients, weights) {
  mapply(function(w, g) {
    w$assign(w - g * learning_rate)                                             # <1>
  }, weights, gradients)
}


optimizer <- optimizer_sgd(learning_rate = 1e-3)

update_weights <- function(gradients, weights) {
  optimizer$apply(gradients, weights)
}


library(tensorflow)

x <- tf$zeros(shape = shape())                                                  # <1>
with(tf$GradientTape() %as% tape, {                                             # <2>
  y <- 2 * x + 3                                                                # <3>
})
grad_of_y_wrt_x <- tape$gradient(y, x)                                          # <4>


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


fit <- function(model, images, labels, epochs, batch_size = 128) {
  for (epoch_counter in seq_len(epochs)) {
    cat("Epoch ", epoch_counter, "\n")
    batch_generator <- new_batch_generator(images, labels, batch_size)
    batch_counter <- 0
    repeat {
      batch <- batch_generator()
      if (is.null(batch))
        break
      batch_counter <- batch_counter + 1
      loss <- one_training_step(model, batch$images, batch$labels)
      if (batch_counter %% 100 == 0)
        cat(sprintf("loss at batch %s: %.2f\n", batch_counter, loss))
    }
  }
}


mnist <- dataset_mnist()
train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28)) / 255
test_images <- array_reshape(mnist$test$x, c(10000, 28 * 28)) / 255
train_labels <- matrix(mnist$train$y)
test_labels <- matrix(mnist$test$y)

fit(model, train_images, train_labels, epochs = 10, batch_size = 128)


predictions <- model$call(test_images)
predictions <- as.array(predictions) # convert Tensorflow Tensor to R array
predicted_labels <- max.col(predictions) - 1
matches <- predicted_labels == test_labels
cat(sprintf("accuracy: %.2f\n", mean(matches)))

