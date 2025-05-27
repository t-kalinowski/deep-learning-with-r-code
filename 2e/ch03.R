## ----setup, include = FALSE-----------------------------------------------
library(tensorflow)
library(keras3)
tf_function(function(x) x+1)(as_tensor(1))


## ---- eval = FALSE--------------------------------------------------------
## install.packages("keras")
##
## library(reticulate)
## virtualenv_create("r-reticulate", python = install_python())
##
## library(keras3)
## install_keras(envname = "r-reticulate")


## -------------------------------------------------------------------------
tensorflow::tf_config()


## -------------------------------------------------------------------------
r_array <- array(1:6, c(2, 3))
tf_tensor <- as_tensor(r_array)
tf_tensor


## -------------------------------------------------------------------------
dim(tf_tensor)
tf_tensor + tf_tensor


## -------------------------------------------------------------------------
methods(class = "tensorflow.tensor")


## -------------------------------------------------------------------------
tf_tensor$ndim


## -------------------------------------------------------------------------
as_tensor(1)$ndim
as_tensor(1:2)$ndim


## -------------------------------------------------------------------------
tf_tensor$shape


## -------------------------------------------------------------------------
methods(class = class(shape())[1])


## -------------------------------------------------------------------------
shape(2, 3)


## -------------------------------------------------------------------------
tf_tensor$dtype


## -------------------------------------------------------------------------
r_array <- array(1)
typeof(r_array)
as_tensor(r_array)$dtype


## -------------------------------------------------------------------------
as_tensor(r_array, dtype = "float32")


## -------------------------------------------------------------------------
as_tensor(0, shape = c(2, 3))


## -------------------------------------------------------------------------
as_tensor(1:6, shape = c(2, 3))


## -------------------------------------------------------------------------
array(1:6, dim = c(2, 3))


## -------------------------------------------------------------------------
array_reshape(1:6, c(2, 3), order = "C")
array_reshape(1:6, c(2, 3), order = "F")


## -------------------------------------------------------------------------
array_reshape(1:6, c(-1, 3))
as_tensor(1:6, shape = c(NA, 3))


## -------------------------------------------------------------------------
train_images <- as_tensor(dataset_mnist()$train$x)
my_slice <- train_images[, 15:NA, 15:NA]


## -------------------------------------------------------------------------
my_slice <- train_images[, 8:-8, 8:-8]


## -------------------------------------------------------------------------
my_slice <- train_images[1:100, all_dims()]


## -------------------------------------------------------------------------
my_slice <- train_images[1:100, , ]


## -------------------------------------------------------------------------
x <- as_tensor(1, shape = c(64, 3, 32, 10))
y <- as_tensor(2, shape = c(32, 10))
z <- x + y


## -------------------------------------------------------------------------
z <- x + y[tf$newaxis, tf$newaxis, , ]


## -------------------------------------------------------------------------
library(tensorflow)
tf$ones(shape(1, 3))
tf$zeros(shape(1, 3))
tf$random$normal(shape(1, 3), mean = 0, stddev = 1)
tf$random$uniform(shape(1, 3))


## ---- error = TRUE--------------------------------------------------------
tf$ones(c(2, 1))


## -------------------------------------------------------------------------
tf$ones(c(2L, 1L))


## -------------------------------------------------------------------------
m <- as_tensor(1:12, shape = c(3, 4))
tf$reduce_mean(m, axis = 0L, keepdims = TRUE)


## -------------------------------------------------------------------------
mean(m, axis = 1, keepdims = TRUE)


## -------------------------------------------------------------------------
x <- array(1, dim = c(2, 2))
x[1, 1] <- 0


## ---- error = TRUE--------------------------------------------------------
x <- as_tensor(1, shape = c(2, 2))
x[1, 1] <- 0


## -------------------------------------------------------------------------
v <- tf$Variable(initial_value = tf$random$normal(shape(3, 1)))
v


## -------------------------------------------------------------------------
v$assign(tf$ones(shape(3, 1)))


## -------------------------------------------------------------------------
v[1, 1]$assign(3)


## -------------------------------------------------------------------------
v$assign_add(tf$ones(shape(3, 1)))


## -------------------------------------------------------------------------
a <- tf$ones(c(2L, 2L))
b <- tf$square(a)
c <- tf$sqrt(a)
d <- b + c
e <- tf$matmul(a, b)
e <- e * d


## -------------------------------------------------------------------------
input_var <- tf$Variable(initial_value = 3)
with(tf$GradientTape() %as% tape, {
  result <- tf$square(input_var)
})
gradient <- tape$gradient(result, input_var)


## -------------------------------------------------------------------------
input_const <- as_tensor(3)
with(tf$GradientTape() %as% tape, {
   tape$watch(input_const)
   result = tf$square(input_const)
})
gradient <- tape$gradient(result, input_const)


## -------------------------------------------------------------------------
time <- tf$Variable(0)
with(tf$GradientTape() %as% outer_tape, {
  with(tf$GradientTape() %as% inner_tape, {
    position <- 4.9 * time ^ 2
  })
  speed <- inner_tape$gradient(position, time)
})
acceleration <- outer_tape$gradient(speed, time)
acceleration


## -------------------------------------------------------------------------
num_samples_per_class <- 1000
Sigma <- rbind(c(1, 0.5),
               c(0.5, 1))
negative_samples <- MASS::mvrnorm(n = num_samples_per_class,
                                  mu = c(0, 3),
                                  Sigma = Sigma)
positive_samples <- MASS::mvrnorm(n = num_samples_per_class,
                                  mu = c(3, 0),
                                  Sigma = Sigma)


## -------------------------------------------------------------------------
inputs <- rbind(negative_samples, positive_samples)


## -------------------------------------------------------------------------
targets <- rbind(array(0, dim = c(num_samples_per_class, 1)),
                 array(1, dim = c(num_samples_per_class, 1)))


## -------------------------------------------------------------------------
plot(x = inputs[, 1], y = inputs[, 2],
     col = ifelse(targets[,1] == 0, "purple", "green"))


## -------------------------------------------------------------------------
input_dim <- 2
output_dim <- 1
W <- tf$Variable(initial_value =
                   tf$random$uniform(shape(input_dim, output_dim)))
b <- tf$Variable(initial_value = tf$zeros(shape(output_dim)))


## -------------------------------------------------------------------------
model <- function(inputs)
  tf$matmul(inputs, W) + b


## -------------------------------------------------------------------------
square_loss <- function(targets, predictions) {
  per_sample_losses <- (targets - predictions)^2
  mean(per_sample_losses)
}


## -------------------------------------------------------------------------
square_loss <- function(targets, predictions) {
  per_sample_losses <- tf$square(targets - predictions)
  tf$reduce_mean(per_sample_losses)
}


## -------------------------------------------------------------------------
learning_rate <- 0.1

training_step <- function(inputs, targets) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model(inputs)
    loss <- square_loss(predictions, targets)
  })
  grad_loss_wrt <- tape$gradient(loss, list(W = W, b = b))
  W$assign_sub(grad_loss_wrt$W * learning_rate)
  b$assign_sub(grad_loss_wrt$b * learning_rate)
  loss
}


## -------------------------------------------------------------------------
inputs <- as_tensor(inputs, dtype = "float32")
for (step in seq(40)) {
  loss <- training_step(inputs, targets)
  cat(sprintf("Loss at step %s: %.4f\n", step, loss))
}


## -------------------------------------------------------------------------
predictions <- model(inputs)

inputs <- as.array(inputs)
predictions <- as.array(predictions)
plot(inputs[, 1], inputs[, 2],
     col = ifelse(predictions[, 1] <= 0.5, "purple", "green"))


## -------------------------------------------------------------------------
plot(x = inputs[, 1], y = inputs[, 2],
     col = ifelse(predictions[, 1] <= 0.5, "purple", "green"))

slope <- -W[1, ] / W[2, ]
intercept <- (0.5 - b) / W[2, ]
abline(as.array(intercept), as.array(slope), col = "red")


## -------------------------------------------------------------------------
layer_simple_dense <- new_layer_class(
  classname = "SimpleDense",

  initialize = function(units, activation = NULL) {
    super$initialize()
    self$units <- as.integer(units)
    self$activation <- activation
  },

  build = function(input_shape) {
    input_dim <- input_shape[length(input_shape)]
    self$W <- self$add_weight(shape = c(input_dim, self$units),
                              initializer = "random_normal")
    self$b <- self$add_weight(shape = c(self$units),
                              initializer = "zeros")
  },

  call = function(inputs) {
    y <- tf$matmul(inputs, self$W) + self$b
    if (!is.null(self$activation))
      y <- self$activation(y)
    y
  }
)


## -------------------------------------------------------------------------
my_dense <- layer_simple_dense(units = 32, activation = tf$nn$relu)
input_tensor <- as_tensor(1, shape = c(2, 784))
output_tensor <- my_dense(input_tensor)
output_tensor$shape


## -------------------------------------------------------------------------
layer <- layer_dense(units = 32, activation = "relu")


## -------------------------------------------------------------------------
model <- keras_model_sequential(list(
    layer_dense(units = 32, activation="relu"),
    layer_dense(units = 32)
))


## ---- eval = FALSE--------------------------------------------------------
## model <- model_naive_sequential(list(
##   layer_naive_dense(input_size = 784, output_size = 32,
##                     activation = "relu"),
##   layer_naive_dense(input_size = 32, output_size = 64,
##                     activation = "relu"),
##   layer_naive_dense(input_size = 64, output_size = 32,
##                     activation = "relu"),
##   layer_naive_dense(input_size = 32, output_size = 10,
##                     activation = "softmax")
## ))


## -------------------------------------------------------------------------
layer <- function(inputs) {
  if(!self$built) {
    self$build(inputs$shape)
    self$built <- TRUE
  }
  self$call(inputs)
}


## -------------------------------------------------------------------------
model <- keras_model_sequential(list(
  layer_simple_dense(units = 32, activation = "relu"),
  layer_simple_dense(units = 64, activation = "relu"),
  layer_simple_dense(units = 32, activation = "relu"),
  layer_simple_dense(units = 10, activation = "softmax")
))


## -------------------------------------------------------------------------
model <- keras_model_sequential()
layer_simple_dense(model, 32, activation = "relu")
layer_simple_dense(model, 64, activation = "relu")
layer_simple_dense(model, 32, activation = "relu")
layer_simple_dense(model, 10, activation = "softmax")


## -------------------------------------------------------------------------
length(model$layers)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_simple_dense(32, activation = "relu") %>%
  layer_simple_dense(64, activation = "relu") %>%
  layer_simple_dense(32, activation = "relu") %>%
  layer_simple_dense(10, activation = "softmax")


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>% layer_dense(1)
model %>% compile(optimizer = "rmsprop",
                  loss = "mean_squared_error",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = loss_mean_squared_error(),
  metrics = metric_binary_accuracy()
)


## ---- eval = FALSE--------------------------------------------------------
## model %>% compile(
##   optimizer = optimizer_rmsprop(learning_rate = 1e-4),
##   loss = my_custom_loss,
##   metrics = c(my_custom_metric_1, my_custom_metric_2)
## )


## -------------------------------------------------------------------------
ls(pattern = "^optimizer_", "package:keras")


## -------------------------------------------------------------------------
ls(pattern = "^loss_", "package:keras")


## -------------------------------------------------------------------------
ls(pattern = "^metric_", "package:keras")


## -------------------------------------------------------------------------
history <- model %>%
  fit(inputs, targets,
      epochs = 5, batch_size = 128)


## -------------------------------------------------------------------------
str(history$metrics)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(1)

model %>% compile(optimizer_rmsprop(learning_rate = 0.1),
                  loss = loss_mean_squared_error(),
                  metrics = metric_binary_accuracy())

n_cases <- dim(inputs)[1]
num_validation_samples <- round(0.3 * n_cases)
val_indices <- sample.int(n_cases, num_validation_samples)

val_inputs <- inputs[val_indices, ]
val_targets <- targets[val_indices, , drop = FALSE]
training_inputs <- inputs[-val_indices, ]
training_targets <- targets[-val_indices, , drop = FALSE]

model %>% fit(
  training_inputs,
  training_targets,
  epochs = 5,
  batch_size = 16,
  validation_data = list(val_inputs, val_targets)
)


## -------------------------------------------------------------------------
loss_and_metrics <- evaluate(model, val_inputs, val_targets,
                             batch_size = 128)


## ---- eval = FALSE--------------------------------------------------------
## predictions <- model(new_inputs)


## ---- eval = FALSE--------------------------------------------------------
## predictions <- model %>%
##   predict(new_inputs, batch_size=128)


## -------------------------------------------------------------------------
predictions <- model %>%
  predict(val_inputs, batch_size=128)
head(predictions, 10)
