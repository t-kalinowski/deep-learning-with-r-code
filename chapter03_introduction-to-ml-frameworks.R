


# -----------------


library(tensorflow)
library(keras3)
use_backend("tensorflow")


tf$ones(shape = shape(2, 2))
tf$zeros(shape = shape(2, 2))
tf$constant(c(1, 2, 3), dtype = "float32")


x <- tf$random$normal(shape = shape(3, 1), mean = 0, stddev = 1)                # <1>
x
x <- tf$random$normal(shape = shape(3, 1), mean = 0., stddev = 1.)              # <2>
x


x <- array(1, dim = c(2, 2))
x[1, 1] <- 0


try({
x <- tf$ones(shape(2, 2))
x@r[1, 1] <- 0.                                                                 # <1>
})


v <- tf$Variable(initial_value = tf$zeros(shape = shape(3, 1)))
v


v$assign(tf$ones(shape(3, 1)))


v


v@r[] <- tf$zeros(shape(3, 1))
v


v@r[1, 1]$assign(3)
v@r[2, 1] <- 4


v


v$assign_add(tf$ones(shape(3, 1)))


a <- tf$ones(shape(2, 2))
b <- tf$square(a)                                                               # <1>
c <- tf$sqrt(a)                                                                 # <2>
d <- b + c                                                                      # <3>
e <- tf$matmul(a, b)                                                            # <4>
f <- tf$concat(c(a, b), axis = 0L)                                              # <5>


dense <- function(inputs, W, b) {
  tf$nn$relu(tf$matmul(inputs, W) + b)
}


input_var <- tf$Variable(initial_value = 3)
with(tf$GradientTape() %as% tape, {
  result <- tf$square(input_var)
})
gradient <- tape$gradient(result, input_var)


input_const <- tf$constant(3)
with(tf$GradientTape() %as% tape, {
  tape$watch(input_const)
  result <- tf$square(input_const)
})
gradient <- tape$gradient(result, input_const)


time <- tf$Variable(0)
with(tf$GradientTape() %as% outer_tape, {
  with(tf$GradientTape() %as% inner_tape, {
    position <- 4.9 * time^2
  })
  speed <- inner_tape$gradient(position, time)
})
acceleration <- outer_tape$gradient(speed, time)
acceleration


dense <- tf_function(\(inputs, W, b) {
  tf$nn$relu(tf$matmul(inputs, W) + b)
})


dense <- tf_function(jit_compile = TRUE, \(inputs, W, b) {
  tf$nn$relu(tf$matmul(inputs, W) + b)
})


num_samples_per_class <- 1000
Sigma <- rbind(c(1, 0.5),
               c(0.5, 1))
negative_samples <- MASS::mvrnorm(                                              # <1>
  n = num_samples_per_class,                                                    # <1>
  mu = c(0, 3),                                                                 # <1>
  Sigma = Sigma                                                                 # <1>
)                                                                               # <1>
positive_samples <- MASS::mvrnorm(                                              # <2>
  n = num_samples_per_class,                                                    # <2>
  mu = c(3, 0),                                                                 # <2>
  Sigma = Sigma                                                                 # <2>
)


inputs <- rbind(negative_samples, positive_samples)


targets <- rbind(array(0, dim = c(num_samples_per_class, 1)),
                 array(1, dim = c(num_samples_per_class, 1)))


if(basename(getwd()) != "manuscript") setwd("manuscript")
save(inputs, targets, file = "ch3-training-data.RData")


plot(x = inputs[, 1], y = inputs[, 2],
     col = ifelse(targets[, 1] == 0, "purple", "green"))


input_dim <- 2                                                                  # <1>
output_dim <- 1                                                                 # <2>
W <- tf$Variable(
  initial_value = tf$random$uniform(shape(input_dim, output_dim))
)
b <- tf$Variable(
  initial_value = tf$zeros(shape(output_dim))
)


model <- function(inputs, W, b) {
  tf$matmul(inputs, W) + b
}


mean_squared_error <- function(targets, predictions) {
  per_sample_losses <- tf$square(targets - predictions)                         # <1>
  tf$reduce_mean(per_sample_losses)                                             # <2>
}


learning_rate <- 0.1

training_step <- tf_function(jit_compile = TRUE,                                # <1>
                             \(inputs, targets, W, b) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model(inputs, W, b)                                          # <2>
    loss <- mean_squared_error(predictions, targets)                            # <2>
  })
  grad_loss_wrt <- tape$gradient(loss, list(W = W, b = b))                      # <3>
  W$assign_sub(grad_loss_wrt$W * learning_rate)                                 # <4>
  b$assign_sub(grad_loss_wrt$b * learning_rate)                                 # <4>
  loss
})


inputs <- np_array(inputs, dtype = "float32")
targets <- np_array(targets, dtype = "float32")

for (step in 1:40) {
  loss <- training_step(inputs, targets, W, b)
  cat(sprintf("Loss at step %d: %.4f\n", step, loss))
}


predictions <- model(inputs, W, b)

predictions <- as.array(predictions)                                            # <1>
inputs <- as.array(inputs)                                                      # <1>
targets <- as.array(targets)                                                    # <1>

plot(x = inputs[, 1], y = inputs[, 2],
     col = ifelse(predictions[, 1] > 0.5, "green", "purple"))


plot(x = inputs[, 1], y = inputs[, 2],
     col = ifelse(predictions[, 1] <= 0.5, "purple", "green"))

slope <- -W[1,] / W[2,]
intercept <- (0.5 - b) / W[2,]
abline(as.array(intercept), as.array(slope), col = "red")


# -----------------


library(reticulate)
library(keras3)
use_backend("torch")

torch <- import("torch")


torch$ones(size = shape(2, 2))                                                  # <1>
torch$zeros(size = shape(2, 2))
torch$tensor(c(1, 2, 3), dtype = torch$float32)                                 # <2>


torch$normal(                                                                   # <1>
  mean = torch$zeros(size = shape(3, 1)),                                       # <1>
  std = torch$ones(size = shape(3, 1))                                          # <1>
)


torch$rand(3L, 1L)


x <- torch$zeros(size = shape(2, 1))
x@py[0, 0] <- 1
x


x <- torch$zeros(shape(2, 1))
p <- torch$nn$parameter$Parameter(data = x)                                     # <1>


a <- torch$ones(shape(2, 2))
b <- torch$square(a)                                                            # <1>
c <- torch$sqrt(a)                                                              # <2>
d <- b + c                                                                      # <3>
e <- torch$matmul(a, b)                                                         # <4>
f <- torch$cat(list(a, b), axis = 0L)                                           # <5>


dense <- function(inputs, W, b) {
  torch$nn$relu(torch$matmul(inputs, W) + b)
}


input_var <- torch$tensor(3.0, requires_grad = TRUE)                            # <1>
result <- torch$square(input_var)
result$backward()                                                               # <2>
gradient <- input_var$grad                                                      # <2>
gradient


result <- torch$square(input_var)
result$backward()
input_var$grad                                                                  # <1>


input_var$grad <- NULL


input_dim <- 2L
output_dim <- 1L

W <- torch$rand(input_dim, output_dim, requires_grad = TRUE)
b <- torch$zeros(output_dim, requires_grad = TRUE)


model <- function(inputs, W, b) {
  torch$matmul(inputs, W) + b
}


mean_squared_error <- function(targets, predictions) {
  per_sample_losses <- torch$square(targets - predictions)
  torch$mean(per_sample_losses)
}


learning_rate <- 0.1

training_step <- function(inputs, targets, W, b) {
  predictions <- model(inputs)                                                  # <1>
  loss <- mean_squared_error(targets, predictions)                              # <1>
  loss$backward()                                                               # <2>
  grad_loss_wrt_W <- W$grad                                                     # <3>
  grad_loss_wrt_b <- b$grad                                                     # <3>
  with(torch$no_grad(), {
    W$sub_(grad_loss_wrt_W * learning_rate)                                     # <4>
    b$sub_(grad_loss_wrt_b * learning_rate)                                     # <4>
  })
  W$grad <- b$grad <- NULL                                                      # <5>
  loss
}


LinearModel(torch$nn$Module) %py_class% {

  `__init__` <- function(self) {
    super()$initialize()
    self$W <- torch$nn$Parameter(torch$rand(input_dim, output_dim))
    self$b <- torch$nn$Parameter(torch$zeros(output_dim))
  }

  forward <- function(self, inputs) {
    torch$matmul(inputs, self$W) + self$b
  }
}


model <- LinearModel()


if(basename(getwd()) != "manuscript") setwd("manuscript")
load("ch3-training-data.RData") # loads `inputs`, `targets`


torch_inputs <- torch$tensor(inputs, dtype = torch$float32)
output <- model(torch_inputs)


optimizer <- torch$optim$SGD(model$parameters(), lr = learning_rate)


training_step <- function(inputs, targets) {
  predictions <- model(inputs)
  loss <- mean_squared_error(targets, predictions)
  loss$backward()
  optimizer$step()
  model$zero_grad()
  loss
}


compiled_model <- model$compile()


dense <- torch$compile(function(inputs, W, b) {
  torch$nn$relu(torch$matmul(inputs, W) + b)
})


# -----------------


library(reticulate)
library(keras3)
use_backend("jax")

jax <- import("jax")
jnp <- import("jax.numpy")


jnp$ones(shape = shape(2, 2))
jnp$zeros(shape = shape(2, 2))
jnp$array(c(1, 2, 3), dtype = "float32")


runif(3)
runif(3)


seed_key <- jax$random$key(1337L)


seed_key <- jax$random$key(0L)
jax$random$normal(seed_key, shape = shape(3))


seed_key <- jax$random$key(123L)
jax$random$normal(seed_key, shape = shape(3))
jax$random$normal(seed_key, shape = shape(3))


seed_key <- jax$random$key(123L)
jax$random$normal(seed_key, shape = shape(3))
new_seed_key <- jax$random$split(seed_key, num = 1L)[0]                         # <1>
jax$random$normal(new_seed_key, shape = shape(3))


x <- jnp$array(c(1, 2, 3), dtype = "float32")
new_x <- x$at[0]$set(10)
new_x


x@r[1] <- 20
x


a <- jnp$ones(shape(2, 2))
b <- jnp$square(a)                                                              # <1>
c <- jnp$sqrt(a)                                                                # <2>
d <- b + c                                                                      # <3>
e <- jnp$matmul(a, b)                                                           # <4>
e <- e * d                                                                      # <5>


dense <- function(inputs, W, b) {
  jax$nn$relu(jnp$matmul(inputs, W) + b)
}


compute_loss <- function(input_var) {
  jnp$square(input_var)
}


grad_fn <- jax$grad(compute_loss)


input_var <- jnp$array(3)
grad_of_loss_wrt_input_var <- grad_fn(input_var)


grad_fn <- jax$value_and_grad(compute_loss)
.[output, grad_of_loss_wrt_input_var] <- grad_fn(input_var)


dense <- jax$jit(\(inputs, W, b) {
  jax$nn$relu(jnp$matmul(inputs, W) + b)
})


model <- function(inputs, W, b) {
  jnp$matmul(inputs, W) + b
}

mean_squared_error <- function(targets, predictions) {
  jnp$mean(jnp$square(targets - predictions))
}


learning_rate = 0.1

compute_loss <- function(state, inputs, targets) {
  .[W, b] <- state
  predictions <- model(inputs, W, b)
  mean_squared_error(targets, predictions)
}


grad_fn <- jax$value_and_grad(compute_loss)


learning_rate <- 0.1

training_step <- jax$jit(\(inputs, targets, W, b) {                             # <1>
  .[loss, grads] <- grad_fn(list(W, b), inputs, targets)                        # <2>
  .[grad_loss_wrt_W, grad_loss_wrt_b] <- grads
  W <- W - grad_loss_wrt_W * learning_rate                                      # <3>
  b <- b - grad_loss_wrt_b * learning_rate                                      # <3>
  list(loss, W, b)                                                              # <4>
})


if(basename(getwd()) != "manuscript") setwd("manuscript")
load("ch3-training-data.RData") # loads `inputs`, `targets`


input_dim <- 2
output_dim <- 1

W <- jnp$array(array(runif(input_dim * output_dim),
                     dim = c(input_dim, output_dim)))
b <- jnp$array(array(0, dim = c(output_dim)))
state <- list(W, b)
for (step in 1:40) {
  .[loss, W, b] <- training_step(inputs, targets, W, b)
  cat(sprintf("Loss at step %d: %.4f\n", step, loss))
}


# -----------------


library(keras3)
use_backend("jax")


layer_simple_dense <- new_layer_class(
  classname = "SimpleDense",
  initialize = function(units, activation = NULL) {
    super$initialize()
    self$units <- units
    self$activation <- activation
  },
  build = function(input_shape) {                                               # <1>
    .[batch_dim, input_dim] <- input_shape
    self$W <- self$add_weight(shape(input_dim, self$units),                     # <2>
                              initializer = "random_normal")                    # <2>
    self$b <- self$add_weight(shape(self$units), initializer = "zeros")         # <2>
  },
  call = function(inputs) {                                                     # <3>
    y <- op_matmul(inputs, self$W) + self$b
    if (!is.null(self$activation)) {
      y <- self$activation(y)
    }
    y
  }
)


my_dense <- layer_simple_dense(units = 32, activation = op_relu)                # <1>
input_tensor <- op_ones(shape = shape(2, 784))                                  # <2>
output_tensor <- my_dense(input_tensor)                                         # <3>
op_shape(output_tensor)


layer <- layer_dense(units = 32, activation = "relu")                           # <1>


model <- keras_model_sequential() |>
  layer_dense(units = 32, activation = "relu") |>
  layer_dense(units = 32)


`__call__` <- function(inputs) {
  if (!self$built) {
    self$build(op_shape(inputs))
    self$built <- TRUE
  }
  self$call(inputs)
}


model <- keras_model_sequential() |>
  layer_simple_dense(units = 32, activation = "relu") |>
  layer_simple_dense(units = 64, activation = "relu") |>
  layer_simple_dense(units = 32, activation = "relu") |>
  layer_simple_dense(units = 10, activation = "softmax")


model <- keras_model_sequential() |> layer_dense(units = 1)                     # <1>
model |> compile(optimizer = "rmsprop",                                         # <2>
                 loss = "mean_squared_error",                                   # <3>
                 metrics = c("accuracy"))                                       # <4>


model |> compile(
  optimizer = optimizer_rmsprop(),
  loss = loss_mean_squared_error(),
  metrics = list(metric_binary_accuracy())
)


if(basename(getwd()) != "manuscript") setwd("manuscript")
load("ch3-training-data.RData") # loads `inputs`, `targets`


history <- model |> fit(
  inputs,                                                                       # <1>
  targets,                                                                      # <2>
  epochs = 5,                                                                   # <3>
  batch_size = 128                                                              # <4>
)


str(history$metrics)


tibble::as_tibble(history)


model <- keras_model_sequential() |> layer_dense(units = 1)

model |> compile(optimizer = optimizer_rmsprop(learning_rate = 0.1),
                 loss = loss_mean_squared_error(),
                 metrics = list(metric_binary_accuracy()))

indices_permutation <- sample.int(nrow(inputs))                                 # <1>
shuffled_inputs <- inputs[indices_permutation, , drop = FALSE]                  # <1>
shuffled_targets <- targets[indices_permutation, , drop = FALSE]                # <1>

num_validation_samples <- as.integer(0.3 * nrow(inputs))                        # <2>
val_inputs <- shuffled_inputs[1:num_validation_samples, ]                       # <2>
val_targets <- shuffled_targets[1:num_validation_samples, ]                     # <2>
training_inputs <- shuffled_inputs[-(1:num_validation_samples), ]               # <2>
training_targets <- shuffled_targets[-(1:num_validation_samples), ]             # <2>

model |> fit(
  training_inputs, training_targets,                                            # <3>
  epochs = 5, batch_size = 16,
  validation_data = list(val_inputs, val_targets)                               # <4>
)


loss_and_metrics <- evaluate(model, val_inputs, val_targets, batch_size = 128)


predictions <- model |> predict(val_inputs, batch_size = 128)
head(predictions)


# -----------------


