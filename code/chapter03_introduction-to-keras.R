# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "tibble")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
#| lst-cap: A simple dense layer from scratch in Keras.
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


# ----------------------------------------------------------------------
my_dense <- layer_simple_dense(units = 32, activation = op_relu)                # <1>
input_tensor <- op_ones(shape = shape(2, 784))                                  # <2>
output_tensor <- my_dense(input_tensor)                                         # <3>
op_shape(output_tensor)


# ----------------------------------------------------------------------
layer <- layer_dense(units = 32, activation = "relu")                           # <1>


# ----------------------------------------------------------------------
model <- keras_model_sequential() |>
  layer_dense(units = 32, activation = "relu") |>
  layer_dense(units = 32)


# ----------------------------------------------------------------------
#| eval: false
# model <- keras_model_sequential() |>
#   layer_naive_dense(input_size = 784, output_size = 32, activation = "relu") |>
#   layer_naive_dense(input_size = 32, output_size = 64, activation = "relu") |>
#   layer_naive_dense(input_size = 64, output_size = 32, activation = "relu") |>
#   layer_naive_dense(input_size = 32, output_size = 10, activation = "softmax")


# ----------------------------------------------------------------------
`__call__` <- function(inputs) {
  if (!self$built) {
    self$build(op_shape(inputs))
    self$built <- TRUE
  }
  self$call(inputs)
}


# ----------------------------------------------------------------------
model <- keras_model_sequential() |>
  layer_simple_dense(units = 32, activation = "relu") |>
  layer_simple_dense(units = 64, activation = "relu") |>
  layer_simple_dense(units = 32, activation = "relu") |>
  layer_simple_dense(units = 10, activation = "softmax")


# ----------------------------------------------------------------------
model <- keras_model_sequential() |> layer_dense(units = 1)                     # <1>
model |> compile(
  optimizer = "rmsprop",                                                        # <2>
  loss = "mean_squared_error",                                                  # <3>
  metrics = c("accuracy")                                                       # <4>
)


# ----------------------------------------------------------------------
model |> compile(
  optimizer = optimizer_rmsprop(),
  loss = loss_mean_squared_error(),
  metrics = list(metric_binary_accuracy())
)


# ----------------------------------------------------------------------
#| eval: false
# model |> compile(
#   optimizer = optimizer_rmsprop(learning_rate = 1e-4),
#   loss = my_custom_loss,
#   metrics = c(my_custom_metric_1, my_custom_metric_2)
# )


# ----------------------------------------------------------------------
num_samples_per_class <- 1000
Sigma <- rbind(c(1, 0.5),
               c(0.5, 1))
negative_samples <- MASS::mvrnorm(
  n = num_samples_per_class,
  mu = c(0, 3),
  Sigma = Sigma
)
positive_samples <- MASS::mvrnorm(
  n = num_samples_per_class,
  mu = c(3, 0),
  Sigma = Sigma
)
inputs <- rbind(negative_samples, positive_samples)
targets <- rbind(
  array(0, dim = c(num_samples_per_class, 1)),
  array(1, dim = c(num_samples_per_class, 1))
)


# ----------------------------------------------------------------------
#| lst-cap: "Calling `fit` with data as R Arrays"
history <- model |> fit(
  inputs,                                                                       # <1>
  targets,                                                                      # <2>
  epochs = 5,                                                                   # <3>
  batch_size = 128                                                              # <4>
)


# ----------------------------------------------------------------------
str(history$metrics)


# ----------------------------------------------------------------------
tibble::as_tibble(history)


# ----------------------------------------------------------------------
#| lst-cap: Using the validation data argument
model <- keras_model_sequential() |> layer_dense(units = 1)

model |> compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.1),
  loss = loss_mean_squared_error(),
  metrics = list(metric_binary_accuracy())
)

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


# ----------------------------------------------------------------------
#| eval: false
# predictions <- model(new_inputs)                                                # <1>


# ----------------------------------------------------------------------
#| eval: false
# predictions <- predict(model, new_inputs, batch_size = 128)                     # <1>


# ----------------------------------------------------------------------
predictions <- model |> predict(val_inputs, batch_size = 128)
head(predictions)


