# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "tibble")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(reticulate)
library(keras3)
use_backend("jax")
py_require(c("keras-tuner", "scikit-learn"))
kt <- import("keras_tuner")


# ----------------------------------------------------------------------
top_n <- 4L


# ----------------------------------------------------------------------
build_model <- function(hp, num_classes = 10) {
  units  <- hp$Int(                                                             # <1>
    name = "units",                                                             # <1>
    min_value = 16L,                                                            # <1>
    max_value = 64L,                                                            # <1>
    step = 16L)                                                                 # <1>
  model  <- keras_model_sequential() |>
    layer_dense(units, activation = "relu") |>
    layer_dense(num_classes, activation = "softmax")

  optimizer <- hp$Choice(name = "optimizer",                                    # <2>
                         values = c("rmsprop", "adam"))                         # <2>
  model |> compile(optimizer = optimizer,
                   loss = "sparse_categorical_crossentropy",
                   metrics = "accuracy")
  model                                                                         # <3>
}

get_best_epoch <- function(hp) {
  model <- build_model(hp)
  callbacks <- list(
    callback_early_stopping(
      monitor = "val_loss",
      mode = "min",
      patience = 10                                                             # <1>
    )
  )

  history <- model |> fit(
    x_train, y_train,
    validation_data = list(x_val, y_val),
    epochs = 100,
    batch_size = 128,
    callbacks = callbacks
  )

  best_epoch <- which.min(history$metrics$val_loss)
  cat(sprintf("Best epoch: %d\n", best_epoch))
  best_epoch
}

get_best_trained_model <- function(hp) {
    best_epoch <- get_best_epoch(hp)
    model <- build_model(hp)
    model |> fit(
        x_train_full, y_train_full,
        batch_size=128L, epochs=as.integer(best_epoch * 1.2))
    model
}


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
config_set_dtype_policy("float16")


# ----------------------------------------------------------------------
config_set_dtype_policy("mixed_float16")


# ----------------------------------------------------------------------
optimizer <- optimizer_adam(learning_rate = 1e-3, loss_scale_factor = 10)


# ----------------------------------------------------------------------
optimizer <-
  optimizer_adam(learning_rate = 1e-3) |>
  optimizer_loss_scale()


# ----------------------------------------------------------------------
x <- op_array(rbind(c(0.1, 0.9), c(1.2, -0.8)))
kernel <- op_array(rbind(c(-0.1, -2.2), c(1.1, 0.7)))


# ----------------------------------------------------------------------
abs_max_quantize <- function(value) {
  abs_max <- op_max(op_abs(value), keepdims = TRUE)                             # <1>
  scale <- op_divide(127, abs_max + 1e-7)                                       # <2>
  scaled_value <- value * scale                                                 # <3>
  scaled_value <- op_clip(op_round(scaled_value), -127, 127)                    # <4>
  scaled_value <- op_cast(scaled_value, dtype = "int8")                         # <5>
  list(scaled_value, scale)
}

.[int_x, x_scale] <- abs_max_quantize(x)
.[int_kernel, kernel_scale] <- abs_max_quantize(kernel)


# ----------------------------------------------------------------------
int_y <- op_matmul(int_x, int_kernel)
y <- op_cast(int_y, dtype = "float32") / (x_scale * kernel_scale)


# ----------------------------------------------------------------------
y
op_matmul(x, kernel)


# ----------------------------------------------------------------------
#| eval: false
# model <- ...                                                                    # <1>
# model |> quantize_weights("int8")                                               # <2>
# predictions <- model |> predict(...)                                            # <3>


