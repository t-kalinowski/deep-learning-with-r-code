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
#| lst-cap: "A large, densely connected model"
model <- keras_model_sequential(input_shape = c(16000)) |>
  layer_dense(64000, activation = "relu") |>
  layer_dense(8000, activation = "sigmoid")


# ----------------------------------------------------------------------
#| eval: false
# .[half_kernel_0, half_kernel_1] <- op_split(kernel, 2)
# .[half_bias_0, half_bias_1] <- op_split(bias, 2)
# 
# with(keras$device("gpu:0"), {
#   half_output_0 <- op_matmul(inputs, half_kernel_0) + half_bias_0
# })
# 
# with(keras$device("gpu:1"), {
#   half_output_1 <- op_matmul(inputs, half_kernel_1) + half_bias_1
# })


# ----------------------------------------------------------------------
#| eval: false
# keras$distribution$set_distribution(
#   keras$distribution$DataParallel()
# )


# ----------------------------------------------------------------------
keras$distribution$list_devices()


# ----------------------------------------------------------------------
#| eval: false
# keras$distribution$set_distribution(
#   keras$distribution$DataParallel(devices = c("gpu:0", "gpu:1"))
# )


# ----------------------------------------------------------------------
#| eval: false
# device_mesh <- keras$distribution$DeviceMesh(
#   shape = shape(2, 4),                                                          # <1>
#   axis_names = c("data", "model")                                               # <2>
# )


# ----------------------------------------------------------------------
devices <- paste0("gpu:", 0:7)
device_mesh <- keras$distribution$DeviceMesh(
  shape = shape(2, 4),
  axis_names = c("data", "model"),
  devices = devices
)


# ----------------------------------------------------------------------
#| eval: false
# list(
#   "sequential/dense_1/kernel" = tuple(NULL, "model"),                           # <1>
#   "sequential/dense_1/bias" = tuple("model"),                                   # <2>
#   ...
# )


# ----------------------------------------------------------------------
for (v in model$variables)
  cat(v$path, "\n")


# ----------------------------------------------------------------------
layout_map <- keras$distribution$LayoutMap(device_mesh)
layout_map["sequential/dense/kernel"]   <- tuple(NULL, "model")
layout_map["sequential/dense/bias"]     <- tuple("model")
layout_map["sequential/dense_1/kernel"] <- tuple(NULL, "model")
layout_map["sequential/dense_1/bias"]   <- tuple("model")


# ----------------------------------------------------------------------
model_parallel <- keras$distribution$ModelParallel(
  layout_map = layout_map,
  batch_dim_name = "data"                                                       # <1>
)
keras$distribution$set_distribution(model_parallel)


# ----------------------------------------------------------------------
model$layers[[1]]$kernel$value$sharding


# ----------------------------------------------------------------------
#| eval: false
# jax <- reticulate::import("jax")
# value <- model$layers[[1]]$kernel$value
# jax$debug$visualize_sharding(value$shape, value$sharding)


# ----------------------------------------------------------------------
#| eval: false
# model |> compile(..., steps_per_execution = 8)


