Sys.unsetenv("CUDA_VISIBLE_DEVICES")
Sys.setenv("XLA_FLAGS" = "--xla_force_host_platform_device_count=8")
library(reticulate)
library(keras3)
use_backend("jax")
py_require(c("keras-tuner", "scikit-learn"))


library(keras3)
library(reticulate)
use_backend("jax")
py_require(c("keras-tuner", "scikit-learn"))


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


kt <- import("keras_tuner")

SimpleMLP(kt$HyperModel) %py_class%  {
  `__init__` <- function(self, num_classes) {                                   # <2>
    self.num_classes = num_classes
  }

  build <- function(self, hp) {
    build_model(hp, self$num_classes)
  }
}

hypermodel <- SimpleMLP(num_classes=10)


tuner <- kt$BayesianOptimization(
  build_model,                                                                  # <1>
  objective = "val_accuracy",                                                   # <2>
  max_trials = 100L,                                                            # <3>
  executions_per_trial = 2L,                                                    # <4>
  directory = "mnist_kt_test",                                                  # <5>
  overwrite = TRUE                                                              # <6>
)


tuner$search_space_summary()


.[.[x_train_full, y_train_full], .[x_test, y_test]] <- dataset_mnist()          # <1>
x_train_full <- x_train_full |> array_reshape(c(-1, 28 * 28))                   # <1>
x_train_full <- x_train_full / 255                                              # <1>
x_test <- x_test |> array_reshape(c(-1, 28 * 28))                               # <1>
x_test <- x_test / 255                                                          # <1>

num_val_samples <- 10000                                                        # <2>
val_i <- seq_len(num_val_samples)
x_val <- x_train_full[val_i, ]                                                  # <2>
x_train <- x_train_full[-val_i, ]                                               # <2>
y_val <- y_train_full[val_i] |> as.matrix()                                     # <2>
y_train <- y_train_full[-val_i] |> as.matrix()                                  # <2>

callbacks <- list(                                                              # <3>
  callback_early_stopping(monitor = "val_loss", patience = 5)                   # <3>
)


top_n <- 4L
best_hps <- tuner$get_best_hyperparameters(top_n)                               # <1>


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

  best_epoch <- history |>
    tibble::as_tibble() |>
    dplyr::filter(metric == "loss" & value == min(value)) |>
    _$epoch
  glue::glue("Best epoch: {best_epoch}")
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

best_models <- py_eval("[]", convert = FALSE)
for (hp in best_hps) {
  model <- get_best_trained_model(hp)
  model |> evaluate(x_test, y_test) |> print()
  best_models$append(model)
}


best_models <- tuner$get_best_models(top_n)


model <- keras_model_sequential(input_shape = c(16000)) |>
  layer_dense(64000, activation = "relu") |>
  layer_dense(8000, activation = "sigmoid")


keras$distribution$list_devices()


devices <- paste0("gpu:", 0:7)
device_mesh <- keras$distribution$DeviceMesh(
  shape = shape(2, 4),
  axis_names = c("data", "model"),
  devices = devices
)


for (v in model$variables)
  cat(v$path, "\n")


layout_map <- keras$distribution$LayoutMap(device_mesh)
layout_map["sequential/dense/kernel"]   <- tuple(NULL, "model")
layout_map["sequential/dense/bias"]     <- tuple("model")
layout_map["sequential/dense_1/kernel"] <- tuple(NULL, "model")
layout_map["sequential/dense_1/bias"]   <- tuple("model")


model_parallel <- keras$distribution$ModelParallel(
    layout_map=layout_map,
    batch_dim_name="data"                                                       # <1>
)
keras$distribution$set_distribution(model_parallel)


model$layers[[1]]$kernel$value$sharding


jax <- reticulate::import("jax")
value <- model$layers[[1]]$kernel$value
jax$debug$visualize_sharding(value$shape, value$sharding)


keras$distribution$set_distribution(
  keras$distribution$DataParallel(devices = c("gpu:0", "gpu:1"))
)


config_set_dtype_policy("float16")


config_set_dtype_policy("mixed_float16")


optimizer <- optimizer_adam(learning_rate = 1e-3, loss_scale_factor = 10)


optimizer <-
  optimizer_adam(learning_rate = 1e-3) |>
  optimizer_loss_scale()


x <- op_array(rbind(c(0.1, 0.9), c(1.2, -0.8)))
kernel <- op_array(rbind(c(-0.1, -2.2), c(1.1, 0.7)))


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


int_y <- op_matmul(int_x, int_kernel)
y <- op_cast(int_y, dtype = "float32") / (x_scale * kernel_scale)


y
op_matmul(x, kernel)



