# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "tibble")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
# Sys.setenv("XLA_FLAGS" = "--xla_force_host_platform_device_count=8")
library(reticulate)
library(keras3)
use_backend("jax")
py_require(c("keras-tuner", "scikit-learn"))


# ----------------------------------------------------------------------
library(keras3)
library(reticulate)
use_backend("jax")
py_require(c("keras-tuner", "scikit-learn"))


# ----------------------------------------------------------------------
#| lst-cap: KerasTuner model-building function
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


# ----------------------------------------------------------------------
#| lst-cap: "KerasTuner `HyperModel`"
kt <- import("keras_tuner")

SimpleMLP(kt$HyperModel) %py_class%  {
  `__init__` <- function(self, num_classes) {                                   # <1>
    self.num_classes = num_classes
  }

  build <- function(self, hp) {
    build_model(hp, self$num_classes)
  }
}

hypermodel <- SimpleMLP(num_classes=10)


# ----------------------------------------------------------------------
tuner <- kt$BayesianOptimization(
  build_model,                                                                  # <1>
  objective = "val_accuracy",                                                   # <2>
  max_trials = 20L,                                                             # <3>
  executions_per_trial = 2L,                                                    # <4>
  directory = "mnist_kt_test",                                                  # <5>
  overwrite = TRUE                                                              # <6>
)


# ----------------------------------------------------------------------
tuner$search_space_summary()


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
tuner$search(                                                                   # <1>
  x_train, y_train,
  batch_size = 128L,
  epochs = 100L,
  validation_data = list(x_val, y_val),
  callbacks = callbacks,
  verbose = 2L
)


# ----------------------------------------------------------------------
#| lst-cap: Querying the best hyperparameter configurations
top_n <- 4L
best_hps <- tuner$get_best_hyperparameters(top_n)                               # <1>


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
best_models <- tuner$get_best_models(top_n)


# ----------------------------------------------------------------------
#| eval: false
# preds_a  <- model_a |> predict(x_val)                                           # <1>
# preds_b  <- model_b |> predict(x_val)                                           # <1>
# preds_c  <- model_c |> predict(x_val)                                           # <1>
# preds_d  <- model_d |> predict(x_val)
# final_preds <- 0.25 * (preds_a + preds_b + preds_c + preds_d)                   # <2>


# ----------------------------------------------------------------------
#| eval: false
# 
# preds_a  <- model_a |> predict(x_val)
# preds_b  <- model_b |> predict(x_val)
# preds_c  <- model_c |> predict(x_val)
# preds_d  <- model_d |> predict(x_val)
# final_preds <- .5*preds_a + .25*preds_b + .1*preds_c + .15*preds_d              # <1>


