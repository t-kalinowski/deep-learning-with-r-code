## ----setup, include=FALSE-------------------------------------------------
library(tensorflow)
library(keras3)
tensorflow::as_tensor(1)


## ---- eval = FALSE--------------------------------------------------------
## reticulate::py_install("keras-tuner", pip = TRUE)


## -------------------------------------------------------------------------

build_model <- function(hp, num_classes = 10) {

  units <- hp$Int(name = "units",
                  min_value = 16L, max_value = 64L, step = 16L)

  model <- keras_model_sequential() %>%
    layer_dense(units, activation = "relu") %>%
    layer_dense(num_classes, activation = "softmax")

  optimizer <- hp$Choice(name = "optimizer",
                         values = c("rmsprop", "adam"))

  model %>% compile(optimizer = optimizer,
                    loss = "sparse_categorical_crossentropy",
                    metrics = c("accuracy"))
  model
}


## -------------------------------------------------------------------------
kt <- reticulate::import("kerastuner")

SimpleMLP(kt$HyperModel) %py_class% {

  `__init__` <- function(num_classes) {
    self$num_classes <- num_classes
  }

  build <- function(hp) {
    build_model(hp, self$num_classes)
  }

}

hypermodel <- SimpleMLP(num_classes = 10)


## import kerastuner as kt
##
## class SimpleMLP(kt.HyperModel):
##    def __init__(self, num_classes):
##         self.num_classes = num_classes
##
##     def build(self, hp):
##       build_model(hp, self.num_classes)
##
## hypermodel = SimpleMLP(num_classes = 10)


## -------------------------------------------------------------------------
tuner <- kt$BayesianOptimization(
  build_model,
  objective = "val_accuracy",
  max_trials = 100L,
  executions_per_trial = 2L,
  directory = "mnist_kt_test",
  overwrite = TRUE
)


## -------------------------------------------------------------------------
tuner$search_space_summary()


## ---- eval = FALSE--------------------------------------------------------
## objective <- kt$Objective(
##   name = "val_accuracy",
##   direction = "max"
## )
##
## tuner <- kt$BayesianOptimization(
##   build_model,
##   objective = objective,
##   ...
## )


## -------------------------------------------------------------------------
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()
x_train %<>% { array_reshape(., c(-1, 28 * 28)) / 255 }
x_test  %<>% { array_reshape(., c(-1, 28 * 28)) / 255 }
x_train_full <- x_train
y_train_full <- y_train
num_val_samples <- 10000
c(x_train, x_val) %<-% list(x_train[seq(num_val_samples), ],
                            x_train[-seq(num_val_samples), ])
c(y_train, y_val) %<-% list(y_train[seq(num_val_samples)],
                            y_train[-seq(num_val_samples)])

callbacks <- c(
    callback_early_stopping(monitor="val_loss", patience=5)
)

tuner$search(
  x_train, y_train,
  batch_size = 128L,
  epochs = 100L,
  validation_data = list(x_val, y_val),
  callbacks = callbacks,
  verbose = 2L
)


## -------------------------------------------------------------------------
top_n <- 4L
best_hps <- tuner$get_best_hyperparameters(top_n)
str(best_hps)


## -------------------------------------------------------------------------
get_best_epoch <- function(hp) {
  model <- build_model(hp)

  callbacks <- c(
    callback_early_stopping(monitor = "val_loss", mode = "min",
                            patience = 10))

  history <- model %>% fit(
    x_train, y_train,
    validation_data = list(x_val, y_val),
    epochs = 100,
    batch_size = 128,
    callbacks = callbacks
  )

  best_epoch <- which.min(history$metrics$val_loss)
  print(glue::glue("Best epoch: {best_epoch}"))
  invisible(best_epoch)
}


## -------------------------------------------------------------------------
get_best_trained_model <- function(hp) {
  best_epoch <- get_best_epoch(hp)
  model <- build_model(hp)
  model %>% fit(
    x_train_full,
    y_train_full,
    batch_size = 128,
    epochs = round(best_epoch * 1.2)
  )
  model
}

best_models <- best_hps %>%
  lapply(get_best_trained_model)


## -------------------------------------------------------------------------
best_models <- tuner$get_best_models(top_n)


## ---- eval = FALSE--------------------------------------------------------
## preds_a <- model_a %>% predict(x_val)
## preds_b <- model_b %>% predict(x_val)
## preds_c <- model_c %>% predict(x_val)
## preds_d <- model_d %>% predict(x_val)
## final_preds <- 0.25 * (preds_a + preds_b + preds_c + preds_d)


## ---- eval = FALSE--------------------------------------------------------
## preds_a <- model_a %>% predict(x_val)
## preds_b <- model_b %>% predict(x_val)
## preds_c <- model_c %>% predict(x_val)
## preds_d <- model_d %>% predict(x_val)
## final_preds <-
##   0.5 * preds_a + 0.25 * preds_b + 0.1 * preds_c + 0.15 * preds_d


## ---- eval = FALSE--------------------------------------------------------
## <sign> * (2 ^ (<exponent> - 127)) * 1.<mantissa>


## -------------------------------------------------------------------------
r_array <- base::array(0, dim = c(2, 2))
tf_tensor <- tensorflow::as_tensor(r_array)
tf_tensor$dtype


## -------------------------------------------------------------------------
r_array <- base::array(0, dim = c(2, 2))
tf_tensor <- tensorflow::as_tensor(r_array, dtype = "float32")
tf_tensor$dtype


## -------------------------------------------------------------------------
keras::keras$mixed_precision$set_global_policy("mixed_float16")


## ---- eval = FALSE--------------------------------------------------------
## library(tensorflow)
## strategy <- tf$distribute$MirroredStrategy()
## cat("Number of devices:", strategy$num_replicas_in_sync, "\n")
## with(strategy$scope(), {
##   model <- get_compiled_model()
## })
## model %>% fit(
##   train_dataset,
##   epochs = 100,
##   validation_data = val_dataset,
##   callbacks = callbacks
## )


## -------------------------------------------------------------------------
build_model <- function(input_size) {
  resnet <- application_resnet50(weights = NULL,
                                 include_top = FALSE,
                                 pooling = "max")

  inputs <- layer_input(c(input_size, 3))

  outputs <- inputs %>%
    resnet_preprocess_input() %>%
    resnet() %>%
    layer_dense(10, activation = "softmax")

  model <- keras_model(inputs, outputs)

  model %>% compile(
    optimizer = "rmsprop",
    loss = "sparse_categorical_crossentropy",
    metrics = "accuracy"
  )

  model
}

strategy <- tf$distribute$MirroredStrategy()
cat("Number of replicas:", strategy$num_replicas_in_sync, "\n")

with(strategy$scope(), {
  model <- build_model(input_size = c(32, 32))
})


## ---- eval = FALSE--------------------------------------------------------
## c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_cifar10()
## model %>% fit(x_train, y_train, batch_size = 1024)


## ---- eval = FALSE--------------------------------------------------------
## tpu <- tf$distribute$cluster_resolver$TPUClusterResolver$connect()
## cat("Device:", tpu$master(), "\n")
## strategy <- tf$distribute$TPUStrategy(tpu)
## with(strategy$scope(), { ... })
