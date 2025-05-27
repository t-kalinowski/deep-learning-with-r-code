## ----setup, include = FALSE-----------------------------------------------
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
library(keras3)
use_virtualenv("r-reticulate")
# Sys.unsetenv("RETICULATE_PYTHON")
# Sys.setenv("LD_DEBUG" = "libcudnn.so.8")
# use_condaenv("tf")
#
#
# Sys.getenv("LD_LIBRARY_PATH")
#
# Sys.setenv("LD_LIBRARY_PATH" = paste0(c(
#   "/home/tomasz/.local/share/r-miniconda/envs/tf/lib/",
#   "/home/tomasz/.local/share/r-miniconda/envs/tf/lib/python3.9/site-packages/nvidia/cudnn/lib"
# ), collapse = ":"))

  # "/home/tomasz/.virtualenvs/r-reticulate/lib/python3.9/site-packages/nvidia/cudnn/lib")
tensorflow::tf_function(function(x) x + 1)(1)


## -------------------------------------------------------------------------
library(keras3)

model <- keras_model_sequential() %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(10, activation = "softmax")


## -------------------------------------------------------------------------
model <- keras_model_sequential()
model %>% layer_dense(64, activation="relu")
model %>% layer_dense(10, activation="softmax")


## ---- error = TRUE--------------------------------------------------------
# model$weights


## -------------------------------------------------------------------------
model$build(input_shape = shape(NA, 3))
str(model$weights)


## -------------------------------------------------------------------------
model


## -------------------------------------------------------------------------
model = keras_model_sequential(name = "my_example_model")
model %>% layer_dense(64, activation = "relu", name = "my_first_layer")
model %>% layer_dense(10, activation = "softmax", name = "my_last_layer")
model$build(shape(NA, 3))
model


## -------------------------------------------------------------------------
model <-
  keras_model_sequential(input_shape = c(3)) %>%
  layer_dense(64, activation="relu")


## -------------------------------------------------------------------------
model
model %>% layer_dense(10, activation="softmax")
model


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(3), name = "my_input")
features <- inputs %>% layer_dense(64, activation = "relu")
outputs <- features %>% layer_dense(10, activation = "softmax")
model <- keras_model(inputs = inputs, outputs = outputs)


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(3), name = "my_input")


## -------------------------------------------------------------------------
inputs$shape
inputs$dtype


## -------------------------------------------------------------------------
features <- inputs %>% layer_dense(64, activation="relu")


## ---- eval = FALSE--------------------------------------------------------
## layer_instance <- layer_dense(units = 64, activation = "relu")
## features <- layer_instance(inputs)


## ---- eval = FALSE--------------------------------------------------------
## layer_instance <- layer_dense(units = 64, activation = "relu")
## model$add(layer_instance)


## -------------------------------------------------------------------------
features$shape


## -------------------------------------------------------------------------
dim(features)


## -------------------------------------------------------------------------
outputs <- layer_dense(features, 10, activation="softmax")
model <- keras_model(inputs = inputs, outputs = outputs)


## -------------------------------------------------------------------------
model


## -------------------------------------------------------------------------
vocabulary_size <- 10000
num_tags <- 100
num_departments <- 4

title     <- layer_input(shape = c(vocabulary_size), name = "title")
text_body <- layer_input(shape = c(vocabulary_size), name = "text_body")
tags      <- layer_input(shape = c(num_tags),        name = "tags")

features <- layer_concatenate(list(title, text_body, tags)) %>%
  layer_dense(64, activation="relu")

priority <- features %>%
  layer_dense(1, activation = "sigmoid", name = "priority")

department <- features %>%
  layer_dense(num_departments, activation = "softmax", name = "department")

model <- keras_model(
  inputs = list(title, text_body, tags),
  outputs = list(priority, department)
)


## -------------------------------------------------------------------------
num_samples <- 1280

random_uniform_array <- function(dim)
  array(runif(prod(dim)), dim)

random_vectorized_array <- function(dim)
  array(sample(0:1, prod(dim), replace = TRUE), dim)

title_data     <- random_vectorized_array(c(num_samples, vocabulary_size))
text_body_data <- random_vectorized_array(c(num_samples, vocabulary_size))
tags_data      <- random_vectorized_array(c(num_samples, num_tags))

priority_data    <- random_vectorized_array(c(num_samples, 1))
department_data  <- random_vectorized_array(c(num_samples, num_departments))

model %>% compile(
  optimizer = "rmsprop",
  loss = c("mean_squared_error", "categorical_crossentropy"),
  metrics = c("mean_absolute_error", "accuracy")
)

model %>% fit(
  x = list(title_data, text_body_data, tags_data),
  y = list(priority_data, department_data),
  epochs = 1
)
model %>% evaluate(x = list(title_data, text_body_data, tags_data),
                   y = list(priority_data, department_data))

c(priority_preds, department_preds) %<-% {
   model %>% predict(list(title_data, text_body_data, tags_data))
}


## -------------------------------------------------------------------------
model %>%
  compile(optimizer = "rmsprop",
          loss = c(priority = "mean_squared_error",
                   department = "categorical_crossentropy"),
          metrics = c(priority = "mean_absolute_error",
                      department = "accuracy"))

model %>%
  fit(list(title = title_data,
           text_body = text_body_data,
           tags = tags_data),
      list(priority = priority_data,
           department = department_data), epochs = 1)

model %>%
  evaluate(list(title = title_data,
                text_body = text_body_data,
                tags = tags_data),
           list(priority = priority_data,
                department = department_data))

c(priority_preds, department_preds) %<-%
  predict(model, list(title = title_data,
                      text_body = text_body_data,
                      tags = tags_data))


## -------------------------------------------------------------------------
plot(model)


## -------------------------------------------------------------------------
plot(model, show_shapes = TRUE)


## -------------------------------------------------------------------------
str(model$layers)

str(model$layers[[4]]$input)

str(model$layers[[4]]$output)


## -------------------------------------------------------------------------
features <- model$layers[[5]]$output
difficulty <- features %>%
  layer_dense(3, activation = "softmax", name = "difficulty")

new_model <- keras_model(
  inputs = list(title, text_body, tags),
  outputs = list(priority, department, difficulty)
)


## -------------------------------------------------------------------------
plot(new_model, show_shapes=TRUE)


## -------------------------------------------------------------------------
CustomerTicketModel <- new_model_class(
  classname = "CustomerTicketModel",

  initialize = function(num_departments) {
    super$initialize()
    self$concat_layer <- layer_concatenate()
    self$mixing_layer <-
      layer_dense(units = 64, activation = "relu")
    self$priority_scorer <-
      layer_dense(units = 1, activation = "sigmoid")
    self$department_classifier <-
      layer_dense(units = num_departments,  activation = "softmax")
  },

  call = function(inputs) {
    title <- inputs$title
    text_body <- inputs$text_body
    tags <- inputs$tags

    features <- list(title, text_body, tags) %>%
      self$concat_layer() %>%
      self$mixing_layer()
    priority <- self$priority_scorer(features)
    department <- self$department_classifier(features)
    list(priority, department)
  }
)


## -------------------------------------------------------------------------
model <- CustomerTicketModel(num_departments = 4)

c(priority, department) %<-% model(list(title = title_data,
                                        text_body = text_body_data,
                                        tags = tags_data))


## -------------------------------------------------------------------------
inputs <- list(title = title_data,
               text_body = text_body_data,
               tags = tags_data)

layer_customer_ticket_model <- create_layer_wrapper(CustomerTicketModel)

outputs <- inputs %>%
  layer_customer_ticket_model(num_departments = 4)
c(priority, department) %<-% outputs


## -------------------------------------------------------------------------
model %>%
  compile(optimizer = "rmsprop",
          loss = c("mean_squared_error", "categorical_crossentropy"),
          metrics = c("mean_absolute_error", "accuracy"))

x <- list(title = title_data,
          text_body = text_body_data,
          tags = tags_data)
y <- list(priority_data, department_data)

model %>% fit(x, y, epochs = 1)
model %>% evaluate(x, y)
c(priority_preds, department_preds) %<-% {
  model %>% predict(x)
}


## -------------------------------------------------------------------------
ClassifierModel <- new_model_class(
  classname = "Classifier",
  initialize = function(num_classes = 2) {
    super$initialize()
    if (num_classes == 2) {
      num_units <- 1
      activation <- "sigmoid"
    } else {
      num_units <- num_classes
      activation <- "softmax"
    }
    self$dense <- layer_dense(units = num_units, activation = activation)
  },

  call = function(inputs)
    self$dense(inputs)
)

inputs  <- layer_input(shape= c(3))
classifier <- ClassifierModel(num_classes = 10)

outputs <- inputs %>%
  layer_dense(64, activation="relu") %>%
  classifier()

model <- keras_model(inputs = inputs, outputs = outputs)


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(64))
outputs <- inputs %>% layer_dense(1, activation = "sigmoid")
binary_classifier <- keras_model(inputs = inputs, outputs = outputs)

MyModel <- new_model_class(
  classname = "MyModel",
  initialize = function(num_classes = 2) {
    super$initialize()
    self$dense <- layer_dense(units = 64, activation = "relu")
    self$classifier <- binary_classifier
  },

  call = function(inputs) {
    inputs %>%
      self$dense() %>%
      self$classifier()
  }
)

model <- MyModel()


## -------------------------------------------------------------------------
get_mnist_model <- function() {
  inputs <- layer_input(shape = c(28 * 28))
  outputs <- inputs %>%
    layer_dense(512, activation = "relu") %>%
    layer_dropout(0.5) %>%
    layer_dense(10, activation = "softmax")

  keras_model(inputs, outputs)
}

c(c(images, labels), c(test_images, test_labels)) %<-% dataset_mnist()

images <- array_reshape(images, c(-1, 28 * 28)) / 255
test_images <- array_reshape(test_images, c(-1, 28 * 28)) / 255

val_idx <- seq(10000)
val_images <- images[val_idx, ]
val_labels <- labels[val_idx]
train_images <- images[-val_idx, ]
train_labels <- labels[-val_idx]

model <- get_mnist_model()
model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")
model %>% fit(train_images, train_labels,
              epochs = 3,
              validation_data = list(val_images, val_labels))
test_metrics <-  model %>% evaluate(test_images, test_labels)
predictions <- model %>% predict(test_images)


## -------------------------------------------------------------------------
library(tensorflow)

metric_root_mean_squared_error <- new_metric_class(
  classname = "RootMeanSquaredError",

  initialize = function(name = "rmse", ...) {
    super$initialize(name = name, ...)
    self$mse_sum <- self$add_weight(name = "mse_sum",
                                    initializer = "zeros",
                                    dtype = "float32")
    self$total_samples <- self$add_weight(name = "total_samples",
                                          initializer = "zeros",
                                          dtype = "int32")
  },

  update_state = function(y_true, y_pred, sample_weight = NULL) {
    num_samples <- tf$shape(y_pred)[1]
    num_features <- tf$shape(y_pred)[2]

    y_true <- tf$one_hot(y_true, depth = num_features)

    mse <- sum((y_true - y_pred) ^ 2)
    self$mse_sum$assign_add(mse)
    self$total_samples$assign_add(num_samples)
  },

  result = function() {
    sqrt(self$mse_sum / tf$cast(self$total_samples, "float32"))
  },

  reset_state = function() {
    self$mse_sum$assign(0)
    self$total_samples$assign(0L)
  }
)


## ---- eval = FALSE--------------------------------------------------------
##   result = function()
##     sqrt(self$mse_sum / tf$cast(self$total_samples, "float32"))


## ---- eval = FALSE--------------------------------------------------------
##   reset_state = function() {
##     self$mse_sum$assign(0)
##     self$total_samples$assign(0L)
##   }


## -------------------------------------------------------------------------
model <- get_mnist_model()
model %>%
  compile(optimizer = "rmsprop",
          loss = "sparse_categorical_crossentropy",
          metrics = list("accuracy", metric_root_mean_squared_error()))
model %>%
  fit(train_images, train_labels,
      epochs=3,
      validation_data = list(val_images, val_labels))
test_metrics <- model %>% evaluate(test_images, test_labels)


## ---- eval = FALSE--------------------------------------------------------
## callback_model_checkpoint()
## callback_early_stopping()
## callback_learning_rate_scheduler()
## callback_reduce_lr_on_plateau()
## callback_csv_logger()
## ...


## -------------------------------------------------------------------------
callbacks_list <- list(
  callback_early_stopping(monitor = "val_accuracy", patience = 2),
  callback_model_checkpoint(filepath = "checkpoint_path.keras",
                            monitor = "val_loss",
                            save_best_only = TRUE)
)

model <- get_mnist_model()
model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")
model %>% fit(
  train_images, train_labels,
  epochs = 10,
  callbacks = callbacks_list,
  validation_data = list(val_images, val_labels)
)


## -------------------------------------------------------------------------
model <- load_model("checkpoint_path.keras")


## ---- eval = FALSE--------------------------------------------------------
## on_epoch_begin(epoch, logs)
## on_epoch_end(epoch, logs)
## on_batch_begin(batch, logs)
## on_batch_end(batch, logs)
## on_train_begin(logs)
## on_train_end(logs)


## -------------------------------------------------------------------------
callback_plot_per_batch_loss_history <- new_callback_class(
  classname = "PlotPerBatchLossHistory",

  initialize = function(file = "training_loss.pdf") {
    private$outfile <- file
  },

  on_train_begin = function(logs = NULL) {
    private$plots_dir <- tempfile()
    dir.create(private$plots_dir)
    private$per_batch_losses <-
      fastmap::faststack(init = self$params$steps)
  },

  on_epoch_begin = function(epoch, logs = NULL) {
    private$per_batch_losses$reset()
  },

  on_batch_end = function(batch, logs = NULL) {
    private$per_batch_losses$push(logs$loss)
  },

  on_epoch_end = function(epoch, logs = NULL) {
    losses <- as.numeric(private$per_batch_losses$as_list())

    filename <- sprintf("epoch_%04i.pdf", epoch)
    filepath <- file.path(private$plots_dir, filename)

    pdf(filepath, width = 7, height = 5)
    on.exit(dev.off())

    plot(losses, type = "o",
         ylim = c(0, max(losses)),
         panel.first = grid(),
         main = sprintf("Training Loss for Each Batch\n(Epoch %i)", epoch),
         xlab = "Batch", ylab = "Loss")
  },

  on_train_end = function(logs) {
    private$per_batch_losses <- NULL
    plots <- sort(list.files(private$plots_dir, full.names = TRUE))
    qpdf::pdf_combine(plots, private$outfile)
    unlink(private$plots_dir, recursive = TRUE)
  }
)


## -------------------------------------------------------------------------
model <- get_mnist_model()
model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")
model %>% fit(train_images, train_labels,
              epochs = 10,
              callbacks = list(callback_plot_per_batch_loss_history()),
              validation_data = list(val_images, val_labels))


## ---- echo=FALSE----------------------------------------------------------
pth <- pdftools::pdf_convert("training_loss.pdf", pages = 1, dpi = 300,
                      filenames = "per_batch_training_loss_pg%s.%s", verbose = FALSE)
par(mar = c(0,0,0,0))
plot.new()
plot.window(0:1, 0:1)
rasterImage(png::readPNG(pth, native = TRUE), 0, 0, 1, 1)


## -------------------------------------------------------------------------
model <- get_mnist_model()
model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")

model %>% fit(train_images, train_labels,
              epochs = 10,
              validation_data = list(val_images, val_labels),
              callbacks = callback_tensorboard(log_dir = "logs/"))


## ---- eval = FALSE--------------------------------------------------------
## tensorboard(log_dir = "logs/")


## ---- eval = FALSE--------------------------------------------------------
## library(tensorflow)
##
## train_step <- function(inputs, targets) {
##   with(tf$GradientTape() %as% tape, {
##     predictions <- model(inputs, training = TRUE)
##     loss <- loss_fn(targets, predictions)
##   })
##
##   gradients <- tape$gradients(loss, model$trainable_weights)
##   optimizer$apply_gradients(zip_lists(gradients, model$trainable_weights))
## }


## -------------------------------------------------------------------------
metric <- metric_sparse_categorical_accuracy()
targets <- c(0, 1, 2)
predictions <- rbind(c(1, 0, 0),
                     c(0, 1, 0),
                     c(0, 0, 1))
metric$update_state(targets, predictions)
current_result <- metric$result()
sprintf("result: %.2f",  as.array(current_result))


## -------------------------------------------------------------------------
values <- c(0, 1, 2, 3, 4)
mean_tracker <- metric_mean()
for (value in values)
    mean_tracker$update_state(value)
sprintf("Mean of values: %.2f", as.array(mean_tracker$result()))


## -------------------------------------------------------------------------
model <- get_mnist_model()

loss_fn <- loss_sparse_categorical_crossentropy()
optimizer <- optimizer_rmsprop()
metrics <- list(metric_sparse_categorical_accuracy())
loss_tracking_metric <- metric_mean()

train_step <- function(inputs, targets) {

  with(tf$GradientTape() %as% tape, {
    predictions <- model(inputs, training = TRUE)
    loss <- loss_fn(targets, predictions)
  })
  gradients <- tape$gradient(loss, model$trainable_weights)
  optimizer$apply_gradients(zip_lists(gradients, model$trainable_weights))

  logs <- list()
  for (metric in metrics) {
    metric$update_state(targets, predictions)
    logs[[metric$name]] <- metric$result()
  }

  loss_tracking_metric$update_state(loss)
  logs$loss <- loss_tracking_metric$result()
  logs
}


## -------------------------------------------------------------------------
reset_metrics <- function() {
  for (metric in metrics)
    metric$reset_state()
  loss_tracking_metric$reset_state()
}


## -------------------------------------------------------------------------
library(tfdatasets, exclude = "shape")
training_dataset <-
  list(train_images, train_labels) %>%
  tensor_slices_dataset() %>%
  dataset_batch(32)

epochs <- 3
training_dataset_iterator <- as_iterator(training_dataset)
for (epoch in seq(epochs)) {
  reset_metrics()
  c(inputs_batch, targets_batch) %<-% iter_next(training_dataset_iterator)
  logs <- train_step(inputs_batch, targets_batch)

  writeLines(c(
    sprintf("Results at the end of epoch %s", epoch),
    sprintf("...%s: %.4f", names(logs), sapply(logs, as.numeric))
  ))
}


## -------------------------------------------------------------------------
test_step <- function(inputs, targets) {
  predictions <- model(inputs, training = FALSE)
  loss <- loss_fn(targets, predictions)

  logs <- list()
  for (metric in metrics) {
    metric$update_state(targets, predictions)
    logs[[paste0("val_", metric$name)]] <- metric$result()
  }

  loss_tracking_metric$update_state(loss)
  logs[["val_loss"]] <- loss_tracking_metric$result()
  logs
}

val_dataset <- list(val_images, val_labels) %>%
  tensor_slices_dataset() %>%
  dataset_batch(32)

reset_metrics()

val_dataset_iterator <- as_iterator(val_dataset)
repeat {
  batch <- iter_next(val_dataset_iterator)
  if(is.null(batch)) break
  c(inputs_batch, targets_batch) %<-% batch
  logs <- test_step(inputs_batch, targets_batch)
}

writeLines(c(
  "Evaluation results:",
  sprintf("...%s: %.4f", names(logs), sapply(logs, as.numeric))
))


## -------------------------------------------------------------------------
tf_test_step <- tf_function(test_step)

val_dataset_iterator <- as_iterator(val_dataset)
reset_metrics()

while(!is.null(iter_next(val_dataset_iterator) -> batch)) {
  c(inputs_batch, targets_batch) %<-% batch
  logs <- tf_test_step(inputs_batch, targets_batch)
}

writeLines(c(
  "Evaluation results:",
  sprintf("...%s: %.4f", names(logs), sapply(logs, as.numeric))
))


## ---- include=FALSE-------------------------------------------------------
run_eval <- function(fn) {
  val_dataset_iterator <- as_iterator(val_dataset)
  reset_metrics()

  while (!is.null(iter_next(val_dataset_iterator) -> batch)) {
    c(inputs_batch, targets_batch) %<-% batch
    logs <- fn(inputs_batch, targets_batch)
  }
  NULL
}
manual_elapsed_time <- system.time(run_eval(test_step))[["elapsed"]] %>% sprintf("%1.1f", .)
compiled_elapsed_time <- system.time(run_eval(tf_test_step))[["elapsed"]] %>% sprintf("%1.1f", .)


## -------------------------------------------------------------------------
my_evaluate <- tf_function(function(model, dataset) {
  reset_metrics()

  for (batch in dataset) {
    c(inputs_batch, targets_batch) %<-% batch
    logs <- test_step(inputs_batch, targets_batch)
  }
  logs
})
system.time(my_evaluate(model, val_dataset))


## -------------------------------------------------------------------------
loss_fn <- loss_sparse_categorical_crossentropy()
loss_tracker <- metric_mean(name = "loss")

CustomModel <- new_model_class(
  classname = "CustomModel",

  compile = function(optimizer, loss_fn, ...) {
    super$compile(...)
    optimizer$build(self$variables)
    self$optimizer <- optimizer
    self$loss_fn <- loss_fn
  },

  train_step = function(data) {
    c(inputs, targets) %<-% data
    with(tf$GradientTape() %as% tape, {
      predictions <- self(inputs, training = TRUE)
      loss <- self$loss_fn(targets, predictions)
    })
    gradients <- tape$gradient(loss, model$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(gradients, model$trainable_weights))

    loss_tracker$update_state(loss)
    list(loss = loss_tracker$result())
  },

  metrics = mark_active(function() list(loss_tracker))
)


## -------------------------------------------------------------------------
inputs <- layer_input(shape=c(28 * 28))
features <- inputs %>%
  layer_dense(512, activation="relu") %>%
  layer_dropout(0.5)
outputs <- features %>%
  layer_dense(10, activation="softmax")

model <- CustomModel(inputs = inputs, outputs = outputs)

model %>% compile(optimizer = optimizer_rmsprop(), loss = loss_fn)
model %>% fit(train_images, train_labels, epochs = 3)


## -------------------------------------------------------------------------
CustomModel <- new_model_class(

  classname = "CustomModel",

  compile = function(optimizer, loss_fn, ...) {
    super$compile(...)
    optimizer$build(self$variables)
    self$optimizer <- optimizer
    self$loss_fn <- loss_fn
  },

  train_step = function(data) {
    c(inputs, targets) %<-% data
    with(tf$GradientTape() %as% tape, {
      predictions <- self(inputs, training = TRUE)
      loss <- self$compiled_loss(targets, predictions)
    })
    gradients <- tape$gradient(loss, model$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(gradients, model$trainable_weights))
    self$compiled_metrics$update_state(targets, predictions)
    results <- list()
    for(metric in self$metrics)
      results[[metric$name]] <- metric$result()
    results
  }
)


## -------------------------------------------------------------------------
inputs <- layer_input(shape=c(28 * 28))
features <- inputs %>%
  layer_dense(512, activation="relu") %>%
  layer_dropout(0.5)

outputs <- features %>% layer_dense(10, activation="softmax")
model <- CustomModel(inputs = inputs, outputs = outputs)

model %>% compile(optimizer = optimizer_rmsprop(),
                  loss = loss_sparse_categorical_crossentropy(),
                  metrics = metric_sparse_categorical_accuracy())
model %>% fit(train_images, train_labels, epochs = 3)
