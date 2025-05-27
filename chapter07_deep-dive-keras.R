library(keras3)


model <- keras_model_sequential() |>
  layer_dense(64, activation = "relu") |>
  layer_dense(10, activation = "softmax")


model <- keras_model_sequential()
model |> layer_dense(64, activation = "relu")
model |> layer_dense(10, activation = "softmax")


model$weights                                                                   # <1>


model$build(input_shape = shape(NA, 3))                                         # <1>
model$weights                                                                   # <2>


model


model <- keras_model_sequential(name = "my_example_model")
model |> layer_dense(64, activation = "relu", name = "my_first_layer")
model |> layer_dense(10, activation = "softmax", name = "my_last_layer")
model$build(shape(NA, 3))
model


model <- keras_model_sequential(input_shape = c(3))                             # <1>
model |> layer_dense(64, activation = "relu")


model

model |> layer_dense(10, activation = "softmax")
model


inputs <- keras_input(shape = c(3), name = "my_input")
features <- inputs |> layer_dense(64, activation = "relu")
outputs <- features |> layer_dense(10, activation = "softmax")
model <- keras_model(inputs = inputs, outputs = outputs,
                     name = "my_functional_model")


inputs <- keras_input(shape = c(3), name = "my_input")


op_shape(inputs)

op_dtype(inputs)


features <- inputs |> layer_dense(64, activation = "relu")


op_shape(features)


outputs <- features |> layer_dense(10, activation = "softmax")
model <- keras_model(inputs = inputs, outputs = outputs,
                     name = "my_functional_model")


model


vocabulary_size <- 10000
num_tags <- 100
num_departments <- 4

title <- keras_input(c(vocabulary_size), name = "title")                        # <1>
text_body <- keras_input(c(vocabulary_size), name = "text_body")                # <1>
tags <- keras_input(c(num_tags), name = "tags")                                 # <1>

features <-
  layer_concatenate(c(title, text_body, tags)) |>                               # <2>
  layer_dense(64, activation = "relu", name = "dense_features")                 # <3>

priority <- features |>                                                         # <4>
  layer_dense(1, activation = "sigmoid", name = "priority")

department <- features |>                                                       # <4>
  layer_dense(num_departments, activation = "softmax", name = "department")

model <- keras_model(                                                           # <5>
  inputs = c(title, text_body, tags),                                           # <5>
  outputs = c(priority, department)                                             # <5>
)                                                                               # <5>


num_samples <- 1280

random_uniform_array <- function(dim) {
  array(runif(prod(dim)), dim)
}

random_integer_array <- function(dim, minval = 0, maxval = 1) {
  array(sample(minval:maxval, prod(dim), replace = TRUE), dim)
}

random_one_hot_array <- function(num_rows, num_classes) {
  to_categorical(sample(0:(num_classes - 1), num_rows, replace = TRUE),
                 num_classes = num_classes)
}

random_binary_array <- function(...) {
  random_integer_array(dim = c(...), 0, 1)
}

title_data <- random_binary_array(num_samples, vocabulary_size)                 # <1>
text_body_data <- random_binary_array(num_samples, vocabulary_size)             # <1>
tags_data <- random_binary_array(num_samples, num_tags)                         # <1>

priority_data <- random_uniform_array(c(num_samples, 1))                        # <2>
department_data <- random_one_hot_array(num_samples, num_departments)           # <2>

model |> compile(
  optimizer = "adam",
  loss = c("mean_squared_error", "categorical_crossentropy"),
  metrics = list(c("mean_absolute_error"), c("accuracy"))
)

model |> fit(
  x = list(title_data, text_body_data, tags_data),
  y = list(priority_data, department_data),
  epochs = 1
)

model |> evaluate(
  x = list(title_data, text_body_data, tags_data),
  y = list(priority_data, department_data)
)

.[priority_preds, department_preds] <- model |> predict(
  list(title_data, text_body_data, tags_data)
)


model |> compile(
  optimizer = "adam",
  loss = list(
    priority = "mean_squared_error",
    department = "categorical_crossentropy"
  ),
  metrics = list(
    priority = c("mean_absolute_error"),
    department = c("accuracy")
  )
)

model |> fit(
  x = list(title = title_data, text_body = text_body_data, tags = tags_data),
  y = list(priority = priority_data, department = department_data),
  epochs = 1
)

model |> evaluate(
  x = list(title = title_data, text_body = text_body_data, tags = tags_data),
  y = list(priority = priority_data, department = department_data)
)

.[priority_preds, department_preds] <- model |> predict(
  list(title = title_data, text_body = text_body_data, tags = tags_data)
)


plot(model)


plot(model, show_shapes = TRUE, show_layer_names = TRUE)


model$layers |> str()
model$layers[[4]]$input |> str()
model$layers[[4]]$output |> str()


features <- model$layers[[5]]$output                                            # <1>
difficulty <- features |>
  layer_dense(3, activation = "softmax", name = "difficulty")

new_model <- keras_model(
  inputs = list(title, text_body, tags),
  outputs = list(priority, department, difficulty)
)


plot(new_model, show_shapes = TRUE, show_layer_names = TRUE)


CustomerTicketModel <- new_model_class(
  classname = "CustomerTicketModel",

  initialize = function(num_departments) {
    super$initialize()                                                          # <1>
    self$concat_layer <- layer_concatenate()                                    # <2>
    self$mixing_layer <- layer_dense(, 64, activation = "relu")                 # <2>
    self$priority_scorer <- layer_dense(, 1, activation = "sigmoid")            # <2>
    self$department_classifier <- layer_dense(, num_departments,                # <2>
                                              activation = "softmax")           # <2>
  },

  call = function(inputs) {                                                     # <3>
    .[title = title, text_body = text_body, tags = tags] <- inputs

    features <- list(title, text_body, tags) |>
      self$concat_layer() |>
      self$mixing_layer()
    priority <- features |> self$priority_scorer()
    department <- features |> self$department_classifier()
    list(priority, department)
  }
)


model <- CustomerTicketModel(num_departments = 4)
.[priority, department] <- model(
  list(
    title = title_data,
    text_body = text_body_data,
    tags = tags_data
  )
)


model |> compile(
  optimizer = "adam",
  loss = c("mean_squared_error", "categorical_crossentropy"),                   # <1>
  metrics = list(c("mean_absolute_error"), c("accuracy"))                       # <1>
)
model |> fit(
  x = list(title = title_data,                                                  # <2>
           text_body = text_body_data,                                          # <2>
           tags = tags_data),                                                   # <2>
  y = list(priority_data, department_data),                                     # <2>
  epochs = 1
)
model |> evaluate(
  x = list(title = title_data,
           text_body = text_body_data,
           tags = tags_data),
  y = list(priority_data, department_data)
)
.[priority_preds, department_preds] <- model |> predict(
  list(title = title_data,
       text_body = text_body_data,
       tags = tags_data)
)


Classifier <- new_model_class(
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
    self$dense <- layer_dense(, num_units, activation = activation)
  },

  call = function(inputs) {
    self$dense(inputs)
  }
)

classifier <- Classifier(num_classes = 10)

inputs <- keras_input(shape = c(3))
outputs <- inputs |>
  layer_dense(64, activation = "relu") |>
  classifier()
model <- keras_model(inputs = inputs, outputs = outputs)


inputs <- keras_input(shape = c(64))
outputs <- inputs |> layer_dense(64, activation = "relu")
binary_classifier <- keras_model(inputs = inputs, outputs = outputs)

MyModel <- new_model_class(
  classname = "MyModel",

  initialize = function(num_classes = 2) {
    super$initialize()
    self$dense <- layer_dense(units = 64, activation = "relu")
    self$classifier <- binary_classifier
  },

  call = function(inputs) {
    inputs |>
      self$dense() |>
      self$classifier()
  }
)

model <- MyModel()


get_mnist_model <- function() {                                                 # <1>
  inputs <- keras_input(shape = c(28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(10, activation = "softmax")
  keras_model(inputs, outputs)
}

.[.[images, labels], .[test_images, test_labels]] <- dataset_mnist()            # <2>
images <- array_reshape(images, c(60000, 28 * 28)) / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28)) / 255
train_images <- images[10001:60000, ]
val_images <- images[1:10000, ]
train_labels <- labels[10001:60000]
val_labels <- labels[1:10000]

model <- get_mnist_model()                                                      # <3>
model |> compile(                                                               # <3>
  optimizer = "adam",                                                           # <3>
  loss = "sparse_categorical_crossentropy",                                     # <3>
  metrics = "accuracy"                                                          # <3>
)
model |> fit(                                                                   # <4>
  train_images, train_labels,                                                   # <4>
  epochs = 3,                                                                   # <4>
  validation_data = list(val_images, val_labels)                                # <4>
)
test_metrics <- model |> evaluate(test_images, test_labels)                     # <5>
predictions <- model |> predict(test_images)                                    # <6>


metric_root_mean_squared_error <- new_metric_class(                             # <1>
  classname = "RootMeanSquaredError",                                           # <1>

  initialize = function(name = "rmse", ...) {                                   # <2>
    super$initialize(name = name, ...)                                          # <2>
    self$mse_sum <- self$add_weight(                                            # <2>
      name = "mse_sum", initializer = "zeros"                                   # <2>
    )                                                                           # <2>
    self$total_samples <- self$add_weight(                                      # <2>
      name = "total_samples", initializer = "zeros"                             # <2>
    )                                                                           # <2>
  },                                                                            # <2>

  update_state = function(y_true, y_pred, sample_weight = NULL) {               # <3>
    .[num_samples, num_classes] <- op_shape(y_pred)                             # <3>
    y_true <- op_one_hot(y_true,                                                # <3>
                         zero_indexed = TRUE,                                   # <3>
                         num_classes = num_classes)                             # <3>
    mse <- op_sum(op_square(y_true - y_pred))                                   # <3>
    self$mse_sum$assign_add(mse)                                                # <3>
    self$total_samples$assign_add(num_samples)                                  # <3>
  },                                                                            # <3>
  result = function() {
    op_sqrt(self$mse_sum / self$total_samples)
  },
  reset_state = function() {
    self$mse_sum$assign(0)
    self$total_samples$assign(0)
  }
)


model <- get_mnist_model()
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = list("accuracy", metric_root_mean_squared_error())
)
model |> fit(
  train_images, train_labels,
  epochs = 3,
  validation_data = list(val_images, val_labels)
)
test_metrics <- model |> evaluate(test_images, test_labels)


callbacks_list <- list(                                                         # <1>
  callback_early_stopping(                                                      # <2>
    monitor = "accuracy",                                                       # <3>
    patience = 1                                                                # <4>
  ),
  callback_model_checkpoint(                                                    # <5>
    filepath = "checkpoint_path.keras",                                         # <6>
    monitor = "val_loss",                                                       # <7>
    save_best_only = TRUE                                                       # <7>
  )
)

model <- get_mnist_model()
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"                                                          # <8>
)
model |> fit(                                                                   # <9>
  train_images, train_labels,                                                   # <9>
  epochs = 10,                                                                  # <9>
  callbacks = callbacks_list,                                                   # <9>
  validation_data = list(val_images, val_labels)                                # <9>
)                                                                               # <9>


model <- load_model("checkpoint_path.keras")


callback_plot_per_batch_loss_history <- new_callback_class(
  classname = "PlotPerBatchLossHistory",

  initialize = function(file = "training_loss.pdf") {
    private$outfile <- file
  },

  on_train_begin = function(logs = NULL) {
    private$plots_dir <- tempfile()
    dir.create(private$plots_dir)
    private$per_batch_losses <-
      reticulate::py_eval("[]", convert = FALSE)                                # <1>
  },

  on_epoch_begin = function(epoch, logs = NULL) {
    private$per_batch_losses$clear()
  },

  on_batch_end = function(batch, logs = NULL) {
    private$per_batch_losses$append(logs$loss)
  },

  on_epoch_end = function(epoch, logs = NULL) {
    losses <- as.numeric(reticulate::py_to_r(private$per_batch_losses))

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


model <- get_mnist_model()
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)
model |> fit(
  train_images, train_labels,
  epochs = 10,
  callbacks = list(callback_plot_per_batch_loss_history()),
  validation_data = list(val_images, val_labels)
)


model <- get_mnist_model()
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

tensorboard <- callback_tensorboard(
  log_dir = "./full_path_to_your_log_dir"
)
model |> fit(
  train_images, train_labels,
  epochs = 10,
  validation_data = list(val_images, val_labels),
  callbacks = list(tensorboard)
)


train_step <- function(inputs, targets) {
  predictions <-  model(inputs, training = TRUE)                                # <1>
  loss <- loss_fn(targets, predictions)                                         # <2>
  gradients <- get_gradients_of(loss, wrt = model$trainable_weights)            # <3>
  optimizer$apply(gradients, model$trainable_weights)                           # <4>
}


metric <- metric_sparse_categorical_accuracy()
targets <- op_array(c(0, 1, 2), dtype = "int32")
predictions <- op_array(rbind(c(1, 0, 0), c(0, 1, 0), c(0, 0, 1)))
metric$update_state(targets, predictions)
current_result <- metric$result()
cat(sprintf("result: %.2f\n", current_result))


mean_tracker <- metric_mean()
for(value in 0:4) {
  value <- op_array(value)
  mean_tracker$update_state(value)
}
cat(sprintf("Mean of values: %.2f\n", mean_tracker$result()))


metric <- metric_sparse_categorical_accuracy()
targets <- op_array(c(0, 1, 2), dtype = "int32")
predictions <-  op_array(rbind(c(1, 0, 0), c(0, 1, 0), c(0, 0, 1)))

metric_variables <- metric$variables                                            # <1>
metric_variables <- metric$stateless_update_state(                              # <2>
  metric_variables, targets, predictions                                        # <2>
)                                                                               # <2>
current_result <- metric$stateless_result(metric_variables)                     # <3>
cat(sprintf("result: %.2f\n", current_result))

metric_variables <- metric$stateless_reset_state()                              # <4>


# ---- Custom Train Step with TensorFlow ----


library(tensorflow, exclude = c("set_random_seed", "shape"))
library(keras3)

use_backend("tensorflow")


get_mnist_model <- function() {
  inputs <- keras_input(shape = c(28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(10, activation = "softmax")
  keras_model(inputs, outputs)
}


loss_fn <- loss_sparse_categorical_crossentropy()
optimizer <- optimizer_adam()
model <- get_mnist_model()

train_step <- function(inputs, targets) {
  with(tf$GradientTape() %as% tape, {                                           # <1>
    predictions <- model(inputs, training = TRUE)                               # <2>
    loss <- loss_fn(targets, predictions)                                       # <2>
  })
  gradients <- tape$gradient(loss, model$trainable_weights)                     # <3>
  optimizer$apply(gradients, model$trainable_weights)                           # <4>
  loss
}


batch_size <- 32
inputs <- train_images[1:batch_size, ]
targets <- train_labels[1:batch_size]
loss <- train_step(inputs, targets)
loss


# ---- Custom Train Step with PyTorch ----


library(reticulate)
library(keras3)

use_backend("torch")
torch <- import("torch")


get_mnist_model <- function() {
  inputs <- keras_input(shape = c(28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(10, activation = "softmax")
  keras_model(inputs, outputs)
}


loss_fn <- loss_sparse_categorical_crossentropy()
optimizer <- optimizer_adam()
model <- get_mnist_model()

train_step <- function(inputs, targets) {
  predictions <- model(inputs, training = TRUE)                                 # <1>
  loss <- loss_fn(targets, predictions)                                         # <1>
  loss$backward()                                                               # <2>
  gradients <- model$trainable_weights |>
    lapply(\(weight) weight$value$grad)                                         # <3>
  with(torch$no_grad(), {
    optimizer$apply(gradients, model$trainable_weights)
  })                                                                            # <4>
  model$zero_grad()                                                             # <5>
  loss
}


batch_size <- 32
inputs <- train_images[1:batch_size, ]
targets <- train_labels[1:batch_size]
loss <- train_step(inputs, targets)
loss


# ---- Custom Train Step with Jax ----


library(reticulate)
library(keras3)

use_backend("jax")
jax <- import("jax")

get_mnist_model <- function() {
  inputs <- keras_input(shape = c(28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(10, activation = "softmax")
  keras_model(inputs, outputs)
}


model <- get_mnist_model()
model$build(shape(28 * 28))
loss_fn <- loss_sparse_categorical_crossentropy()

compute_loss_and_updates <- function(
    trainable_variables,                                                         # <1>
    non_trainable_variables,
    inputs,
    targets
    ) {
  .[outputs, non_trainable_variables] <-
    model$stateless_call(trainable_variables,                                   # <2>
                         non_trainable_variables,
                         inputs,
                         training = TRUE)
  loss <- loss_fn(targets, outputs)
  list(loss, non_trainable_variables)                                           # <3>
}


grad_fn <- jax$value_and_grad(compute_loss_and_updates, has_aux = TRUE)


learning_rate <- 1e-3

update_weights <- function(gradients, weights) {
  purrr::walk2(weights, gradients, \(w, g) {
    w$assign(w - g * learning_rate)
  })
}


optimizer <- optimizer_adam()
optimizer$build(model$trainable_variables)

train_step <- function(state, inputs, targets) {                                # <1>
  .[trainable_variables, non_trainable_variables, optimizer_variables] <- state # <2>

  .[.[loss, non_trainable_variables], grads] <- grad_fn(                        # <3>
    trainable_variables, non_trainable_variables,
    inputs, targets                                                             # <3>
  )
  .[trainable_variables, optimizer_variables] <- optimizer$stateless_apply(     # <4>
    optimizer_variables, grads, trainable_variables                             # <4>
  )
  list(loss, list(trainable_variables,                                          # <5>
                  non_trainable_variables,
                  optimizer_variables))
}


batch_size = 32
inputs <- train_images[1:batch_size, ]
targets <- train_labels[1:batch_size]

trainable_variables     <- model$trainable_variables     |> lapply(\(w) w$value)
non_trainable_variables <- model$non_trainable_variables |> lapply(\(w) w$value)
optimizer_variables     <- optimizer$variables           |> lapply(\(w) w$value)

state <- list(trainable_variables, non_trainable_variables, optimizer_variables)
.[loss, state] <- train_step(state, inputs, targets)
loss


# ---- Customizing Fit with TensorFlow ----


library(reticulate)
library(keras3)

use_backend("tensorflow")
tf <- import("tensorflow")


loss_fn <- loss_sparse_categorical_crossentropy()
loss_tracker <- metric_mean(name="loss")                                        # <1>

new_custom_model <- new_model_class(
  "CustomModel",

  train_step = function(data) {                                                 # <2>
    .[inputs, targets] <- data
    with(tf$GradientTape() %as% tape, {
      predictions <- self(inputs, training = TRUE)                              # <3>
      loss <- loss_fn(targets, predictions)
    })
    gradients <- tape$gradient(loss, model$trainable_weights)
    self$optimizer$apply(gradients, model$trainable_weights)

    loss_tracker$update_state(loss)                                             # <4>
    list("loss" = loss_tracker$result())                                        # <5>
  },

  metrics = active_property(\() {
    list(loss_tracker)                                                          # <6>
  })
)


get_custom_model <- function() {
  inputs <- keras_input(shape = (28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5)  |>
    layer_dense(10, activation = "softmax")
  model <- new_custom_model(inputs, outputs)
  model |> compile(optimizer = optimizer_adam())
  model
}


model <- get_custom_model()
model |> fit(train_images, train_labels, epochs=3)


# ---- Customizing Fit with Torch ----


library(reticulate)
library(keras3)

use_backend("torch")
torch <- import("torch")


loss_fn <- loss_sparse_categorical_crossentropy()
loss_tracker <- metric_mean(name = "loss")                                      # <1>

new_custom_model <- new_model_class(
  "CustomModel",

  train_step = function(data) {                                                 # <2>
    .[inputs, targets] <- data
    predictions <- self(inputs, training = TRUE)                                # <1>
    loss <- loss_fn(targets, predictions)                                       # <1>

    loss$backward()                                                             # <2>
    trainable_weights <- self$trainable_weights                                 # <2>
    gradients <- trainable_weights |> lapply(\(v) v$value$grad)                 # <2>

    with(torch$no_grad(), {
      self$optimizer$apply(gradients, trainable_weights)                        # <3>
    })

    loss_tracker$update_state(loss)                                             # <4>
    list("loss" = loss_tracker$result())                                        # <5>
  },

  metrics = active_property(\() {
    list(loss_tracker)
  })
)


get_custom_model <- function() {
  inputs <- keras_input(shape = (28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5)  |>
    layer_dense(10, activation = "softmax")
  model <- new_custom_model(inputs, outputs)
  model |> compile(optimizer = optimizer_adam())
  model
}


model <- get_custom_model()
model |> fit(train_images, train_labels, epochs=3)


# ---- Customizing Fit with Jax ----


library(reticulate)
library(keras3)

use_backend("jax")
jax <- import("jax")


loss_fn <- loss_sparse_categorical_crossentropy()

new_custom_model <- new_model_class(
  "CustomModel",

  compute_loss_and_updates = function(trainable_variables,
                                      non_trainable_variables,
                                      inputs,
                                      targets,
                                      training = FALSE) {
    .[predictions, non_trainable_variables] <- self$stateless_call(
        trainable_variables,
        non_trainable_variables,
        inputs,
        training = training
    )
    loss <- loss_fn(targets, predictions)
    list(loss, non_trainable_variables)                                         # <1>
  },
  train_step = function(state, data) {                                          # <1>
    .[trainable_variables,                                                      # <1>
      non_trainable_variables,                                                  # <1>
      optimizer_variables,                                                      # <1>
      metrics_variables] <- state                                               # <1>
    .[inputs, targets] <- data

    grad_fn <- jax$value_and_grad(                                              # <2>
      self$compute_loss_and_updates, has_aux = TRUE                             # <2>
    )                                                                           # <2>

    .[.[loss, non_trainable_variables], grads] <- grad_fn(                      # <3>
      trainable_variables,                                                      # <3>
      non_trainable_variables,                                                  # <3>
      inputs,                                                                   # <3>
      targets,                                                                  # <3>
      training = TRUE                                                           # <3>
    )                                                                           # <3>

    .[trainable_variables, optimizer_variables] <-                              # <4>
      self$optimizer$stateless_apply(                                           # <4>
        optimizer_variables,                                                    # <4>
        grads,                                                                  # <4>
        trainable_variables                                                     # <4>
      )

    logs <- list("loss" = loss)                                                 # <5>
    state <- list(
      trainable_variables,
      non_trainable_variables,
      optimizer_variables,
      metrics_variables
    )
    list(logs, state)                                                           # <6>
  }
)


get_custom_model <- function() {
  inputs <- keras_input(shape = (28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5)  |>
    layer_dense(10, activation = "softmax")
  model <- new_custom_model(inputs, outputs)
  model |> compile(optimizer = optimizer_adam())
  model
}


model <- get_custom_model()
model |> fit(train_images, train_labels, epochs=3)


# ---- Custom Train Step Metrics with TensorFlow ----


library(reticulate)
library(keras3)

use_backend("tensorflow")
tf <- import("tensorflow")


CustomModel <- new_model_class(
  "CustomModel",
  train_step = function(data) {
    .[inputs, targets] <- data
    with(tf$GradientTape() %as% tape, {
      predictions <- self(inputs, training = TRUE)
      loss <- self$compute_loss(y=targets, y_pred=predictions)                  # <1>
    })
    gradients <- tape$gradient(loss, self$trainable_weights)
    self$optimizer$apply(gradients, self$trainable_weights)

    logs <- list()
    for (metric in self$metrics) {                                              # <2>
      if (metric$name == "loss")                                                # <2>
        metric$update_state(loss)                                               # <2>
      else                                                                      # <2>
        metric$update_state(targets, predictions)                               # <2>

      logs[[metric$name]] <- metric$result()                                    # <2>
    }

    logs                                                                        # <3>
  }
)


get_custom_model <- function() {
  inputs <- keras_input(shape = (28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(10, activation = "softmax")

  model <- new_custom_model(inputs, outputs)
  model |> compile(
    optimizer = optimizer_adam(),
    loss = loss_sparse_categorical_crossentropy(),
    metrics = c(metric_sparse_categorical_accuracy())
  )
  model
}

model <- get_custom_model()
model |> fit(train_images, train_labels, epochs = 3)


# ---- Custom Train Step Metrics with Torch ----


library(reticulate)
library(keras3)

use_backend("torch")
torch <- import("torch")


CustomModel <- new_model_class(
  "CustomModel",

  train_step = function(data) {
    .[inputs, targets] <- data
    predictions <- self(inputs, training = TRUE)
    loss <- self$compute_loss(y = targets, y_pred = predictions)

    loss$backward()
    trainable_weights <- self$trainable_weights
    gradients <- trainable_weights |> lapply(\(v) v$value$grad)

    with(torch$no_grad(), {
      self$optimizer$apply(gradients, trainable_weights)
    })

    logs <- list()
    for (metric in self$metrics) {
      if (metric$name == "loss")
        metric$update_state(loss)
      else
        metric$update_state(targets, predictions)

      logs[[metric$name]] <- metric$result()
    }

    logs
  }
)


get_custom_model <- function() {
  inputs <- keras_input(shape = (28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(10, activation = "softmax")

  model <- CustomModel(inputs, outputs)
  model |> compile(
    optimizer = optimizer_adam(),
    loss = loss_sparse_categorical_crossentropy(),
    metrics = c(metric_sparse_categorical_accuracy())
  )
  model
}

model <- get_custom_model()
model |> fit(train_images, train_labels, epochs = 3)


# ---- Custom Train Step Metrics with Jax ----


library(reticulate)
library(keras3)

use_backend("jax")
jax <- import("jax")


CustomModel <- new_model_class(
  "CustomModel",
  compute_loss_and_updates = function(trainable_variables,
                                      non_trainable_variables,
                                      inputs,
                                      targets,
                                      training = FALSE) {
    .[predictions, non_trainable_variables] <-
      self$stateless_call(trainable_variables,
                          non_trainable_variables,
                          inputs,
                          training = training)
    loss <- self$compute_loss(y = targets, y_pred = predictions)
    tuple(loss, tuple(predictions, non_trainable_variables))
  },

  train_step = function(self, state, data) {
    .[trainable_variables,
      non_trainable_variables,
      optimizer_variables,
      metrics_variables] <- state                                               # <1>
    .[inputs, targets] <- data

    grad_fn <-
      jax$value_and_grad(self$compute_loss_and_updates, has_aux = TRUE)

    .[.[loss, .[predictions, non_trainable_variables]], grads] <-
      grad_fn(trainable_variables,
              non_trainable_variables,
              inputs,
              targets,
              training = TRUE)

    .[trainable_variables, optimizer_variables] <-
      self$optimizer$stateless_apply(optimizer_variables,
                                     grads,
                                     trainable_variables)

    new_metrics_vars <- list()
    logs <- list()
    for (metric in self$metrics) {                                              # <2>
      this_metric_vars <- metrics_variables[                                    # <3>
        seq(length(new_metrics_vars) + 1,                                       # <3>
        along.with = metric$variables)                                          # <3>
      ]
      this_metric_vars <- if (metric$name == "loss") {                          # <4>
        metric$stateless_update_state(this_metric_vars, loss)                   # <4>
      } else {                                                                  # <4>
        metric$stateless_update_state(this_metric_vars, targets, predictions)   # <4>
      }                                                                         # <4>
      logs[[metric$name]] <- metric$stateless_result(this_metric_vars)
      new_metrics_vars <- c(new_metrics_vars, this_metric_vars)                 # <5>
    }

    state <- list(
      trainable_variables,
      non_trainable_variables,
      optimizer_variables,
      new_metrics_vars                                                          # <6>
    )
    list(logs, state)
  }
)


get_custom_model <- function() {
  inputs <- keras_input(shape = (28 * 28))
  outputs <- inputs |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(10, activation = "softmax")

  model <- CustomModel(inputs, outputs)
  model |> compile(
    optimizer = optimizer_adam(),
    loss = loss_sparse_categorical_crossentropy(),
    metrics = c(metric_sparse_categorical_accuracy())
  )
  model
}

model <- get_custom_model()
model |> fit(train_images, train_labels, epochs = 3)



