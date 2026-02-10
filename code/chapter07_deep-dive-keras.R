# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "qpdf")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
Sys.setenv("KERAS_BACKEND" = "jax")
library(keras3)


# ----------------------------------------------------------------------
#| lst-cap: "The `Sequential` class"
library(keras3)

model <- keras_model_sequential() |>
  layer_dense(64, activation = "relu") |>
  layer_dense(10, activation = "softmax")


# ----------------------------------------------------------------------
#| lst-cap: "Incrementally building a `Sequential` model"
model <- keras_model_sequential()
model |> layer_dense(64, activation = "relu")
model |> layer_dense(10, activation = "softmax")


# ----------------------------------------------------------------------
#| lst-cap: "Model that isn't built and has no weights"
model$weights                                                                   # <1>


# ----------------------------------------------------------------------
#| lst-cap: Calling a model for the first time to build it
model$build(input_shape = shape(NA, 3))                                         # <1>
str(model$weights)                                                              # <2>


# ----------------------------------------------------------------------
#| lst-cap: The summary method
model


# ----------------------------------------------------------------------
#| lst-cap: "Naming models and layers with the `name` argument"
model <- keras_model_sequential(name = "my_example_model")
model |> layer_dense(64, activation = "relu", name = "my_first_layer")
model |> layer_dense(10, activation = "softmax", name = "my_last_layer")
model$build(shape(NA, 3))
model


# ----------------------------------------------------------------------
#| lst-cap: Specifying the input shape of a model in advance
model <- keras_model_sequential(input_shape = c(3))                             # <1>
model |> layer_dense(64, activation = "relu")


# ----------------------------------------------------------------------
model


# ----------------------------------------------------------------------
model |> layer_dense(10, activation = "softmax")
model


# ----------------------------------------------------------------------
#| lst-cap: "A simple Functional model with two `Dense` layers"
inputs <- keras_input(shape = c(3), name = "my_input")
features <- inputs |> layer_dense(64, activation = "relu")
outputs <- features |> layer_dense(10, activation = "softmax")
model <- keras_model(inputs = inputs, outputs = outputs,
                     name = "my_functional_model")


# ----------------------------------------------------------------------
inputs <- keras_input(shape = c(3), name = "my_input")


# ----------------------------------------------------------------------
op_shape(inputs)                                                                # <1>


# ----------------------------------------------------------------------
op_dtype(inputs)                                                                # <1>


# ----------------------------------------------------------------------
features <- inputs |> layer_dense(64, activation = "relu")


# ----------------------------------------------------------------------
op_shape(features)


# ----------------------------------------------------------------------
outputs <- features |> layer_dense(10, activation = "softmax")
model <- keras_model(inputs = inputs, outputs = outputs,
                     name = "my_functional_model")


# ----------------------------------------------------------------------
model


# ----------------------------------------------------------------------
#| lst-cap: "A multi-input, multi-output Functional model"
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


# ----------------------------------------------------------------------
#| results: hide
#| message: false
#| output: false
#| lst-cap: Training a model by providing lists of input and target arrays
num_samples <- 1280

random_uniform_array <- function(dim) {
  array(runif(prod(dim)), dim)
}

random_integer_array <- function(dim, minval = 0L, maxval = 1L) {
  array(sample(minval:maxval, prod(dim), replace = TRUE), dim)
}

title_data     <- random_integer_array(c(num_samples, vocabulary_size))         # <1>
text_body_data <- random_integer_array(c(num_samples, vocabulary_size))         # <1>
tags_data      <- random_integer_array(c(num_samples, num_tags))                # <1>

priority_data <- random_uniform_array(c(num_samples, 1))                        # <2>
department_data <- random_integer_array(                                        # <2>
  dim = c(num_samples, 1),                                                      # <2>
  maxval = num_departments - 1                                                  # <2>
)                                                                               # <2>

model |> compile(
  optimizer = "adam",
  loss = c("mean_squared_error", "sparse_categorical_crossentropy"),
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


# ----------------------------------------------------------------------
#| output: false
#| lst-cap: Training a model by providing dicts of input and target arrays
model |> compile(
  optimizer = "adam",
  loss = list(
    priority = "mean_squared_error",
    department = "sparse_categorical_crossentropy"
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


# ----------------------------------------------------------------------
#| eval: true
#| fig-cap: "Plot generated by `plot()` on our ticket classifier model"
plot(model)


# ----------------------------------------------------------------------
#| eval: true
#| fig-cap: Model plot with shape information added
plot(model, show_shapes = TRUE, show_layer_names = TRUE)


# ----------------------------------------------------------------------
#| lst-cap: Retrieving inputs or outputs of a layer in a Functional model
model$layers |> str()


# ----------------------------------------------------------------------
model$layers[[4]]$input |> str()


# ----------------------------------------------------------------------
model$layers[[4]]$output |> str()


# ----------------------------------------------------------------------
#| lst-cap: Creating a new model by reusing intermediate layer outputs
features <- model$layers[[5]]$output                                            # <1>
difficulty <- features |>
  layer_dense(3, activation = "softmax", name = "difficulty")

new_model <- keras_model(
  inputs = list(title, text_body, tags),
  outputs = list(priority, department, difficulty)
)


# ----------------------------------------------------------------------
#| eval: true
#| fig-cap: Updated ticket classifier model with added difficulty output
plot(new_model, show_shapes = TRUE, show_layer_names = TRUE)


# ----------------------------------------------------------------------
#| lst-cap: A simple subclassed model
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


# ----------------------------------------------------------------------
model <- CustomerTicketModel(num_departments = 4)
.[priority, department] <- model(list(
  title = title_data,
  text_body = text_body_data,
  tags = tags_data
))


# ----------------------------------------------------------------------
#| results: hide
#| output: false
model |> compile(
  optimizer = "adam",
  loss = c("mean_squared_error", "sparse_categorical_crossentropy"),            # <1>
  metrics = c("mean_absolute_error", "accuracy")                                # <1>
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


# ----------------------------------------------------------------------
#| lst-cap: Creating a Functional model that includes a subclassed model
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


# ----------------------------------------------------------------------
#| lst-cap: Creating a subclassed model that includes a Functional model
inputs <- keras_input(shape = c(64))
outputs <- inputs |> layer_dense(1, activation = "sigmoid")
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


# ----------------------------------------------------------------------
#| results: false
#| lst-cap: "Standard workflow: `compile()` / `fit()` / `evaluate()` / `predict()`"
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


# ----------------------------------------------------------------------
#| lst-cap: "Implementing a custom metric by subclassing the `Metric` class"
metric_sparse_root_mean_squared_error <- new_metric_class(                      # <1>
  classname = "SparseRootMeanSquaredError",                                     # <1>

  initialize = function(name = "rmse", ...) {                                   # <2>
    super$initialize(name = name, ...)                                          # <2>
    self$sum_sq_error <- self$add_weight(                                       # <2>
      name = "sum_sq_error", initializer = "zeros"                              # <2>
    )                                                                           # <2>
    self$total_samples <- self$add_weight(                                      # <2>
      name = "total_samples", initializer = "zeros"                             # <2>
    )                                                                           # <2>
  },                                                                            # <2>

  update_state = function(y_true, y_pred, sample_weight = NULL) {               # <3>
    .[num_samples, num_classes] <- op_shape(y_pred)                             # <3>
    y_true <- op_one_hot(                                                       # <3>
      y_true,                                                                   # <3>
      zero_indexed = TRUE,                                                      # <3>
      num_classes = num_classes                                                 # <3>
    )                                                                           # <3>
    sse <- op_sum(op_square(y_true - y_pred))                                   # <3>
    self$sum_sq_error$assign_add(sse)                                           # <3>
    self$total_samples$assign_add(num_samples)                                  # <3>
  },                                                                            # <3>

# ----------------------------------------------------------------------

  result = function() {
    op_sqrt(op_divide_no_nan(
      self$sum_sq_error,
      self$total_samples
    ))
  },

# ----------------------------------------------------------------------

  reset_state = function() {
    self$sum_sq_error$assign(0)
    self$total_samples$assign(0)
  }
)


# ----------------------------------------------------------------------
#| results: hold
model <- get_mnist_model()
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = list("accuracy", metric_sparse_root_mean_squared_error())
)
model |> fit(
  train_images, train_labels,
  epochs = 3,
  validation_data = list(val_images, val_labels)
)
test_metrics <- model |> evaluate(test_images, test_labels)


# ----------------------------------------------------------------------
#| eval: false
# callback_model_checkpoint()
# callback_early_stopping()
# callback_learning_rate_scheduler()
# callback_reduce_lr_on_plateau()
# callback_csv_logger()


# ----------------------------------------------------------------------
#| lst-cap: "Using the `callbacks` argument in the `fit()` method"
callbacks_list <- list(                                                         # <1>
  callback_early_stopping(                                                      # <2>
    monitor = "val_accuracy",                                                   # <3>
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


# ----------------------------------------------------------------------
model <- load_model("checkpoint_path.keras")


# ----------------------------------------------------------------------
#| eval: false
# on_epoch_begin(epoch, logs)                                                     # <1>
# on_epoch_end(epoch, logs)                                                       # <2>
# on_batch_begin(batch, logs)                                                     # <3>
# on_batch_end(batch, logs)                                                       # <4>
# on_train_begin(logs)                                                            # <5>
# on_train_end(logs)                                                              # <6>


# ----------------------------------------------------------------------
#| lst-cap: "Creating a custom callback by subclassing the `Callback` class"
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


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
#| fig-cap: The output of our custom history plotting callback
if (requireNamespace("pdftools", quietly = TRUE)) {
    page <- pdftools::pdf_render_page("training_loss.pdf", dpi = 300)
    par(mar = c(0,0,0,0))
    plot(as.raster(aperm(unclass(page))))
}


# ----------------------------------------------------------------------
model <- get_mnist_model()
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

model |> fit(
  train_images, train_labels,
  epochs = 10,
  validation_data = list(val_images, val_labels),
  callbacks = c(
    callback_tensorboard(
      log_dir = "./full_path_to_your_log_dir"
    )
  )
)


# ----------------------------------------------------------------------
#| eval: false
# # Load TensorBoard in R
# tensorboard(log_dir = "./full_path_to_your_log_dir")


# ----------------------------------------------------------------------
train_step <- function(inputs, targets) {
  predictions <-  model(inputs, training = TRUE)                                # <1>
  loss <- loss_fn(targets, predictions)                                         # <2>
  gradients <- get_gradients_of(loss, wrt = model$trainable_weights)            # <3>
  optimizer$apply(gradients, model$trainable_weights)                           # <4>
}


# ----------------------------------------------------------------------
metric <- metric_sparse_categorical_accuracy()
targets <- op_array(c(0, 1, 2), dtype = "int32")
predictions <- op_array(rbind(c(1, 0, 0), c(0, 1, 0), c(0, 0, 1)))
metric$update_state(targets, predictions)
current_result <- metric$result()
cat(sprintf("result: %.2f\n", current_result))


# ----------------------------------------------------------------------
mean_tracker <- metric_mean()
for(value in 0:4) {
  value <- op_array(value)
  mean_tracker$update_state(value)
}
cat(sprintf("Mean of values: %.2f\n", mean_tracker$result()))


# ----------------------------------------------------------------------
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


