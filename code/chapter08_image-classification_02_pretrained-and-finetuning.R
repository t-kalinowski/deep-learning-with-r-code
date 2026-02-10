# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "envir", "fs", "glue", "tfdatasets", "tibble", "zip")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
library(tfdatasets, exclude = "shape")
keras3::use_backend("tensorflow")
reticulate::py_require("keras-hub")
library(fs)
library(glue)
library(tfdatasets, exclude = c("shape"))                                       # <1>


# ----------------------------------------------------------------------
batch_size <- 32


# ----------------------------------------------------------------------
make_subset <- function(subset_name, start_index, end_index) {                  # <3>
  for (category in c("dog", "cat")) {
    file_name <- glue("{category}.{start_index:end_index}.jpg")
    dir_create(new_base_dir / subset_name / category)
    file_copy(original_dir / file_name,
              new_base_dir / subset_name / category / file_name)
  }
}

data_augmentation <- function(images, targets) {                                # <2>
  for (layer in data_augmentation_layers)
    images <- layer(images)
  list(images, targets)
}


# ----------------------------------------------------------------------
library(fs)
library(tfdatasets, exclude = "shape")

new_base_dir <- path("dogs_vs_cats_small")
if (!dir_exists(new_base_dir / "train")) {
  stop(
    "Missing directory: 'dogs_vs_cats_small/train'. Run the earlier section that builds the subsampled dataset first.",
    call. = FALSE
  )
}

image_size <- shape(180, 180)
batch_size <- 32

train_dataset <-
  image_dataset_from_directory(new_base_dir / "train",
                               image_size = image_size,
                               batch_size = batch_size)
validation_dataset <-
  image_dataset_from_directory(new_base_dir / "validation",
                               image_size = image_size,
                               batch_size = batch_size)
test_dataset <-
  image_dataset_from_directory(new_base_dir / "test",
                               image_size = image_size,
                               batch_size = batch_size)

data_augmentation_layers <- list(
  layer_random_flip(, "horizontal"),
  layer_random_rotation(, 0.1),
  layer_random_zoom(, 0.2)
)

data_augmentation <- function(images, targets) {
  for (layer in data_augmentation_layers)
    images <- layer(images)
  list(images, targets)
}

augmented_train_dataset <- train_dataset |>
  dataset_map(data_augmentation, num_parallel_calls = 8) |>
  dataset_prefetch()


# ----------------------------------------------------------------------
#| lst-cap: Instantiating the Xception convolutional base
py_require("keras-hub")

keras_hub <- import("keras_hub")
conv_base <- keras_hub$models$Backbone$from_preset("xception_41_imagenet")


# ----------------------------------------------------------------------
preprocessor <- keras_hub$layers$ImageConverter$from_preset(
  "xception_41_imagenet",
  image_size = shape(180, 180)
)


# ----------------------------------------------------------------------
#| lst-cap: Extracting image features and corresponding labels
get_features_and_labels <- function(dataset) {
  dataset |>
    as_array_iterator() |>
    iterate(function(batch) {
      .[images, labels] <- batch
      preprocessed_images <- preprocessor(images)
      features <- conv_base |> predict(preprocessed_images, verbose = 0)
      tibble::tibble(features, labels)
    }) |>
    dplyr::bind_rows()
}

.[train_features, train_labels] <- get_features_and_labels(train_dataset)
.[val_features, val_labels] <- get_features_and_labels(validation_dataset)
.[test_features, test_labels] <- get_features_and_labels(test_dataset)


# ----------------------------------------------------------------------
dim(train_features)


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Defining and training the densely connected classifier
inputs <- keras_input(shape = c(6, 6, 2048))
outputs <- inputs |>
  layer_global_average_pooling_2d() |>                                          # <1>
  layer_dense(256, activation = "relu") |>
  layer_dropout(0.25) |>
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)

model |> compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

callbacks <- list(
  callback_model_checkpoint(
    filepath = "feature_extraction.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model |> fit(
  train_features, train_labels,
  epochs = 10,
  validation_data = list(val_features, val_labels),
  callbacks = callbacks
)


# ----------------------------------------------------------------------
#| fig-cap: Training and validation metrics for plain feature extraction
#| lst-cap: Plotting the results
plot(history)


# ----------------------------------------------------------------------
test_model <- load_model("feature_extraction.keras")
result <- evaluate(test_model, test_features, test_labels)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


# ----------------------------------------------------------------------
#| lst-cap: Creating the frozen convolutional base
conv_base <- keras_hub$models$Backbone$from_preset(
  "xception_41_imagenet",
  trainable = FALSE
)


# ----------------------------------------------------------------------
#| lst-cap: Printing trainable weights before and after freezing
unfreeze_weights(conv_base)                                                     # <1>
length(conv_base$trainable_weights)


# ----------------------------------------------------------------------
freeze_weights(conv_base)                                                       # <1>
length(conv_base$trainable_weights)


# ----------------------------------------------------------------------
inputs <- keras_input(shape=c(180, 180, 3))
outputs <- inputs |>
  preprocessor() |>
  conv_base() |>
  layer_global_average_pooling_2d() |>
  layer_dense(256) |>
  layer_dropout(0.25) |>
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model |> compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)


# ----------------------------------------------------------------------
#| results: hide
callbacks <- list(
  callback_model_checkpoint(
    filepath = "feature_extraction_with_data_augmentation.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model |> fit(
  augmented_train_dataset,
  epochs = 30,
  validation_data = validation_dataset,
  callbacks = callbacks
)


# ----------------------------------------------------------------------
#| fig-cap: Training and validation metrics for feature extraction with data augmentation
plot(history)


# ----------------------------------------------------------------------
#| lst-cap: Evaluating the model on the test set
test_model <- load_model("feature_extraction_with_data_augmentation.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


# ----------------------------------------------------------------------
conv_base$trainable <- TRUE
for (layer in head(conv_base$layers, -4))
  layer$trainable <- FALSE


# ----------------------------------------------------------------------
unfreeze_weights(conv_base, from = -4)


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Fine-tuning the model
model |> compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics = "accuracy"
)

callbacks <- list(
  callback_model_checkpoint(
    filepath = "fine_tuning.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model |> fit(
  augmented_train_dataset,
  epochs = 30,
  validation_data = validation_dataset,
  callbacks = callbacks
)


# ----------------------------------------------------------------------
model <- load_model("fine_tuning.keras")
result <-  evaluate(model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


# ----------------------------------------------------------------------
#| fig-cap: Training and validation metrics for fine-tuning
plot(history)


