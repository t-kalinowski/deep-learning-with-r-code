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


# ----------------------------------------------------------------------
#| lst-cap: Instantiating a small convnet
library(keras3)

inputs <- keras_input(shape = c(28, 28, 1))
outputs <- inputs |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") |>
  layer_global_average_pooling_2d() |>
  layer_dense(10, activation = "softmax")

model <- keras_model(inputs = inputs, outputs = outputs)


# ----------------------------------------------------------------------
#| lst-cap: "Displaying the model's summary"
model


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Training the convnet on MNIST images
.[.[train_images, train_labels],
  .[test_images, test_labels]] <- dataset_mnist()
train_images <- array_reshape(train_images, c(-1, 28, 28, 1)) / 255             # <1>
test_images  <- array_reshape(test_images, c(-1, 28, 28, 1)) / 255              # <1>

model |> compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)
model |> fit(train_images, train_labels, epochs = 5, batch_size = 64)


# ----------------------------------------------------------------------
#| lst-cap: Evaluating the convnet
result <- evaluate(model, test_images, test_labels)
cat("Test accuracy:", result$accuracy, "\n")


# ----------------------------------------------------------------------
#| lst-cap: Incorrectly structured convnet missing max pooling layers
inputs <- keras_input(shape = c(28, 28, 1))
outputs <- inputs |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") |>
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") |>
  layer_global_average_pooling_2d() |>
  layer_dense(10, activation = "softmax")

model_no_max_pool <- keras_model(inputs = inputs, outputs = outputs)


# ----------------------------------------------------------------------
model_no_max_pool


# ----------------------------------------------------------------------
#| eval: false
# library(fs)
# dir_create("~/.kaggle")
# file_move("~/Downloads/kaggle.json", "~/.kaggle/")
# file_chmod("~/.kaggle/kaggle.json", "0600")


# ----------------------------------------------------------------------
#| eval: false
# reticulate::uv_run_tool("kaggle competitions download -c dogs-vs-cats")


# ----------------------------------------------------------------------
#| eval: false
# py_require("kagglehub")
# kagglehub <- import("kagglehub")
# kagglehub$login()


# ----------------------------------------------------------------------
#| eval: false
# kagglehub$competition_download("dogs-vs-cats")


# ----------------------------------------------------------------------
library(fs)
original_dir <- path("dogs-vs-cats/train")
if (!dir_exists(original_dir)) {
  stop(
    "Missing dataset directory: 'dogs-vs-cats/train'. Download/unzip the Kaggle Dogs vs Cats dataset first.",
    call. = FALSE
  )
}


# ----------------------------------------------------------------------
# zip::unzip('dogs-vs-cats.zip', exdir = "dogs-vs-cats", files = "train.zip")
# zip::unzip("dogs-vs-cats/train.zip", exdir = "dogs-vs-cats")


# ----------------------------------------------------------------------
fs::dir_tree("dogs_vs_cats_small/", recurse = 1)


# ----------------------------------------------------------------------
#| lst-cap: "Copying images to training, validation, and test directories"
library(fs)
library(glue)

original_dir <- path("dogs-vs-cats/train")                                      # <1>
new_base_dir <- path("dogs_vs_cats_small")                                      # <2>

make_subset <- function(subset_name, start_index, end_index) {                  # <3>
  for (category in c("dog", "cat")) {
    file_name <- glue("{category}.{start_index:end_index}.jpg")
    dir_create(new_base_dir / subset_name / category)
    file_copy(original_dir / file_name,
              new_base_dir / subset_name / category / file_name)
  }
}

make_subset("train", start_index = 1, end_index = 1000)                         # <4>
make_subset("validation", start_index = 1001, end_index = 1500)                 # <5>
make_subset("test", start_index = 1501, end_index = 2500)                       # <6>


# ----------------------------------------------------------------------
#| lst-cap: Instantiating a small convnet for dogs vs. cats classification
inputs <- keras_input(shape = c(180, 180, 3))                                   # <1>
outputs <- inputs |>
  layer_rescaling(1 / 255) |>                                                   # <2>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 512, kernel_size = 3, activation = "relu") |>
  layer_global_average_pooling_2d() |>                                          # <3>
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)


# ----------------------------------------------------------------------
model


# ----------------------------------------------------------------------
#| lst-cap: Configuring the model for training
model |> compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: "Using `image_dataset_from_directory()` to read images from directories"
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


# ----------------------------------------------------------------------
example_array <- array(seq(100*6), c(100, 6))
head(example_array)


# ----------------------------------------------------------------------
#| lst-cap: Instantiating a Dataset from an R array
library(tfdatasets, exclude = c("shape"))                                       # <1>

dataset <- tensor_slices_dataset(example_array)                                 # <2>


# ----------------------------------------------------------------------
#| lst-cap: Iterating on a dataset
dataset_iterator <- as_iterator(dataset)
for (i in 1:3) {
  element <- iter_next(dataset_iterator)
  print(element)
}


# ----------------------------------------------------------------------
#| lst-cap: Batching a dataset
batched_dataset <- dataset |> dataset_batch(3)
batched_dataset_iterator <- as_iterator(batched_dataset)
for (i in 1:3) {
  element <- iter_next(batched_dataset_iterator)
  print(element)
}


# ----------------------------------------------------------------------
#| lst-cap: "Applying a Dataset transformation using `dataset_map()`"
reshaped_dataset <- dataset |>
  dataset_map(\(element) tf$reshape(element, shape(2, 3)))

reshaped_dataset_iterator <- as_iterator(reshaped_dataset)
for (i in 1:3) {
  element <- iter_next(reshaped_dataset_iterator)
  print(element)
}


# ----------------------------------------------------------------------
#| lst-cap: "Displaying the shapes yielded by the `Dataset`"
.[data_batch, labels_batch] <- train_dataset |> as_iterator() |> iter_next()
op_shape(data_batch)


# ----------------------------------------------------------------------
op_shape(labels_batch)


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: "Fitting the model using a `Dataset`"
callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_from_scratch.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model |> fit(
  train_dataset,
  epochs = 50,
  validation_data = validation_dataset,
  callbacks = callbacks
)


# ----------------------------------------------------------------------
#| lst-cap: Displaying curves of loss and accuracy during training
#| fig-cap: Training and validation metrics for a simple convnet
plot(history)


# ----------------------------------------------------------------------
#| lst-cap: Evaluating the model on the test set
test_model <- load_model("convnet_from_scratch.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


# ----------------------------------------------------------------------
#| lst-cap: Defining a data augmentation stage
data_augmentation_layers <- list(                                               # <1>
  layer_random_flip(, "horizontal"),
  layer_random_rotation(, 0.1),
  layer_random_zoom(, 0.2)
)

data_augmentation <- function(images, targets) {                                # <2>
  for (layer in data_augmentation_layers)
    images <- layer(images)
  list(images, targets)
}

augmented_train_dataset <- train_dataset |>
  dataset_map(data_augmentation, num_parallel_calls = 8) |>                     # <3>
  dataset_prefetch()                                                            # <4>


# ----------------------------------------------------------------------
#| eval: false
# layer <- layer_random_flip(, "horizontal")
# result <- layer(object)


# ----------------------------------------------------------------------
#| eval: false
# result <- object |> layer_random_flip("horizontal")


# ----------------------------------------------------------------------
#| lst-cap: Randomly augmented training images
#| fig-cap: Generating variations of a very good boy via random data augmentation
batch <- train_dataset |> as_iterator() |> iter_next()
.[images, labels] <- batch

par(mfrow = c(3, 3), mar = rep(.5, 4))

image <- images[1, , , ]
plot(as.raster(image, max = 255))                                               # <1>

for (i in 2:9) {
  .[augmented_images, ..] <- data_augmentation(images, NULL)                    # <2>
  augmented_image <- augmented_images@r[1] |> as.array()                        # <3>
  plot(as.raster(augmented_image, max = 255))                                   # <3>
}


# ----------------------------------------------------------------------
#| lst-cap: Defining a new convnet that includes dropout
inputs <- keras_input(shape = c(180, 180, 3))
outputs <- inputs |>
  layer_rescaling(1 / 255) |>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 512, kernel_size = 3, activation = "relu") |>
  layer_global_average_pooling_2d() |>
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
#| lst-cap: Training the regularized convnet on augmented images
callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_from_scratch_with_augmentation.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model |> fit(
  augmented_train_dataset,
  epochs = 100,                                                                 # <1>
  validation_data = validation_dataset,
  callbacks = callbacks
)


# ----------------------------------------------------------------------
#| fig-cap: Training and validation metrics with data augmentation
plot(history)


# ----------------------------------------------------------------------
#| lst-cap: Evaluating the model on the test set
test_model <- load_model("convnet_from_scratch_with_augmentation.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))



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



