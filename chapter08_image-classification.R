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


model


.[.[train_images, train_labels],
  .[test_images, test_labels]] <- dataset_mnist()
train_images <- array_reshape(train_images, c(60000, 28, 28, 1)) / 255
test_images  <- array_reshape(test_images, c(10000, 28, 28, 1)) / 255

model |> compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)
model |> fit(train_images, train_labels, epochs = 5, batch_size = 64)


result <- evaluate(model, test_images, test_labels)
cat("Test accuracy:", result$accuracy, "\n")


inputs <- keras_input(shape = c(28, 28, 1))
outputs <- inputs |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") |>
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") |>
  layer_global_average_pooling_2d() |>
  layer_dense(10, activation = "softmax")

model_no_max_pool <- keras_model(inputs = inputs, outputs = outputs)


model_no_max_pool


unlink("dogs-vs-cats", recursive = TRUE)
unlink("dogs_vs_cats_small", recursive = TRUE)


zip::unzip('dogs-vs-cats.zip', exdir = "dogs-vs-cats", files = "train.zip")


zip::unzip("dogs-vs-cats/train.zip", exdir = "dogs-vs-cats")


library(fs)
original_dir <- path("dogs-vs-cats/train")                                      # <1>
new_base_dir <- path("dogs_vs_cats_small")                                      # <2>

make_subset <- function(subset_name, start_index, end_index) {                  # <3>
  for (category in c("dog", "cat")) {
    file_name <- glue::glue("{category}.{ start_index:end_index }.jpg")
    dir_create(new_base_dir / subset_name / category)
    file_copy(original_dir / file_name,
              new_base_dir / subset_name / category / file_name)
  }
}

make_subset("train", start_index = 1, end_index = 1000)                         # <4>
make_subset("validation", start_index = 1001, end_index = 1500)                 # <5>
make_subset("test", start_index = 1501, end_index = 2500)                       # <6>


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


model


model |> compile(loss = "binary_crossentropy",
                 optimizer = "adam",
                 metrics = "accuracy")


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


reticulate::import("tensorflow")$constant(1L)


library(tfdatasets, exclude = c("shape"))

example_array <- array(seq(100*6), c(100, 6))
head(example_array)
dataset <- tensor_slices_dataset(example_array)                                 # <1>


dataset_iterator <- as_iterator(dataset)
for (i in 1:3) {
  element <- iter_next(dataset_iterator)
  print(element)
}


dataset_iterator <- as_iterator(dataset)
for (i in 1:3) {
  element <- iter_next(dataset_iterator)
  print(element)
}


batched_dataset <- dataset |> dataset_batch(3)
batched_dataset_iterator <- as_iterator(batched_dataset)
for (i in 1:3) {
  element <- iter_next(batched_dataset_iterator)
  print(element)
}


reshaped_dataset <- dataset |>
  dataset_map(\(element) tf$reshape(element, shape(2, 3)))

reshaped_dataset_iterator <- as_iterator(reshaped_dataset)
for (i in 1:3) {
  element <- iter_next(reshaped_dataset_iterator)
  print(element)
}


.[data_batch, labels_batch] <- train_dataset |> as_iterator() |> iter_next()
op_shape(data_batch)
op_shape(labels_batch)


callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_from_scratch.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model |>
  fit(
    train_dataset,
    epochs = 50,
    validation_data = validation_dataset,
    callbacks = callbacks
  )


plot(history)


test_model <- load_model("convnet_from_scratch.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


data_augmentation_layers <- list(                                               # <1>
  layer_random_flip(, "horizontal"), #<
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


plot(history)


test_model <- load_model("convnet_from_scratch_with_augmentation.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


conv_base <- application_xception(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(180, 180, 3)
)


preprocess_inputs <- application_preprocess_inputs(conv_base)                   # <1>
get_features_and_labels <- function(dataset) {
  dataset |>
    as_array_iterator() |>
    iterate(function(batch) {
      .[images, labels] <- batch
      preprocessed_images <- preprocess_inputs(images)
      features <- conv_base |> predict(preprocessed_images, verbose = 0)
      tibble::tibble(features, labels)
    }) |>
    dplyr::bind_rows()
}

.[train_features, train_labels] <- get_features_and_labels(train_dataset)
.[val_features, val_labels] <- get_features_and_labels(validation_dataset)
.[test_features, test_labels] <- get_features_and_labels(test_dataset)


dim(train_features)


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
  epochs = 2,
  validation_data = list(val_features, val_labels),
  callbacks = callbacks
)


plot(history)


test_model <- load_model("feature_extraction.keras")
result <- evaluate(test_model, test_features, test_labels)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


conv_base <- application_xception(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(180, 180, 3)
)
freeze_weights(conv_base)


unfreeze_weights(conv_base)                                                     # <1>
length(conv_base$trainable_weights)

freeze_weights(conv_base)                                                       # <2>
length(conv_base$trainable_weights)


inputs <- keras_input(shape=c(180, 180, 3))
outputs <- inputs |>
  preprocess_inputs() |>
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


plot(history)


test_model <- load_model(
  "feature_extraction_with_data_augmentation.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


unfreeze_weights(conv_base, from = -4)
conv_base


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


model <- load_model("fine_tuning.keras")
result <-  evaluate(model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))

