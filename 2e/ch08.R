## ----setup, include = FALSE-----------------------------------------------
library(keras3)
tensorflow::tf_function(function(x) x + 1)(1)


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(28, 28, 1))

outputs <- inputs %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(10, activation = "softmax")

model <- keras_model(inputs, outputs)


## -------------------------------------------------------------------------
model


## -------------------------------------------------------------------------
c(c(train_images, train_labels), c(test_images, test_labels)) %<-%
  dataset_mnist()
train_images <- array_reshape(train_images, c(60000, 28, 28, 1)) / 255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1)) / 255

model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = c("accuracy"))
model %>% fit(train_images, train_labels, epochs = 5, batch_size = 64)


## -------------------------------------------------------------------------
result <- evaluate(model, test_images, test_labels)
cat("Test accuracy:", result$accuracy, "\n")


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(28, 28, 1))
outputs <- inputs %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(10, activation = "softmax")
model_no_max_pool <- keras_model(inputs = inputs, outputs = outputs)


## -------------------------------------------------------------------------
model_no_max_pool


## ---- eval = FALSE--------------------------------------------------------
## library(fs)
## dir_create("~/.kaggle")
## file_move("~/Downloads/kaggle.json", "~/.kaggle/")
## file_chmod("~/.kaggle/kaggle.json", "0600")


## ---- eval = FALSE--------------------------------------------------------
## reticulate::py_install("kaggle", pip = TRUE)


## ---- eval = FALSE--------------------------------------------------------
## system('kaggle competitions download -c dogs-vs-cats')


## ---- include = FALSE-----------------------------------------------------
# just to make the following chunks reproducible
unlink("dogs-vs-cats", recursive = TRUE)
# note bene, we use the {zip} package instead of base::unzip
# because the latter raises an error on this file.


## -------------------------------------------------------------------------
zip::unzip('dogs-vs-cats.zip', exdir = "dogs-vs-cats", files = "train.zip")
zip::unzip("dogs-vs-cats/train.zip", exdir = "dogs-vs-cats")


## ---- include = FALSE-----------------------------------------------------
unlink("cats_vs_dogs_small", recursive = TRUE)
library(fs)


## -------------------------------------------------------------------------
library(fs)
original_dir <- path("dogs-vs-cats/train")
new_base_dir <- path("cats_vs_dogs_small")

make_subset <- function(subset_name, start_index, end_index) {
  for (category in c("dog", "cat")) {
    file_name <- glue::glue("{category}.{ start_index:end_index }.jpg")
    dir_create(new_base_dir / subset_name / category)
    file_copy(original_dir / file_name,
              new_base_dir / subset_name / category / file_name)
  }
}

make_subset("train", start_index = 1, end_index = 1000)
make_subset("validation", start_index = 1001, end_index = 1500)
make_subset("test", start_index = 1501, end_index = 2500)


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(180, 180, 3))
outputs <- inputs %>%
  layer_rescaling(1 / 255) %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)


## -------------------------------------------------------------------------
model


## -------------------------------------------------------------------------
model %>% compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
train_dataset <-
  image_dataset_from_directory(new_base_dir / "train",
                               image_size = c(180, 180),
                               batch_size = 32)
validation_dataset <-
  image_dataset_from_directory(new_base_dir / "validation",
                               image_size = c(180, 180),
                               batch_size = 32)
test_dataset <-
  image_dataset_from_directory(new_base_dir / "test",
                               image_size = c(180, 180),
                               batch_size = 32)


## -------------------------------------------------------------------------
library(tfdatasets, exclude = "shape")
example_array <- array(seq(100*6), c(100, 6))
head(example_array)
dataset <- tensor_slices_dataset(example_array)


## -------------------------------------------------------------------------
dataset_iterator <- as_iterator(dataset)
for(i in 1:3) {
  element <- iter_next(dataset_iterator)
  print(element)
}


## -------------------------------------------------------------------------
dataset_array_iterator <- as_array_iterator(dataset)
for(i in 1:3) {
  element <- iter_next(dataset_array_iterator)
  str(element)
}


## -------------------------------------------------------------------------
batched_dataset <- dataset %>%
  dataset_batch(3)
batched_dataset_iterator <- as_iterator(batched_dataset)
for(i in 1:3) {
  element <- iter_next(batched_dataset_iterator)
  print(element)
}


## -------------------------------------------------------------------------
reshaped_dataset <- dataset %>%
  dataset_map(function(element) tf$reshape(element, shape(2, 3)))

reshaped_dataset_iterator <- as_iterator(reshaped_dataset)
for(i in 1:3) {
  element <- iter_next(reshaped_dataset_iterator)
  print(element)
}


## -------------------------------------------------------------------------
c(data_batch, labels_batch) %<-% iter_next(as_iterator(train_dataset))
data_batch$shape
labels_batch$shape


## -------------------------------------------------------------------------
callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_from_scratch.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>%
  fit(
    train_dataset,
    epochs = 30,
    validation_data = validation_dataset,
    callbacks = callbacks
  )


## -------------------------------------------------------------------------
plot(history)


## -------------------------------------------------------------------------
test_model <- load_model("convnet_from_scratch.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


## -------------------------------------------------------------------------
data_augmentation <- keras_model_sequential() %>%
  layer_random_flip("horizontal") %>%
  layer_random_rotation(0.1) %>%
  layer_random_zoom(0.2)


## -------------------------------------------------------------------------



## -------------------------------------------------------------------------
library(tfdatasets, exclude = "shape")
batch <- train_dataset %>%
  as_iterator() %>%
  iter_next()

c(images, labels) %<-% batch

par(mfrow = c(3, 3), mar = rep(.5, 4))

image <- images[1, , , ]
plot(as.raster(as.array(image), max = 255))

# plot augmented images
for (i in 2:9) {
  augmented_images <- data_augmentation(images)
  augmented_image <- augmented_images[1, , , ]
  plot(as.raster(as.array(augmented_image), max = 255))
}


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(180, 180, 3))
outputs <- inputs %>%
  data_augmentation() %>%
  layer_rescaling(1 / 255) %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)

model %>% compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_from_scratch_with_augmentation.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>% fit(
  train_dataset,
  epochs = 100,
  validation_data = validation_dataset,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
plot(history)


## -------------------------------------------------------------------------
test_model <- load_model("convnet_from_scratch_with_augmentation.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


## -------------------------------------------------------------------------
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(180, 180, 3)
)


## -------------------------------------------------------------------------
conv_base


## -------------------------------------------------------------------------
get_features_and_labels <- function(dataset) {
  n_batches <- length(dataset)
  all_features <- vector("list", n_batches)
  all_labels <- vector("list", n_batches)
  iterator <- as_array_iterator(dataset)
  for (i in 1:n_batches) {
    c(images, labels) %<-% iter_next(iterator)
    preprocessed_images <- imagenet_preprocess_input(images)
    features <- conv_base %>% predict(preprocessed_images)

    all_labels[[i]] <- labels
    all_features[[i]] <- features
  }

  all_features <- listarrays::bind_on_rows(all_features)
  all_labels <- listarrays::bind_on_rows(all_labels)

  list(all_features, all_labels)
}

c(train_features, train_labels) %<-% get_features_and_labels(train_dataset)
c(val_features, val_labels) %<-% get_features_and_labels(validation_dataset)
c(test_features, test_labels) %<-% get_features_and_labels(test_dataset)


## -------------------------------------------------------------------------
dim(train_features)


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(5, 5, 512))
outputs <- inputs %>%
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_dropout(.5) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)

model %>% compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = "accuracy")

callbacks <- list(
  callback_model_checkpoint(
    filepath = "feature_extraction.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>% fit(
  train_features, train_labels,
  epochs = 20,
  validation_data = list(val_features, val_labels),
  callbacks = callbacks
)


## -------------------------------------------------------------------------
plot(history)


## -------------------------------------------------------------------------
conv_base <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE)
freeze_weights(conv_base)


## -------------------------------------------------------------------------
unfreeze_weights(conv_base)
cat("This is the number of trainable weights",
    "before freezing the conv base:",
    length(conv_base$trainable_weights), "\n")


## -------------------------------------------------------------------------
freeze_weights(conv_base)
cat("This is the number of trainable weights",
    "after freezing the conv base:",
    length(conv_base$trainable_weights), "\n")


## -------------------------------------------------------------------------
data_augmentation <- keras_model_sequential() %>%
  layer_random_flip("horizontal") %>%
  layer_random_rotation(0.1) %>%
  layer_random_zoom(0.2)

inputs <- layer_input(shape = c(180, 180, 3))
outputs <- inputs %>%
  data_augmentation() %>%
  imagenet_preprocess_input() %>%
  conv_base() %>%
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = "accuracy")


## -------------------------------------------------------------------------
callbacks <- list(
  callback_model_checkpoint(
    filepath = "feature_extraction_with_data_augmentation.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>% fit(
  train_dataset,
  epochs = 50,
  validation_data = validation_dataset,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
test_model <- load_model(
  "feature_extraction_with_data_augmentation.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))


## -------------------------------------------------------------------------
conv_base


## -------------------------------------------------------------------------
unfreeze_weights(conv_base, from = -4)
conv_base


## -------------------------------------------------------------------------
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-5),
  metrics = "accuracy"
)

callbacks <- list(
  callback_model_checkpoint(
    filepath = "fine_tuning.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>% fit(
  train_dataset,
  epochs = 30,
  validation_data = validation_dataset,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
model <- load_model("fine_tuning.keras")
result <-  evaluate(model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result$accuracy))
