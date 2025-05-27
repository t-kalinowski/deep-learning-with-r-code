library(keras3)


inputs <- keras_input(shape = c(32, 32, 3))
x <- inputs |> layer_conv_2d(32, 3, activation = "relu")
residual <- x                                                                   # <1>
x <- x |> layer_conv_2d(64, 3, activation = "relu", padding = "same")           # <2>
residual <- residual |> layer_conv_2d(64, 1)                                    # <3>
x <- layer_add(c(x, residual))                                                  # <4>


inputs <- keras_input(shape = c(32, 32, 3))
x <- inputs |> layer_conv_2d(32, 3, activation = "relu")
residual <- x                                                                   # <1>
x <- x |>
  layer_conv_2d(64, 3, activation = "relu", padding = "same") |>                # <2>
  layer_max_pooling_2d(2, padding = "same")                                     # <2>
residual <- residual |>
  layer_conv_2d(64, 1, strides = 2)                                             # <3>
x <- layer_add(list(x, residual))                                               # <4>


inputs <- keras_input(shape = c(32, 32, 3))
x <- inputs |> layer_rescaling(scale = 1/255)

residual_block <- function(x, filters, pooling = FALSE) {                       # <1>
  residual <- x
  x <- x |>
    layer_conv_2d(filters, 3, activation = "relu", padding = "same") |>
    layer_conv_2d(filters, 3, activation = "relu", padding = "same")

  if (pooling) {
    x <- x |> layer_max_pooling_2d(pool_size = 2, padding = "same")
    residual <- residual |> layer_conv_2d(filters, 1, strides = 2)              # <2>
  } else if (filters != op_shape(residual)[[4]]) {
    residual <- residual |> layer_conv_2d(filters, 1)                           # <3>
  }

  layer_add(list(x, residual))
}

outputs <- x |>
  residual_block(filters = 32, pooling = TRUE) |>                               # <4>
  residual_block(filters = 64, pooling = TRUE) |>                               # <5>
  residual_block(filters = 128, pooling = FALSE) |>                             # <6>
  layer_global_average_pooling_2d() |>
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(inputs = inputs, outputs = outputs)


model


image_size <- c(180, 180)
batch_size <- 32
data_dir <- fs::path("dogs_vs_cats_small")

train_dataset <-
  image_dataset_from_directory(data_dir / "train",
                               image_size = image_size,
                               batch_size = batch_size)
validation_dataset <-
  image_dataset_from_directory(data_dir / "validation",
                               image_size = image_size,
                               batch_size = batch_size)
test_dataset <-
  image_dataset_from_directory(data_dir / "test",
                               image_size = image_size,
                               batch_size = batch_size)


data_augmentation_layers <- list(
  layer_random_flip(, "horizontal"),
  layer_random_rotation(, 0.1),
  layer_random_zoom(, 0.2)
)

data_augmentation <- function(images, targets) {
  for (layer in data_augmentation_layers) {
    images <- layer(images)
  }
  list(images, targets)
}

augmented_train_dataset <- train_dataset |>
  tfdatasets::dataset_map(data_augmentation, num_parallel_calls = 8) |>
  tfdatasets::dataset_prefetch(4)


inputs <- keras_input(shape = c(180, 180, 3))

x <- inputs |>
  layer_rescaling(scale = 1 / 255) |>                                           # <1>
  layer_conv_2d(filters = 32, kernel_size = 5, use_bias = FALSE)                # <2>

for (size in c(32, 64, 128, 256, 512)) {                                        # <3>
  residual <- x

  x <- x |>
    layer_batch_normalization() |>
    layer_activation("relu") |>
    layer_separable_conv_2d(size, 3, padding = "same", use_bias = FALSE) |>

    layer_batch_normalization() |>
    layer_activation("relu") |>
    layer_separable_conv_2d(size, 3, padding = "same", use_bias = FALSE) |>

    layer_max_pooling_2d(pool_size = 3, strides = 2, padding = "same")

  residual <- residual |>
    layer_conv_2d(size, 1, strides = 2, padding = "same", use_bias = FALSE)

  x <- layer_add(x, residual)
}

outputs <- x |>
  layer_global_average_pooling_2d() |>
  layer_dropout(0.25) |>                                                        # <4>
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)


compile(model,
        loss = "binary_crossentropy",
        optimizer = "adam",
        metrics = "accuracy")
history <- fit(model, augmented_train_dataset,
               epochs = 100,
               validation_data = validation_dataset)


plot(history)



