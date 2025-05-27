setwd("~/github/t-kalinowski/deep-learning-with-R-2nd-edition-code/")
reticulate::use_virtualenv("r-keras")
reticulate::py_run_string(r"--(
import signal as signal_mod
og_signal_signal = signal_mod.signal
def signal(signalnum, handler):
  print(f"setting signal.signal({signalnum}, {handler})")
  return og_signal_signal(signalnum, handler)

signal_mod.signal = signal
)--", local = TRUE)

## ----setup, include = FALSE-----------------------------------------------
# Sys.setenv("CUDA_VISIBLE_DEVICES" = "")
# Sys.setenv(TF_USE_LEGACY_KERAS = "1")
# reticulate::use_virtualenv("r-tensorflow")
# reticulate::py_install("keras")
#
library(keras3)
# reticulate::register_class_filter(function(x) {
#   if(!is.na(m <- match("keras.src.models.model.Model", x)))
#     x <- unique(append(x, "keras.engine.training.Model", after = m))
#
#   if(!is.na(m <- match("keras.src.models.sequential.Sequential", x)))
#     x <- unique(append(x, "keras.engine.sequential.Sequential", after = m))
#
#   x
# })
#
# load_model <- load_model_tf
# shape <- dim




# registerS3method("compile", "keras.src.models.model.Model", keras:::compile.keras.engine.training.Model)
# registerS3method("fit", "keras.src.models.model.Model", keras:::compile.keras.engine.training.Model)
# use_backend("jax")
tensorflow::tf_function(function(x) x + 1)(1)
# reticulate::register_class_filter(function(x) {
#   if(!is.na(m <- match("keras.src.models.model.Model", x)))
#     x <- unique(append(x, "keras.engine.training.Model", after = m))
#
#   if(!is.na(m <- match("keras.src.models.sequential.Sequential", x)))
#     x <- unique(append(x, "keras.engine.sequential.Sequential", after = m))
#
#   x
# })



# load_model <- load_model_tf
# shape <- dim

## -------------------------------------------------------------------------
library(fs)
data_dir <- path("pets_dataset")
dir_create(data_dir)


## ---- eval = FALSE--------------------------------------------------------
options(timeout = .Machine$integer.max)
## data_url <- path("http://www.robots.ox.ac.uk/~vgg/data/pets/data")
## for (filename in c("images.tar.gz", "annotations.tar.gz")) {
##   download.file(url =  data_url / filename,
##                 destfile = data_dir / filename)
##   untar(data_dir / filename, exdir = data_dir)
## }


## -------------------------------------------------------------------------
input_dir <- data_dir / "images"
target_dir <- data_dir / "annotations/trimaps/"

image_paths <- tibble::tibble(
  input = sort(dir_ls(input_dir, glob = "*.jpg")),
  target = sort(dir_ls(target_dir, glob = "*.png")))


## ---- eval = FALSE--------------------------------------------------------
## tibble::glimpse(image_paths)


## -------------------------------------------------------------------------
display_image_tensor <- function(x, ..., max = 255,
                                 plot_margins = c(0, 0, 0, 0)) {
  if(!is.null(plot_margins))
    par(mar = plot_margins)

  x %>%
    as.array() %>%
    drop() %>%
    as.raster(max = max) %>%
    plot(..., interpolate = FALSE)
}


## -------------------------------------------------------------------------
library(tensorflow, exclude = c("shape", "set_random_seed"))

image_tensor <- image_paths$input[10] %>%
  tf$io$read_file() %>%
  tf$io$decode_jpeg()

str(image_tensor)
display_image_tensor(image_tensor)


## -------------------------------------------------------------------------
display_target_tensor <- function(target)
  display_image_tensor(target - 1, max = 2)

target <- image_paths$target[10] %>%
   tf$io$read_file() %>%
   tf$io$decode_png()

str(target)
display_target_tensor(target)


## -------------------------------------------------------------------------
library(tfdatasets, exclude = "shape")

tf_read_image <-
  function(path, format = "image", resize = NULL, ...) {

    img <- path %>%
      tf$io$read_file() %>%
      tf$io[[paste0("decode_", format)]](...)

    if (!is.null(resize))
      img <- img %>%
        tf$image$resize(as.integer(resize))

    img
  }

img_size <- c(200, 200)

tf_read_image_and_resize <- function(..., resize = img_size)
  tf_read_image(..., resize = resize)


## ---- message=FALSE-------------------------------------------------------
make_dataset <- function(paths_df) {
    tensor_slices_dataset(paths_df) %>%
    dataset_map(function(path) {
      image <- path$input %>%
        tf_read_image_and_resize("jpeg", channels = 3L)
      target <- path$target %>%
        tf_read_image_and_resize("png", channels = 1L)
      target <- target - 1
      list(image, target)
    }) %>%
    dataset_cache() %>%
    dataset_shuffle(buffer_size = nrow(paths_df)) %>%
    dataset_batch(32)
}

num_val_samples <- 1000
val_idx <- sample.int(nrow(image_paths), num_val_samples)

val_paths <- image_paths[val_idx, ]
train_paths <- image_paths[-val_idx, ]

validation_dataset <- make_dataset(val_paths)
train_dataset <- make_dataset(train_paths)


## -------------------------------------------------------------------------
get_model <- function(img_size, num_classes) {

  conv <- function(..., padding = "same", activation = "relu")
    layer_conv_2d(..., padding = padding, activation = activation)

  conv_transpose <- function(..., padding = "same", activation = "relu")
    layer_conv_2d_transpose(..., padding = padding, activation = activation)

  input <- layer_input(shape = c(img_size, 3))
  output <- input %>%
    layer_rescaling(scale = 1/255) %>%
    conv(64, 3, strides = 2) %>%
    conv(64, 3) %>%
    conv(128, 3, strides = 2) %>%
    conv(128, 3) %>%
    conv(256, 3, strides = 2) %>%
    conv(256, 3) %>%
    conv_transpose(256, 3) %>%
    conv_transpose(256, 3, strides = 2) %>%
    conv_transpose(128, 3) %>%
    conv_transpose(128, 3, strides = 2) %>%
    conv_transpose(64, 3) %>%
    conv_transpose(64, 3, strides = 2) %>%
    conv(num_classes, 3, activation="softmax")

  keras_model(input, output)
}

model <- get_model(img_size = img_size, num_classes = 3)
model


## -------------------------------------------------------------------------
model %>%
  compile(optimizer = "rmsprop",
          loss = "sparse_categorical_crossentropy")

callbacks <- list(
  callback_model_checkpoint("oxford_segmentation.keras",
                            save_best_only = TRUE))

# Sys.sleep(1)
# stop()
#
#
# for (i in 1:5) {
#   cat("\n-------", i, "-----------\n")
# model$fit(train_dataset$take(10L), epochs = 4L)
  # history <- model %>% fit(
  #   train_dataset$take(5L),
  #   epochs = 10,
  #   # callbacks=callbacks,
  #   view_metrics = FALSE,
  #   validation_data = validation_dataset$take(5L)
  # )
# }


## -------------------------------------------------------------------------
# plot(history)


## -------------------------------------------------------------------------
model <- load_model("oxford_segmentation.keras")


## -------------------------------------------------------------------------
test_image <- val_paths$input[309] %>%
  tf_read_image_and_resize("jpeg", channels = 3L)

predicted_mask_probs <- model(test_image[tf$newaxis, , , ])

predicted_mask <- tf$argmax(predicted_mask_probs, axis = -1L)

predicted_target <- predicted_mask + 1

par(mfrow = c(1, 2))
display_image_tensor(test_image)
display_target_tensor(predicted_target)


## ---- eval = FALSE--------------------------------------------------------
## y = f4(f3(f2(f1(x))))


## ---- eval = FALSE--------------------------------------------------------
## x <- ...
## residual <- x
## x <- block(x)
## x <- layer_add(c(x, residual))


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(32, 32, 3))
x <- inputs %>% layer_conv_2d(32, 3, activation = "relu")
residual <- x
x <- x %>% layer_conv_2d(64, 3, activation = "relu", padding = "same")
residual <- residual %>% layer_conv_2d(64, 1)
x <- layer_add(c(x, residual))


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(32, 32, 3))
x <- inputs %>% layer_conv_2d(32, 3, activation = "relu")
residual <- x
x <- x %>%
  layer_conv_2d(64, 3, activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(2, padding = "same")
residual <- residual %>%
  layer_conv_2d(64, 1, strides = 2)
x <- layer_add(list(x, residual))


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(32, 32, 3))
x <- layer_rescaling(inputs, scale = 1/255)

residual_block <- function(x, filters, pooling = FALSE) {
  residual <- x
  x <- x %>%
    layer_conv_2d(filters, 3, activation = "relu", padding = "same") %>%
    layer_conv_2d(filters, 3, activation = "relu", padding = "same")

  if (pooling) {
    x <- x %>% layer_max_pooling_2d(pool_size = 2, padding = "same")
    residual <- residual %>% layer_conv_2d(filters, 1, strides = 2)
  } else if (filters != residual$shape[[4]]) {
    residual <- residual %>% layer_conv_2d(filters, 1)
  }

  layer_add(list(x, residual))
}

outputs <- x %>%
  residual_block(filters = 32, pooling = TRUE) %>%
  residual_block(filters = 64, pooling = TRUE) %>%
  residual_block(filters = 128, pooling = FALSE) %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(inputs = inputs, outputs = outputs)


## ---- eval = FALSE--------------------------------------------------------
## model


## ---- eval = FALSE--------------------------------------------------------
## normalize_data <- apply(data, <axis>, function(x) (x - mean(x)) / sd(x))


## ---- eval = FALSE--------------------------------------------------------
## x <- ...
## x <- x %>%
##   layer_conv_2d(32, 3, use_bias = FALSE) %>%
##   layer_batch_normalization()


## ---- eval = FALSE--------------------------------------------------------
## x %>%
##   layer_conv_2d(32, 3, activation = "relu") %>%
##   layer_batch_normalization()


## ---- eval = FALSE--------------------------------------------------------
## x %>%
##   layer_conv_2d(32, 3, use_bias = FALSE) %>%
##   layer_batch_normalization() %>%
##   layer_activation("relu")


## -------------------------------------------------------------------------
batch_norm_layer_s3_classname <- class(layer_batch_normalization())[1]
batch_norm_layer_s3_classname
is_batch_norm_layer <- function(x)
  inherits(x, batch_norm_layer_s3_classname)

model <- application_efficientnet_b0()
for(layer in model$layers)
  if(is_batch_norm_layer(layer))
    layer$trainable <- FALSE


## -------------------------------------------------------------------------
data_augmentation <- keras_model_sequential() %>%
  layer_random_flip("horizontal") %>%
  layer_random_rotation(0.1) %>%
  layer_random_zoom(0.2)

inputs <- layer_input(shape = c(180, 180, 3))

x <- inputs %>%
  data_augmentation() %>%
  layer_rescaling(scale = 1 / 255)

x <- x %>%
  layer_conv_2d(32, 5, use_bias = FALSE)

for (size in c(32, 64, 128, 256, 512)) {
  residual <- x

  x <- x %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_separable_conv_2d(size, 3, padding = "same", use_bias = FALSE) %>%

    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_separable_conv_2d(size, 3, padding = "same", use_bias = FALSE) %>%

    layer_max_pooling_2d(pool_size = 3, strides = 2, padding = "same")

  residual <- residual %>%
    layer_conv_2d(size, 1, strides = 2, padding = "same", use_bias = FALSE)

  x <- layer_add(list(x, residual))
}

outputs <- x %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)


## -------------------------------------------------------------------------
train_dataset <- image_dataset_from_directory(
  "cats_vs_dogs_small/train",
  image_size = c(180, 180),
  batch_size = 32
)

validation_dataset <- image_dataset_from_directory(
  "cats_vs_dogs_small/validation",
  image_size = c(180, 180),
  batch_size = 32
)

model %>%
  compile(
    loss="binary_crossentropy",
    optimizer="rmsprop",
    metrics="accuracy"
  )


## -------------------------------------------------------------------------
history <- model |> fit(
  train_dataset,
  epochs = 100,
  validation_data = validation_dataset,
  verbose = 2
)


## -------------------------------------------------------------------------
plot(history)


## -------------------------------------------------------------------------
model <- load_model("convnet_from_scratch_with_augmentation.keras")
model


## -------------------------------------------------------------------------
img_path <- get_file(
  fname="cat.jpg",
  origin="https://img-datasets.s3.amazonaws.com/cat.jpg")

img_tensor <- img_path %>%
  tf_read_image(resize = c(180, 180))


## -------------------------------------------------------------------------
display_image_tensor(img_tensor)


## -------------------------------------------------------------------------
conv_layer_s3_classname <- class(layer_conv_2d(NULL, 1, 1))[1]
pooling_layer_s3_classname <- class(layer_max_pooling_2d(NULL))[1]

is_conv_layer <- function(x) inherits(x, conv_layer_s3_classname)
is_pooling_layer <- function(x) inherits(x, pooling_layer_s3_classname)

layer_outputs <- list()
for (layer in model$layers)
  if (is_conv_layer(layer) || is_pooling_layer(layer))
    layer_outputs[[layer$name]] <- layer$output

activation_model <- keras_model(inputs = model$input,
                                outputs = layer_outputs)


## -------------------------------------------------------------------------
activations <- activation_model %>%
  predict(img_tensor[tf$newaxis, , , ])


## -------------------------------------------------------------------------
str(activations)


## -------------------------------------------------------------------------
first_layer_activation <- activations[[ names(layer_outputs)[1] ]]
dim(first_layer_activation)


## -------------------------------------------------------------------------
plot_activations <- function(x, ...) {

  x <- as.array(x)

  if(sum(x) == 0)
    return(plot(as.raster("gray")))

  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
        col = terrain.colors(256), ...)
}

plot_activations(first_layer_activation[, , , 5])



## -------------------------------------------------------------------------
for (layer_name in names(layer_outputs)) {
  layer_output <- activations[[layer_name]]

  n_features <- dim(layer_output) %>% tail(1)
  par(mfrow = n2mfrow(n_features, asp = 1.75),
      mar = rep(.1, 4), oma = c(0, 0, 1.5, 0))
  for (j in 1:n_features)
    plot_activations(layer_output[, , ,j])
  title(main = layer_name, outer = TRUE)
}


## -------------------------------------------------------------------------
model <- application_xception(
  weights = "imagenet",
  include_top = FALSE
)


## -------------------------------------------------------------------------
for (layer in model$layers)
  if(any(grepl("Conv2D", class(layer))))
    print(layer$name)


## -------------------------------------------------------------------------
layer_name <- "block3_sepconv1"
layer <- model %>% get_layer(name = layer_name)
feature_extractor <- keras_model(inputs = model$input,
                                 outputs = layer$output)


## -------------------------------------------------------------------------
activation <- img_tensor %>%
  .[tf$newaxis, , , ] %>%
  xception_preprocess_input() %>%
  feature_extractor()

str(activation)


## -------------------------------------------------------------------------
compute_loss <- function(image, filter_index) {
  activation <- feature_extractor(image)

  filter_index <- as_tensor(filter_index, "int32")
  filter_activation <- activation[, , , filter_index, style = "python"]

  mean(filter_activation[,3:-3, 3:-3])
}


## ---- eval = FALSE--------------------------------------------------------
## predict <- function(model, x) {
##   y <- list()
##   for(x_batch in split_into_batches(x)) {
##     y_batch <- as.array(model(x_batch))
##     y[[length(y)+1]] <- y_batch
##   }
##   unsplit_batches(y)
## }


## -------------------------------------------------------------------------
gradient_ascent_step <-
  function(image, filter_index, learning_rate) {
    with(tf$GradientTape() %as% tape, {
      tape$watch(image)
      loss <- compute_loss(image, filter_index)
    })
    grads <- tape$gradient(loss, image)
    grads <- tf$math$l2_normalize(grads)
    image + (learning_rate * grads)
  }


## -------------------------------------------------------------------------
c(img_width, img_height) %<-% c(200, 200)

generate_filter_pattern <- tf_function(function(filter_index) {
  iterations <- 30
  learning_rate <- 10
  image <- tf$random$uniform(
    minval = 0.4, maxval = 0.6,
    shape = shape(1, img_width, img_height, 3)
  )

  for (i in seq(iterations))
    image <- gradient_ascent_step(image, filter_index, learning_rate)

  image[1, , , ]
})


## -------------------------------------------------------------------------
deprocess_image <- tf_function(function(image, crop = TRUE) {
  image <- image - mean(image)
  image <- image / tf$math$reduce_std(image)
  image <- (image * 64) + 128
  image <- tf$clip_by_value(image, 0, 255)
  if(crop)
    image <- image[26:-26, 26:-26, ]
  image
})


## ---- warning = FALSE-----------------------------------------------------
generate_filter_pattern(filter_index = as_tensor(2L)) %>%
  deprocess_image() %>%
  display_image_tensor()


## ---- fig.asp=1-----------------------------------------------------------
par(mfrow = c(8, 8))
for (i in seq(0, 63)) {
  generate_filter_pattern(filter_index = as_tensor(i)) %>%
    deprocess_image() %>%
    display_image_tensor(plot_margins = rep(.1, 4))
}


## -------------------------------------------------------------------------
model <- application_xception(weights = "imagenet")


## -------------------------------------------------------------------------
img_path <- get_file(
  fname="elephant.jpg",
  origin="https://img-datasets.s3.amazonaws.com/elephant.jpg")

img_tensor <- tf_read_image(img_path, resize = c(299, 299))
preprocessed_img <- xception_preprocess_input(img_tensor[tf$newaxis, , , ])


## -------------------------------------------------------------------------
preds <- predict(model, preprocessed_img)
str(preds)


## -------------------------------------------------------------------------
imagenet_decode_predictions(preds, top=3)[[1]]


## -------------------------------------------------------------------------
which.max(preds[1, ])


## -------------------------------------------------------------------------
last_conv_layer_name <- "block14_sepconv2_act"
classifier_layer_names <- c("avg_pool", "predictions")
last_conv_layer <- model %>% get_layer(last_conv_layer_name)
last_conv_layer_model <- keras_model(model$inputs,
                                     last_conv_layer$output)


## -------------------------------------------------------------------------
classifier_input <- layer_input(batch_shape = last_conv_layer$output$shape)

x <- classifier_input
for (layer_name in classifier_layer_names)
  x <- get_layer(model, layer_name)(x)

classifier_model <- keras_model(classifier_input, x)


## ---- warning=FALSE-------------------------------------------------------
with (tf$GradientTape() %as% tape, {
  last_conv_layer_output <- last_conv_layer_model(preprocessed_img)
  tape$watch(last_conv_layer_output)
  preds <- classifier_model(last_conv_layer_output)
  top_pred_index <- tf$argmax(preds[1, ])
  top_class_channel <- preds[, top_pred_index, style = "python"]
})

grads <- tape$gradient(top_class_channel, last_conv_layer_output)


## -------------------------------------------------------------------------

pooled_grads <- mean(grads, axis = c(1, 2, 3), keepdims = TRUE)

heatmap <-
  (last_conv_layer_output * pooled_grads) %>%
  mean(axis = -1) %>%
  .[1, , ]


## -------------------------------------------------------------------------
par(mar = c(0, 0, 0, 0))
plot_activations(heatmap)


## -------------------------------------------------------------------------
pal <- hcl.colors(256, palette = "Spectral", alpha = .4, rev = TRUE)
heatmap <- as.array(heatmap)
heatmap[] <- pal[cut(heatmap, 256)]
heatmap <- as.raster(heatmap)


## -------------------------------------------------------------------------
img <- tf_read_image(img_path, resize = NULL)
display_image_tensor(img)
rasterImage(heatmap, 0, 0, ncol(img), nrow(img), interpolate = FALSE)
