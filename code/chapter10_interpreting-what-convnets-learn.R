# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "purrr", "tibble", "viridis", "withr")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
use_backend("tensorflow")
reticulate::py_require("keras-hub")


# ----------------------------------------------------------------------
library(keras3)

model <- load_model("convnet_from_scratch_with_augmentation.keras")
model


# ----------------------------------------------------------------------
#| lst-cap: Preprocessing a single image
img_path <- get_file(                                                           # <1>
  fname = "cat.jpg",                                                            # <1>
  origin = "https://img-datasets.s3.amazonaws.com/cat.jpg"                      # <1>
)                                                                               # <1>
get_img_array <- function(img_path, target_size) {
  image <- img_path |>
    image_load(target_size = target_size) |>                                    # <2>
    image_to_array()                                                            # <3>
  dim(image) <- c(1, dim(image))                                                # <4>
  image
}

img <- get_img_array(img_path, target_size = c(180, 180))
str(img)


# ----------------------------------------------------------------------
#| lst-cap: Displaying the test picture
display_image <- function(x, ..., max = 255L, margin = 0) {
  par(mar = rep(margin, 4))

  x |> as.array() |> drop() |>
    as.raster(max = max) |>
    plot(..., interpolate = FALSE)
}


# ----------------------------------------------------------------------
display_image(img)


# ----------------------------------------------------------------------
#| lst-cap: Instantiating a model that returns layer activations
is_conv_layer <- \(x) inherits(x, keras$layers$Conv2D)
is_pooling_layer <- \(x) inherits(x, keras$layers$MaxPooling2D)

layer_outputs <- list()
for (layer in model$layers)                                                     # <1>
  if (is_conv_layer(layer) || is_pooling_layer(layer))                          # <1>
    layer_outputs[[layer$name]] <- layer$output                                 # <1>

activation_model <- keras_model(                                                # <2>
  inputs = model$input,
  outputs = layer_outputs
)


# ----------------------------------------------------------------------
#| lst-cap: Using the model to compute layer activations
activations <- predict(activation_model, img)                                   # <1>
str(activations)


# ----------------------------------------------------------------------
first_layer_activation <- activations[[ names(layer_outputs)[1] ]]
dim(first_layer_activation)


# ----------------------------------------------------------------------
#| lst-cap: Visualizing the sixth channel
#| fig-cap: Sixth channel of the activation of the first layer on the test cat picture
plot_activations <- function(x, ...) {
  withr::local_par(list(mar = c(0,0,0,0)))

  x <- drop(as.array(x))
  if (sum(x) == 0)
    return(plot(as.raster("gray")))

  rotate <- function(x) t(apply(x, 2, rev))
  graphics::image(
    rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
    col = viridis::viridis(256), ...
  )
}


# ----------------------------------------------------------------------
plot_activations(first_layer_activation[, , , 6])


# ----------------------------------------------------------------------
#| lst-cap: Visualizing every channel in every intermediate activation
#| fig-cap: Every channel of every layer activation on the test cat picture
for (layer_name in names(activations)) {                                        # <1>
  layer_activation <- activations[[layer_name]]                                 # <1>

  .[.., n_features] <- dim(layer_activation)                                    # <2>

  par(mfrow = n2mfrow(n_features, asp = 1.75),                                  # <3>
      mar = rep(.1, 4), oma = c(0, 0, 1.5, 0))                                  # <3>

  for (j in 1:n_features)                                                       # <4>
    plot_activations(layer_activation[, , , j])                                 # <4>
  title(main = layer_name, outer = TRUE)                                        # <5>
}



# ----------------------------------------------------------------------
library(keras3)
use_backend("tensorflow")
reticulate::py_require("keras-hub")


# ----------------------------------------------------------------------
get_img_array <- function(img_path, target_size) {
  image <- img_path |>
    image_load(target_size = target_size) |>                                    # <2>
    image_to_array()                                                            # <3>
  dim(image) <- c(1, dim(image))                                                # <4>
  image
}

display_image <- function(x, ..., max = 255L, margin = 0) {
  par(mar = rep(margin, 4))

  x |> as.array() |> drop() |>
    as.raster(max = max) |>
    plot(..., interpolate = FALSE)
}

plot_activations <- function(x, ...) {
  withr::local_par(list(mar = c(0,0,0,0)))

  x <- drop(as.array(x))
  if (sum(x) == 0)
    return(plot(as.raster("gray")))

  rotate <- function(x) t(apply(x, 2, rev))
  graphics::image(
    rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
    col = viridis::viridis(256), ...
  )
}


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
py_require("keras-hub")
keras_hub <- import("keras_hub")

model <- keras_hub$models$Backbone$from_preset(                                 # <1>
  "xception_41_imagenet"                                                        # <1>
)                                                                               # <1>
preprocessor <- keras_hub$layers$ImageConverter$from_preset(                    # <2>
  "xception_41_imagenet",                                                       # <2>
  image_size = shape(180, 180)                                                  # <2>
)


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Printing the names of Xception convolutional layers
unlist(lapply(model$layers, \(layer) {
  if (inherits(layer, keras$layers$Conv2D) ||
      inherits(layer, keras$layers$SeparableConv2D))
    layer$name
}))


# ----------------------------------------------------------------------
#| lst-cap: Feature extractor model returning a specific output
layer_name <- "block3_sepconv1"                                                 # <1>
layer <- get_layer(model, name = layer_name)                                    # <2>
feature_extractor <-
  keras_model(inputs = model$input,                                             # <3>
              outputs = layer$output)


# ----------------------------------------------------------------------
activation <- img |> preprocessor() |> feature_extractor()


# ----------------------------------------------------------------------
compute_loss <- function(image, filter_index) {                                 # <1>
  activation <- feature_extractor(image)
  filter_activation <- activation@r[, 3:-3, 3:-3, filter_index]                 # <2>
  op_mean(filter_activation)                                                    # <3>
}


# ----------------------------------------------------------------------
#| eval: false
# predict <- function(model, x, batch_size = 32) {
#   y <- list()
#   for (x_batch in split_into_batches(x, batch_size)) {
#     y_batch <- as.array(model(x_batch))
#     y[[length(y)+1]] <- y_batch
#   }
#   unsplit_batches(y)
# }


# ----------------------------------------------------------------------
#| lst-cap: Stochastic gradient ascent in TensorFlow
if (keras3::config_backend() == "tensorflow") {
  library(tensorflow, exclude = c("set_random_seed", "shape"))
  
  gradient_ascent_step <- tf_function(\(image, filter_index, learning_rate) {
    with(tf$GradientTape() %as% tape, {
      tape$watch(image)                                                           # <1>
      loss <- compute_loss(image, filter_index)                                   # <2>
    })
    grads <- tape$gradient(loss, image)                                           # <3>
    grads <- op_normalize(grads)                                                  # <4>
    image + (learning_rate * grads)                                               # <5>
  })
}


# ----------------------------------------------------------------------
if (keras3::config_backend() == "torch") {
  torch <- import("torch")
  gradient_ascent_step <- function(image, filter_index, learning_rate) {
    image <- image$clone()$detach()$requires_grad_(TRUE)                          # <1>
    loss <- compute_loss(image, filter_index)
    loss$backward()
    grads <- image$grad
    grads <- op_normalize(grads)
    image + (learning_rate * grads)
  }
}


# ----------------------------------------------------------------------
if (keras3::config_backend() == "jax") {
  jax <- import("jax")
  
  grad_fn <- jax$grad(compute_loss)
  
  gradient_ascent_step <- jax$jit(\(image, filter_index, learning_rate) {
    grads <- grad_fn(image, filter_index)
    grads <- op_normalize(grads)
    image + (learning_rate * grads)
  })
}


# ----------------------------------------------------------------------
#| lst-cap: Function to generate filter visualizations
img_height <- img_width <- 200

generate_filter_pattern <- function(filter_index) {
  iterations <- 30                                                              # <1>
  learning_rate <- 10                                                           # <2>
  image <- random_uniform(                                                      # <3>
    minval = 0.4, maxval = 0.6,                                                 # <3>
    shape = shape(1, img_width, img_height, 3)                                  # <3>
  )

  for (i in seq(iterations))                                                    # <4>
    image <- gradient_ascent_step(image, filter_index, learning_rate)           # <4>

  image
}


# ----------------------------------------------------------------------
#| lst-cap: Utility function to convert a tensor into a valid image
deprocess_image <- function(image, crop = TRUE) {
  image <- op_squeeze(image, axis = 1)                                          # <1>
  image <- image - op_mean(image)                                               # <2>
  image <- image / op_std(image)                                                # <2>
  image <- (image * 64) + 128                                                   # <2>
  image <- op_clip(image, 0, 255)                                               # <2>
  if (crop) {
    image <- image@r[26:-26, 26:-26, ]                                          # <3>
  }
  op_cast(image, "uint8")
}


# ----------------------------------------------------------------------
#| results: hide
#| fig-cap: "Pattern that the third channel in layer `block3_sepconv1` responds to maximally"
generate_filter_pattern(filter_index = 3L) |>
  deprocess_image() |>
  display_image()


# ----------------------------------------------------------------------
#| lst-cap: Generating a grid of all filter response patterns in a layer
#| results: hide
par(mfrow = c(8, 8))
for (i in seq_len(64)) {
  generate_filter_pattern(filter_index = i) |>
    deprocess_image() |>
    display_image(margin = .1)
}



# ----------------------------------------------------------------------
library(keras3)
use_backend("tensorflow")
reticulate::py_require("keras-hub")
py_require("keras-hub")
keras_hub <- import("keras_hub")
library(tensorflow, exclude = c("set_random_seed", "shape"))
torch <- import("torch")
jax <- import("jax")


# ----------------------------------------------------------------------
get_img_array <- function(img_path, target_size) {
  image <- img_path |>
    image_load(target_size = target_size) |>                                    # <2>
    image_to_array()                                                            # <3>
  dim(image) <- c(1, dim(image))                                                # <4>
  image
}

display_image <- function(x, ..., max = 255L, margin = 0) {
  par(mar = rep(margin, 4))

  x |> as.array() |> drop() |>
    as.raster(max = max) |>
    plot(..., interpolate = FALSE)
}

plot_activations <- function(x, ...) {
  withr::local_par(list(mar = c(0,0,0,0)))

  x <- drop(as.array(x))
  if (sum(x) == 0)
    return(plot(as.raster("gray")))

  rotate <- function(x) t(apply(x, 2, rev))
  graphics::image(
    rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
    col = viridis::viridis(256), ...
  )
}

compute_loss <- function(image, filter_index) {                                 # <1>
  activation <- feature_extractor(image)
  filter_activation <- activation@r[, 3:-3, 3:-3, filter_index]                 # <2>
  op_mean(filter_activation)                                                    # <3>
}

gradient_ascent_step <- function(image, filter_index, learning_rate) {
  image <- image$clone()$detach()$requires_grad_(TRUE)                          # <1>
  loss <- compute_loss(image, filter_index)
  loss$backward()
  grads <- image$grad
  grads <- op_normalize(grads)
  image + (learning_rate * grads)
}

generate_filter_pattern <- function(filter_index) {
  iterations <- 30                                                              # <1>
  learning_rate <- 10                                                           # <2>
  image <- random_uniform(                                                      # <3>
    minval = 0.4, maxval = 0.6,                                                 # <3>
    shape = shape(1, img_width, img_height, 3)                                  # <3>
  )

  for (i in seq(iterations))                                                    # <4>
    image <- gradient_ascent_step(image, filter_index, learning_rate)           # <4>

  image
}

deprocess_image <- function(image, crop = TRUE) {
  image <- op_squeeze(image, axis = 1)                                          # <1>
  image <- image - op_mean(image)                                               # <2>
  image <- image / op_std(image)                                                # <2>
  image <- (image * 64) + 128                                                   # <2>
  image <- op_clip(image, 0, 255)                                               # <2>
  if (crop) {
    image <- image@r[26:-26, 26:-26, ]                                          # <3>
  }
  op_cast(image, "uint8")
}


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
img_path <- get_file(                                                           # <1>
  fname = "elephant.jpg",                                                       # <1>
  origin = "https://img-datasets.s3.amazonaws.com/elephant.jpg"                 # <1>
)                                                                               # <1>
img <- img_path |> image_load() |> image_to_array() |> op_expand_dims(1)        # <2>


# ----------------------------------------------------------------------
model <- keras_hub$models$ImageClassifier$from_preset(
  "xception_41_imagenet",
  activation = "softmax",                                                       # <1>
)
preds <- predict(model, img)
str(preds)                                                                      # <2>


# ----------------------------------------------------------------------
decode_imagenet_predictions <- function(preds) {
  decoded <- keras_hub$utils$decode_imagenet_predictions(preds)
  lapply(decoded, \(d) {                                                        # <1>
    .[class_name, score] <- purrr::list_transpose(d)
    tibble::tibble(class_name, score)
  })
}

decode_imagenet_predictions(preds)


# ----------------------------------------------------------------------
which.max(preds[1, ])


# ----------------------------------------------------------------------
img <- model$preprocessor(img)                                                  # <1>


# ----------------------------------------------------------------------
#| lst-cap: Returning the last convolutional output
last_conv_layer_name <- "block14_sepconv2_act"
last_conv_layer <- model$backbone$get_layer(last_conv_layer_name)
last_conv_layer_model <- keras_model(model$inputs, last_conv_layer$output)


# ----------------------------------------------------------------------
#| lst-cap: Going from the last convolutional output to final predictions
classifier_input <- last_conv_layer$output
x <- classifier_input
for (layer_name in c("pooler", "predictions")) {
  layer <- model$get_layer(layer_name)
  x <- layer(x)
}
classifier_model <- keras_model(classifier_input, x)


# ----------------------------------------------------------------------
#| lst-cap: Computing top class gradients with TensorFlow
if (keras3::config_backend() == "tensorflow") {
  tf <- import("tensorflow")
  get_top_class_gradients <- function(image_tensor) {
    last_conv_layer_output <- last_conv_layer_model(image_tensor)                 # <1>
    with(tf$GradientTape() %as% tape, {
      tape$watch(last_conv_layer_output)                                          # <1>
      preds <- classifier_model(last_conv_layer_output)
      top_pred_index <- op_argmax(preds@r[1])
      top_class_channel <- preds@r[, top_pred_index]                              # <2>
    })
  
    grads <- tape$gradient(top_class_channel, last_conv_layer_output)             # <3>
    list(grads, last_conv_layer_output)
  }
}


# ----------------------------------------------------------------------
#| lst-cap: Computing the top class gradients with PyTorch
if (keras3::config_backend() == "torch") {
  torch <- import("torch")
  get_top_class_gradients <- function(image_tensor) {
    last_conv_layer_output <- last_conv_layer_model(image_tensor)$                # <1>
      clone()$detach()$requires_grad_(TRUE)                                       # <2>
  
    preds <- classifier_model(last_conv_layer_output)                             # <3>
    top_pred_index <-  op_argmax(preds@r[1])                                      # <3>
    top_class_channel <- preds@r[, top_pred_index]                                # <3>
    top_class_channel$backward()                                                  # <4>
    grads <- last_conv_layer_output$grad                                          # <4>
    list(grads, last_conv_layer_output)
  }
}


# ----------------------------------------------------------------------
#| lst-cap: Computing the top class gradients with JAX
if (keras3::config_backend() == "jax") {
  jax <- import("jax")
  
  loss_fn <- function(last_conv_layer_output) {                                   # <1>
    preds <- classifier_model(last_conv_layer_output)
    top_pred_index <- op_argmax(preds@r[1])
    top_class_channel <- preds[, top_pred_index]
    top_class_channel@r[1]                                                        # <2>
  }
  grad_fn <- jax$grad(loss_fn)                                                    # <3>
  
  get_top_class_gradients <- function(image_tensor) {
    last_conv_layer_output <- last_conv_layer_model(image_tensor)
    grads <- -grad_fn(last_conv_layer_output)                                     # <4>
    list(grads, last_conv_layer_output)
  }
}


# ----------------------------------------------------------------------
#| lst-cap: Gradient pooling and channel importance weighting
img <- img_path |> image_load() |> image_to_array() |> op_expand_dims(1)        # <1>
img <- model$preprocessor(img)
.[grads, last_conv_layer_output] <- get_top_class_gradients(img)

pooled_grads <- op_mean(grads, axis = c(1, 2, 3), keepdims = TRUE)              # <1>
output <- last_conv_layer_output * pooled_grads                                 # <2>
heatmap <- op_mean(output@r[1], axis = -1)                                      # <3>


# ----------------------------------------------------------------------
#| fig-cap: Standalone class activation heatmap.
#| lst-cap: Visualizing the heatmap
plot_activations(heatmap)


# ----------------------------------------------------------------------
#| fig-cap: African elephant class activation heatmap over the test picture
#| lst-cap: Superimposing the heatmap on the original picture
palette <- hcl.colors(256, palette = "Spectral", alpha = .4)                    # <1>
heatmap <- as.array(-heatmap)                                                   # <1>
heatmap[] <- palette[cut(heatmap, 256)]                                         # <1>
heatmap <- as.raster(heatmap)                                                   # <1>

img <- image_load(img_path) |> image_to_array()                                 # <2>
display_image(img)
rasterImage(                                                                    # <3>
  heatmap,                                                                      # <3>
  0, 0, ncol(img), nrow(img),                                                   # <4>
  interpolate = FALSE                                                           # <5>
)



