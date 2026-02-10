# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "purrr", "tibble", "viridis", "withr")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


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


