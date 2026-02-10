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


