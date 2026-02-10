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


