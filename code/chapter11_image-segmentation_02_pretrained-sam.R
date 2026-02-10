# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "fs", "tfdatasets", "tibble", "tidyr")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
library(tfdatasets, exclude = "shape")
py_require("keras-hub")
library(fs)
library(dplyr, warn.conflicts = FALSE)


# ----------------------------------------------------------------------
i <- 12


# ----------------------------------------------------------------------
display_image <- function(x, ..., max = 255L, margin = 0) {
  par(mar = rep(margin, 4))

  x |> as.array() |> drop() |>
    as.raster(max = max) |>
    plot(..., interpolate = FALSE)
}

display_target <- function(target, ..., offset = TRUE) {
  if (offset)
    target <- target - 1L                                                       # <1>
  display_image(target, max = 2L, ...)
}

tf_image_load <- function(path, target_size = NULL, ...) {
  img <- path |>
    tf$io$read_file() |>
    tf$io$decode_image(..., expand_animations = FALSE)

  if (!is.null(target_size))
    img <- img |> tf$image$resize(target_size)

  img
}

make_dataset <- function(image_paths) {
  stopifnot(is.data.frame(image_paths),
            names(image_paths) == c("input", "target"))

  tensor_slices_dataset(image_paths) |>
    dataset_map(function(example_paths) {

      input_image <- example_paths$input |>
        tf_image_load(channels = 3L, target_size = img_size)                    # <1>

      target <- example_paths$target |>
        tf_image_load(channels = 1L, target_size = img_size)                    # <2>

      target <- tf$cast(target, "uint8") - 1L                                   # <3>

      list(input_image, target)
    }) |>
    dataset_cache() |>                                                          # <4>
    dataset_shuffle(buffer_size = nrow(image_paths)) |>                         # <5>
    dataset_batch(32)
}

get_model <- function(img_size, num_classes) {

  conv <- function(..., padding = "same", activation = "relu")                  # <1>
    layer_conv_2d(..., padding = padding, activation = activation)

  conv_transpose <- function(..., padding = "same", activation = "relu")        # <1>
    layer_conv_2d_transpose(..., padding = padding, activation = activation)

  input <- keras_input(shape = c(img_size, 3))
  output <- input |>
    layer_rescaling(scale = 1/255) |>                                           # <2>
    conv(64, 3, strides = 2) |>
    conv(64, 3) |>
    conv(128, 3, strides = 2) |>
    conv(128, 3) |>
    conv(256, 3, strides = 2) |>
    conv(256, 3) |>
    conv_transpose(256, 3) |>
    conv_transpose(256, 3, strides = 2) |>
    conv_transpose(128, 3) |>
    conv_transpose(128, 3, strides = 2) |>
    conv_transpose(64, 3) |>
    conv_transpose(64, 3, strides = 2) |>
    conv(num_classes, 3, activation = "softmax")                                # <3>

  keras_model(input, output)
}


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
rm(list = setdiff(ls(), "display_image")); gc()
import("gc")$collect()


# ----------------------------------------------------------------------
py_require("keras-hub")                                                         # <1>

keras_hub <- import("keras_hub")                                                # <2>
model <- keras_hub$models$ImageSegmenter$from_preset("sam_huge_sa1b")


# ----------------------------------------------------------------------
count_params(model) |> prettyNum(",")


# ----------------------------------------------------------------------
path <- get_file(
  origin = "https://s3.amazonaws.com/keras.io/img/book/fruits.jpg"              # <1>
)
pil_image <- image_load(path)                                                   # <2>
image_array <- image_to_array(pil_image, dtype = "float32")                     # <3>
str(image_array)


# ----------------------------------------------------------------------
#| eval: true
#| fig-cap: A fruit test image for Segment Anything
display_image(image_array)                                                      # <1>


# ----------------------------------------------------------------------
image_size <- c(1024, 1024)

resize_and_pad <- function(x) {
  op_image_resize(x, image_size, pad_to_aspect_ratio = TRUE)
}

image <- resize_and_pad(image_array)
op_shape(image)


# ----------------------------------------------------------------------
display_points <- function(coords, color = "white") {
  stopifnot(is.matrix(coords), ncol(coords) == 2)
  coords[, 2] <- image_size[1] - coords[, 2]                                    # <1>
  points(coords, col = color, pch = 8, cex = 2, lwd = 2)
}

display_mask <- function(mask, index = 1,
                         color = "dodgerblue", alpha = 0.6) {
  .[r, g, b] <- col2rgb(color)
  color <- rgb(r, g, b, alpha * 255, maxColorValue = 255)

  mask <- mask |> as.array() |> drop() |> _[index, , ]
  mask[] <- ifelse(mask > 0, color, rgb(0, 0, 0, 0))

  .[h, w] <- image_size
  rasterImage(mask, 0, 0, h, w, interpolate = FALSE)
}

display_box <- function(box, ..., color = "red", lwd = 2) {
  stopifnot(is.matrix(box), dim(box) == c(2, 2))
  box[, 2] <- image_size[1] - box[, 2]                                          # <2>
  rect(xleft = box[1, 1], ytop = box[1, 2],
       xright = box[2, 1], ybottom = box[2, 2],
       ..., border = color, lwd = lwd)
}


# ----------------------------------------------------------------------
#| fig-cap: "A prompt point, landing on a peach"
input_point <- rbind(c(580, 480))                                               # <1>
input_label <- 1                                                                # <2>

display_image(image)
display_points(input_point)


# ----------------------------------------------------------------------
str(model$input)                                                                # <1>


# ----------------------------------------------------------------------
np <- import("numpy", convert = FALSE)                                          # <1>

image |>
  np_array("float32") |>
  np$expand_dims(0L) |>                                                         # <2>
  str()


# ----------------------------------------------------------------------
outputs <- model |> predict(list(
  images = image |> np_array("float32") |> np$expand_dims(0L),
  points = input_point |> np_array("float32") |> np$expand_dims(0L),
  labels = input_label |> np_array("float32") |> np$expand_dims(0L)
))


# ----------------------------------------------------------------------
str(outputs)


# ----------------------------------------------------------------------
#| fig-cap: Segmented peach
display_image(image)
display_mask(outputs$masks)
display_points(input_point)


# ----------------------------------------------------------------------
#| fig-cap: Segmented banana
input_label <- 1
input_point <- rbind(c(300, 550))

outputs <- model |> predict(list(
  images = image |> np_array("float32") |> np$expand_dims(0L),
  points = input_point |> np_array("float32") |> np$expand_dims(0L),
  labels = input_label |> np_array("float32") |> np$expand_dims(0L)
))

display_image(image)
display_mask(outputs$masks)
display_points(input_point)


# ----------------------------------------------------------------------
#| fig-cap: Alternative segmentation masks for the banana prompt
par(mfrow = c(1, 3))
for (i in 2:4) {
  display_image(image)
  display_mask(outputs$masks, index = i)
  display_points(input_point)
  title(paste("Mask", i), col.main= "white", line = -1.5)
}


# ----------------------------------------------------------------------
#| fig-cap: Box prompt around the mango
input_box <- rbind(c(520, 180),                                                 # <1>
                   c(770, 420))                                                 # <2>

display_image(image)
display_box(input_box)


# ----------------------------------------------------------------------
#| fig-cap: Segmented mango
outputs <- model |> predict(list(
  images = image |> np_array("float32") |> np$expand_dims(0L),
  boxes = input_box |> np_array("float32") |> np$expand_dims(c(0L, 1L))         # <1>
))

display_image(image)
display_box(input_box)
display_mask(outputs$masks)


