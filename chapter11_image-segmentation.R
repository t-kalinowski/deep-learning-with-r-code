Sys.setenv("XLA_PYTHON_CLIENT_PREALLOCATE"="false")
Sys.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION"="1") #.xx fraction

library(reticulate)
library(fs)
library(dplyr, warn.conflicts = FALSE)
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(tfdatasets, exclude = "shape")
library(keras3)

py_require("keras-hub==0.18.1")
py_require("keras-hub")
display_image <- function(x, ..., max = 255, margin = 0) {
  par(mar = rep(margin, 4))

  x |> as.array() |> drop() |>
    as.raster(max = max) |>
    plot(..., interpolate = FALSE)
}


library(fs)
data_dir <- path("pets_dataset")
dir_create(data_dir)


library(dplyr, warn.conflicts = FALSE)
input_dir <- data_dir / "images"
target_dir <- data_dir / "annotations/trimaps/"

all_image_paths <- tibble(
  input = sort(dir_ls(input_dir, glob = "*.jpg")),
  target = sort(dir_ls(target_dir, glob = "*.png", all = FALSE))                # <1>
)


glimpse(all_image_paths)


display_image <- function(x, ..., max = 255, margin = 0) {
  par(mar = rep(margin, 4))

  x |> as.array() |> drop() |>
    as.raster(max = max) |>
    plot(..., interpolate = FALSE)
}


all_image_paths$input[10] |>                                                    # <1>
  image_load() |> image_to_array() |>
  display_image()


display_target <- function(target, ..., offset = TRUE) {
  if (offset)
    target <- target - 1L
  display_image(target, max = 2L, ...)
}

all_image_paths$target[10] |>
  image_load(color_mode = "grayscale") |> image_to_array() |>                   # <1>
  display_target()


library(tfdatasets, exclude = "shape")

img_size <- shape(200, 200)

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

  image_paths |>
  tensor_slices_dataset() |>
    dataset_map(function(example_paths) {
      .[input_path = input, target_path = target] <- example_paths

      input_image <- input_path |>
        tf_image_load(channels = 3L, target_size = img_size)                    # <1>

      target <- target_path |>
        tf_image_load(channels = 1L, target_size = img_size)                    # <2>

      target <- tf$cast(target, "uint8") - 1L                                   # <3>

      list(input_image, target)
    }) |>
    dataset_cache() |>                                                          # <4>
    dataset_shuffle(buffer_size = nrow(image_paths)) |>                         # <5>
    dataset_batch(32)
}

num_val_samples <- 1000                                                         # <6>

image_paths <- all_image_paths |>
  dplyr::mutate(use = ifelse(sample.int(n()) > num_val_samples,
                             "train", "val")) |>                                # <7>
  tidyr::nest(.by = use) |>
  tibble::deframe()

image_paths

.[train_ds = train, val_ds = val] <- image_paths |> lapply(make_dataset)        # <8>


batch <- train_ds |> as_iterator() |> iter_next()
str(batch)

.[images, targets] <- batch
par(mfrow = c(4, 8))
for (i in 1:16) {
  images@r[i] |> display_image()
  targets@r[i] |> display_target(offset = FALSE)                                # <1>
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

model <- get_model(img_size = img_size, num_classes = 3)
model


foreground_iou <- metric_iou(
  num_classes = 3,                                                              # <1>
  target_class_ids = c(0),                                                      # <2>
  name = "foreground_iou",
  sparse_y_true = TRUE,                                                         # <3>
  sparse_y_pred = FALSE,                                                        # <4>
)


model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = foreground_iou
)

callbacks <- list(
  callback_model_checkpoint("oxford_segmentation.keras",
                            save_best_only = TRUE)
)


history <- readRDS("ch11-history.rds")


plot(history, metrics = "loss")


model <- load_model("oxford_segmentation.keras")


for (i in 1:10) {

test_image <- image_paths$val$input[i] |>
  tf_image_load(channels = 3L, target_size = img_size)

test_mask <- image_paths$val$target[i] |>
  tf_image_load(channels = 1L, target_size = img_size) |>
  tf$subtract(1)

predicted_mask_probs <- model(test_image@r[newaxis])
predicted_mask <- op_argmax(predicted_mask_probs, axis = -1) - 1

par(mfrow = c(1, 3))
display_image(test_image); title("image")
display_target(predicted_mask, offset = FALSE)
display_target(test_mask, offset = FALSE)
}


library(reticulate)
py_require("keras-hub==0.18.1")                                                 # <1>

keras_hub <- import("keras_hub")                                                # <2>


model <- keras_hub$models$ImageSegmenter$from_preset("sam_huge_sa1b")


count_params(model)


path <- get_file(
  origin = "https://s3.amazonaws.com/keras.io/img/book/fruits.jpg"              # <1>
)
pil_image <- image_load(path)                                                   # <2>
image_array <- image_to_array(pil_image, dtype = "float32")                     # <3>
display_image(image_array)                                                      # <4>


image_size <- c(1024, 1024)

resize_and_pad <- function(x) {
  op_image_resize(x, image_size, pad_to_aspect_ratio = TRUE)
}

image <- resize_and_pad(image_array)
op_shape(image)


display_points <- function(coords, color = "white") {
  stopifnot(is.matrix(coords), ncol(coords) == 2)
  coords[, 2] <- image_size[1] - coords[, 2]                                    # <1>
  points(coords, col = color, pch = 8, cex = 2, lwd = 2)
}

display_mask <- function(mask, index = 1, color = "dodgerblue", alpha = 0.6) {
  .[r, g, b] <- col2rgb(color)
  color <- rgb(r, g, b, alpha * 255, maxColorValue = 255)

  mask <- mask |> as.array() |> drop() |> _[index, , ]
  mask[] <- ifelse(mask > 1, color, rgb(0, 0, 0, 0))

  .[h, w] <- image_size
  rasterImage(mask, 0, 0, h, w, interpolate = FALSE)
}

display_box <- function(box, ..., color = "red", lwd = 2) {
  stopifnot(is.matrix(box), dim(box) == c(2, 2))
  box[, 2] <- image_size[1] - box[, 2]                                          # <1>
  rect(xleft = box[1, 1], ytop = box[1, 2],
       xright = box[2, 1], ybottom = box[2, 2],
       ..., border = color, lwd = 2)
}


input_point <- rbind(c(580, 480))                                               # <1>
input_label <- 1                                                                # <2>

display_image(image)
display_points(input_point)


str(model$input)                                                                # <1>


np <- import("numpy", convert = FALSE)                                          # <1>

image |>
  np_array("float32") |>
  np$expand_dims(0L) |>                                                         # <2>
  str()


outputs <- model |> predict(list(
  images = image |> np_array("float32") |> np$expand_dims(0L),
  points = input_point |> np_array("float32") |> np$expand_dims(0L),
  labels = input_label |> np_array("float32") |> np$expand_dims(0L)
))

str(outputs)


display_image(image)
display_mask(outputs$masks)
display_points(input_point)


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


par(mfrow = c(1, 3))
for (i in 2:4) {
  display_image(image)
  display_mask(outputs$masks, index = i)
  display_points(input_point)
}


input_box <- rbind(c(520, 180),                                                 # <1>
                   c(770, 420))                                                 # <2>

display_image(image)
display_box(input_box)


outputs <- model |> predict(list(
  images = image |> np_array("float32") |> np$expand_dims(0L),
  boxes = input_box |> np_array("float32") |> np$expand_dims(c(0L, 1L))         # <1>
))

display_image(image)
display_box(input_box)
display_mask(outputs$masks)



