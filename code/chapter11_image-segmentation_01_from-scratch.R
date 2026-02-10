# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "fs", "tfdatasets", "tibble", "tidyr")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
library(tfdatasets, exclude = "shape")
py_require("keras-hub")


# ----------------------------------------------------------------------
library(fs)
data_dir <- path("pets_dataset")
dir_create(data_dir)


# ----------------------------------------------------------------------
data_url <- path("http://www.robots.ox.ac.uk/~vgg/data/pets/data")

options(timeout = 3600)
for (filename in c("images.tar.gz", "annotations.tar.gz")) {
  download.file(url =  data_url / filename,
                destfile = data_dir / filename)
  untar(data_dir / filename, exdir = data_dir)
}


# ----------------------------------------------------------------------
library(dplyr, warn.conflicts = FALSE)
input_dir <- data_dir / "images"
target_dir <- data_dir / "annotations/trimaps/"

all_image_paths <- tibble(
  input = sort(dir_ls(input_dir, glob = "*.jpg")),
  target = sort(dir_ls(target_dir, glob = "*.png", all = FALSE))                # <1>
)


# ----------------------------------------------------------------------
#| lst-cap: Helper to display an image tensor
display_image <- function(x, ..., max = 255L, margin = 0) {
  par(mar = rep(margin, 4))

  x |> as.array() |> drop() |>
    as.raster(max = max) |>
    plot(..., interpolate = FALSE)
}


# ----------------------------------------------------------------------
#| fig-cap: An example image
all_image_paths$input[10] |>                                                    # <1>
  image_load() |> image_to_array() |>
  display_image()


# ----------------------------------------------------------------------
#| fig-cap: The corresponding target mask
display_target <- function(target, ..., offset = TRUE) {
  if (offset)
    target <- target - 1L                                                       # <1>
  display_image(target, max = 2L, ...)
}

all_image_paths$target[10] |>
  image_load(color_mode = "grayscale") |>                                       # <2>
  image_to_array() |>
  display_target()


# ----------------------------------------------------------------------
#| lst-cap: Preparing the dataset
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

num_val_samples <- 1000                                                         # <6>

image_paths <- all_image_paths |>
  dplyr::mutate(
    use = ifelse(sample.int(n()) > num_val_samples, "train", "val")             # <7>
  ) |>
  tidyr::nest(.by = use) |>
  tibble::deframe()

train_ds <- make_dataset(image_paths$train)                                     # <8>
val_ds   <- make_dataset(image_paths$val)                                       # <8>


# ----------------------------------------------------------------------
#| eval: false
# example_paths <- tensor_slices_dataset(image_paths) |>
#   as_iterator() |> iter_next()


# ----------------------------------------------------------------------
#| eval: false
# tensor_slices_dataset(image_paths$train) |>
#   dataset_map(function(example_paths) {
#     print(example_paths)
#     browser()
#   })


# ----------------------------------------------------------------------
batch <- train_ds |> as_iterator() |> iter_next()
str(batch)


# ----------------------------------------------------------------------
#| fig-cap: A batch of segmentation inputs and masks
.[images, targets] <- batch
par(mfrow = c(4, 8))
for (i in 1:16) {
  images@r[i] |> display_image()
  targets@r[i] |> display_target(offset = FALSE)                                # <1>
}


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
foreground_iou <- metric_iou(
  num_classes = 3,                                                              # <1>
  target_class_ids = c(0),                                                      # <2>
  name = "foreground_iou",
  sparse_y_true = TRUE,                                                         # <3>
  sparse_y_pred = FALSE,                                                        # <4>
)


# ----------------------------------------------------------------------
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = foreground_iou
)

callbacks <- list(
  callback_model_checkpoint("oxford_segmentation.keras", save_best_only = TRUE)
)


# ----------------------------------------------------------------------
history <- model |> fit(
  train_ds,
  epochs = 50,
  callbacks = callbacks,
  validation_data = val_ds
)


# ----------------------------------------------------------------------
#| fig-cap: Displaying training and validation loss curves
#| eval: true
plot(history, metrics = "loss")


# ----------------------------------------------------------------------
model <- load_model("oxford_segmentation.keras")


# ----------------------------------------------------------------------
#| fig-cap: "A test image, its predicted segmentation mask, and its target mask"
i <- 12
test_image <- image_paths$val$input[i] |>
  tf_image_load(channels = 3L, target_size = img_size)

test_mask <- image_paths$val$target[i] |>
  tf_image_load(channels = 1L, target_size = img_size) |>
  tf$subtract(1)

predicted_mask_probs <- model(test_image@r[newaxis])
predicted_mask <- op_argmax(predicted_mask_probs, axis = -1,
                            zero_indexed = TRUE)                                # <1>

par(mfrow = c(1, 3))
display_image(test_image)
display_target(predicted_mask, offset = FALSE)
display_target(test_mask, offset = FALSE)


