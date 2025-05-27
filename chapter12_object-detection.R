library(dplyr, warn.conflicts = FALSE)
library(stringr)
library(xml2)
library(reticulate)
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(tfdatasets, exclude = "shape")
library(keras3)
py_require("matplotlib")
py_require("opencv-python")
py_require("keras-hub==0.18.1")

use_backend("tensorflow")


VOC_url <- \(data) sprintf(
  "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOC%s_06-Nov-2007.tar", data
)
VOC_train_val <- get_file(origin = VOC_url("trainval"))
VOC_test <- get_file(origin = VOC_url("test"))


BASE_DIR       <- fs::path("VOCdevkit", "VOC2007")
IMAGE_DIR      <- BASE_DIR / "JPEGImages"
ANNOTATION_DIR <- BASE_DIR / "Annotations"
IMAGESET_DIR   <- BASE_DIR / "ImageSets" / "Main"

CLASSES <- c(
  "aeroplane",
  "bicycle",
  "bird",
  "boat",
  "bottle",
  "bus",
  "car",
  "cat",
  "chair",
  "cow",
  "diningtable",
  "dog",
  "horse",
  "motorbike",
  "person",
  "pottedplant",
  "sheep",
  "sofa",
  "train",
  "tvmonitor"
)


list.files(IMAGESET_DIR)


read_example <- function(image_id) {
  image_path <- tf$strings$join(list(IMAGE_DIR, "/", image_id, ".jpg"))
  csv_path   <-  tf$strings$join(list(ANNOTATION_DIR, "/", image_id, ".csv"))

  image <- image_path |>
    tf$io$read_file() |>
    tf$io$decode_jpeg(channels = 3L) |>
    tf$ensure_shape(shape(NA, NA, 3))

  csv_data <- csv_path |>
    tf$io$read_file() |>
    tf$strings$regex_replace("\n$", "") |> #<1> drop last newline
    tf$strings$split("\n") |>
    tf$io$decode_csv(list(
      tf$constant(list(), dtype = tf$int32), # class_idx
      tf$constant(list(), dtype = tf$float32), # ymin
      tf$constant(list(), dtype = tf$float32), # xmin
      tf$constant(list(), dtype = tf$float32), # ymax
      tf$constant(list(), dtype = tf$float32)  # xmax
    ))

  .[class_idx, ymin, xmin, ymax, xmax] <- csv_data

  bbox <- tf$stack(list(ymin, xmin, ymax, xmax), axis = -1L)  |>
    tf$ensure_shape(shape(NA, 4))

  label <- class_idx |> tf$ensure_shape(shape(NA))

  list(image = image,
       objects = list(bbox = bbox, label = label))
}


split <- "test"
split_file <- fs::path(IMAGESET_DIR, split, ext = "txt")
image_id <- readLines(split_file, 1) |> tf$constant()

read_example(image_id) |> str()


get_dataset <- function(split, shuffle_files = TRUE, shuffle_buffer_size = 1000) {

  split_file <- fs::path(IMAGESET_DIR, split, ext = "txt")
  ds <- text_line_dataset(split_file, num_parallel_reads = 12)

  if (shuffle_files)
    ds <- ds |> dataset_shuffle(shuffle_buffer_size)

  ds <- ds |>
    dataset_map(read_example, num_parallel_calls = 12) |>
    dataset_prefetch()

  ds
}

train_ds <- get_dataset("trainval", shuffle_files = TRUE)
eval_ds <- get_dataset("test", shuffle_files = TRUE)

train_ds |>
  as_iterator() |> iter_next() |> str()


example  <- train_ds |> dataset_batch(1) |> as_iterator() |> iter_next()

par(mar = c(0,0,0,0))

keras$visualization$plot_bounding_box_gallery(
  example$image,
  bounding_box_format = "rel_yxyx",
  y_true = list(
    boxes = example$objects$bbox, #|> op_convert_to_numpy(),
    labels = example$objects$label #|> op_convert_to_numpy()
  ),
  scale = 8,
  class_mapping = CLASSES
)


BBOX_FORMAT <- "yxyx"

parse_record <- function(record) {
  image <- record$image
  .[h, w, depth] <- tf$shape(image) |> tf$unstack(3L)
  rel_boxes <- record$objects$bbox
  abs_boxes <- keras$utils$bounding_boxes$convert_format(
    rel_boxes,
    source = "rel_yxyx",
    target = BBOX_FORMAT,
    height = h,
    width = w
  )
  labels <- record$objects$label

  list(images = image,
       bounding_boxes = list(boxes = abs_boxes, labels = labels))
}

record  <- train_ds |> as_iterator() |> iter_next()


envir::import_from(keras$visualization, plot_bounding_box_gallery)
IMAGE_SIZE <- shape(640, 640)
BATCH_SIZE <- 4

resizing <- layer_resizing(
  height = IMAGE_SIZE[[1]],
  width = IMAGE_SIZE[[2]],
  interpolation = "bilinear",
  pad_to_aspect_ratio = TRUE,
  bounding_box_format = BBOX_FORMAT,
)

max_box_layer <- layer_max_num_bounding_boxes(
  max_number = 100,
  bounding_box_format = BBOX_FORMAT
)

data_augmentation_layers <- list(
  layer_random_flip(mode = "horizontal", bounding_box_format = BBOX_FORMAT)
)


prepare_dataset <- function(ds, batch_size = 4) {

  ds <- ds |>
    dataset_map(parse_record, num_parallel_calls = 8) |>
    dataset_map(resizing, num_parallel_calls = 8)

  for (layer in data_augmentation_layers)
    ds <- ds |> dataset_map(layer, num_parallel_calls = NULL)

  ds <- ds |>
    dataset_map(max_box_layer, num_parallel_calls = 8) |>
    dataset_batch(batch_size, drop_remainder = TRUE) |>
    dataset_prefetch()

  ds

}

train_ds_prepared <- prepare_dataset(train_ds, batch_size=BATCH_SIZE)
eval_ds_prepared <-  prepare_dataset(eval_ds, batch_size=BATCH_SIZE)

first_images_prepared <- train_ds_prepared |> as_iterator() |> iter_next()

plot_bounding_box_gallery(
  first_images_prepared$images,
  bounding_box_format = BBOX_FORMAT,
  y_true = first_images_prepared$bounding_boxes,
  scale = 4,
  class_mapping = py_dict(seq_along(CLASSES), CLASSES)
)


library(keras3)

keras_hub <- reticulate::import("keras_hub")
model <- keras_hub$models$ImageObjectDetector$
  from_preset("retinanet_resnet50_fpn_coco")


model_with_random_head <- keras_hub$models$ImageObjectDetector$from_preset(
  "retinanet_resnet50_fpn_coco",
  num_classes = length(CLASSES)
)


split_labels <- function(x) {
  list(x$images,
       list(
         boxes =  x$bounding_boxes$boxes,
         classes = x$bounding_boxes$labels
       ))
}

train_ds_prepared <- train_ds_prepared |> dataset_map(split_labels)
eval_ds_prepared <- eval_ds_prepared |> dataset_map(split_labels)

callbacks = list(
  callback_model_checkpoint(
    "pascal_voc_detection.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)


model <- load_model("pascal_voc_detection.keras")

.[images, gt_boxes] <- iter_next(as_iterator(eval_ds_prepared))
predictions <- model |> predict(images)


plot_bounding_box_gallery(
  images,
  bounding_box_format = BBOX_FORMAT,
  y_true = list(boxes = gt_boxes$boxes, labels = gt_boxes$classes),
  y_pred = list(boxes = predictions$boxes, labels = predictions$classes),
  scale = 8,
  class_mapping = py_dict(seq_along(CLASSES), CLASSES)
)

