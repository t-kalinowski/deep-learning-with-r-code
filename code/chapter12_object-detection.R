# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "fs", "httr2", "jpeg", "jsonlite", "tfdatasets", "yyjsonr")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
reticulate::py_require(c("keras-hub", "matplotlib"))
keras_hub <- import("keras_hub")


# ----------------------------------------------------------------------
#| lst-cap: Downloading the 2017 COCO dataset
library(keras3)
py_require("keras-hub")
keras_hub <- import("keras_hub")

images_path <- get_file(
  "coco",
  "http://images.cocodataset.org/zips/train2017.zip",
  extract = TRUE
)

annotations_path <- get_file(
  "annotations",
  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
  extract = TRUE
)


# ----------------------------------------------------------------------
#| eval: false
# # raw_annotations <- jsonlite::read_json(
# #   fs::path(annotations_path, "annotations/instances_train2017.json")
# # )
# 
# full_annotations <- raw_annotations$annotations |>
#   lapply(\(x) {
#     x$segmentation %<>% list()
#     x$bbox %<>% list()
#     list(
#       image_id = x$image_id,
#       category_id = x$category_id,
#       bbox = list(setNames(x$bbox, c("left", "top", "width", "height")))
#     )
#   }) |>
#   bind_rows()
# 
# full_images <- raw_annotations$images |>                                        # <1>
#   bind_rows() |> as_tibble() |>                                                 # <1>
#   rename(image_id = id)
# 
#  raw_annotations$annotations[[1]] |> str(list.len = 100)


# ----------------------------------------------------------------------
#| eval: true
#| lst-cap: Parsing the COCO data
library(dplyr, warn.conflicts = FALSE)

raw_annotations <-
  fs::path(annotations_path, "annotations/instances_train2017.json") |>
  yyjsonr::read_json_file() |>
  lapply(\(x) if (is.data.frame(x)) as_tibble(x) else x)

images <- raw_annotations$images |>                                             # <1>
  select(file_name, height, width, image_id = id)

annotations <- raw_annotations$annotations |>
  summarise(                                                                    # <2>
    .by = image_id,
    labels = list(category_id),
    boxes = list({
      boxes <- matrix(unlist(bbox), byrow = TRUE, ncol = 4)
      colnames(boxes) <- c("left", "top", "width", "height")
      boxes
    })
  )

scale_boxes <- function(boxes, height, width) {                                 # <3>
  if (width > height) {                                                         # <4>
    boxes[, "top"] <- boxes[, "top"] + (width - height) / 2
    scale <- width
  } else if (height > width) {
    boxes[, "left"] <- boxes[, "left"] + (height - width) / 2
    scale <- height
  } else {
    scale <- width
  }

  boxes / scale
}

metadata <-
  inner_join(annotations, images, by = "image_id") |>
  mutate(
    boxes = Map(scale_boxes, boxes, height, width),
    labels,
    path = fs::path(images_path, "train2017", file_name),
    .keep = "none"
  )

rm(raw_annotations, annotations, images)                                        # <5>


# ----------------------------------------------------------------------
#| lst-cap: Inspecting the COCO data
metadata


# ----------------------------------------------------------------------
range(sapply(metadata$boxes, nrow))
max(unlist(metadata$labels))


# ----------------------------------------------------------------------
example <- metadata[436, ] |> lapply(`[[`, 1)
example$labels |> sapply(keras_hub$utils$coco_id_to_name)


# ----------------------------------------------------------------------
#| lst-cap: Visualizing a COCO image with box annotations
label_to_color <- function(label, alpha = 1) {
  ifelse(label == 0, "gray", hsv(
    h = (label * 0.618) %% 1,                                                   # <1>
    s = 0.5,
    v = 0.9,
    alpha = alpha
  ))
}

draw_image <- function(image_path, show_padding = FALSE) {                      # <2>
  img <- jpeg::readJPEG(image_path, native = TRUE)
  par(mar = rep(1.1, 4), xaxs = "i", yaxs = "i")
  plot.new()
  if (nrow(img) > ncol(img)) {                                                  # <3>
    x_pad <- (nrow(img) - ncol(img)) / nrow(img) / 2                            # <3>
    plot.window(
      xlim = if (show_padding) 0:1 else c(x_pad, 1 - x_pad),                    # <3>
      ylim = 0:1,
      asp = 1
    )
    rasterImage(img, x_pad, 0, 1 - x_pad, 1)
  } else if (ncol(img) > nrow(img)) {
    y_pad <- (ncol(img) - nrow(img)) / ncol(img) / 2                            # <3>
    plot.window(
      xlim = 0:1,
      ylim = if (show_padding) 0:1 else c(y_pad, 1 - y_pad),                    # <3>
      asp = 1
    )
    rasterImage(img, 0, y_pad, 1, 1 - y_pad)
  } else {
    plot.window(0:1, 0:1, asp = 1)
    rasterImage(img, 0, 0, 1, 1)
  }
}

draw_boxes <- function(boxes, text, color) {
  boxes <- as.data.frame(as.matrix(boxes))
  stopifnot(c("left", "top", "width", "height") %in% names(boxes))
  rect(                                                                         # <4>
    xleft = boxes$left, xright = boxes$left + boxes$width,
    ytop = 1 - boxes$top, ybottom = 1 - boxes$top - boxes$height,
    border = color, lwd = 3
  )
  rect(                                                                         # <5>
    xleft = boxes$left, xright = boxes$left + strwidth(text, cex = 1.4),
    ytop = 1 - boxes$top + strheight(text, cex = 1.4), ybottom = 1 - boxes$top,
    col = color, border = color, lwd = 3
  )
  text(boxes$left, 1 - boxes$top, text,                                         # <6>
       adj = c(0, 0), col = "black", cex = 1.4, xpd = NA)
}


# ----------------------------------------------------------------------
#| fig-cap: "YOLO outputs a bounding box prediction and class label for each image region.^[Image from the COCO 2017 dataset, <https://cocodataset.org/>. Image from Flickr, <http://farm8.staticflickr.com/7250/7520201840_3e01349e3f_z.jpg>, CC BY 2.0 <https://creativecommons.org/licenses/by/2.0/>.]"
example <- metadata[436, ] |> lapply(`[[`, 1)
draw_image(example$path)
draw_boxes(
  boxes = example$boxes,
  text =  example$labels |> sapply(keras_hub$utils$coco_id_to_name),
  color = example$labels |> label_to_color()
)


# ----------------------------------------------------------------------
metadata <- metadata |>
  filter(lengths(labels) <= 4) |>
  slice(sample.int(n()))


# ----------------------------------------------------------------------
#| lst-cap: Loading the ResNet model
image_size <- c(448, 448)

backbone <- keras_hub$models$Backbone$from_preset(
  "resnet_50_imagenet"
)
preprocessor <- keras_hub$layers$ImageConverter$from_preset(
  "resnet_50_imagenet",
  image_size = shape(image_size)
)


# ----------------------------------------------------------------------
#| lst-cap: Attaching a YOLO prediction head
grid_size <- 6L
num_labels <- 91L

inputs <- keras_input(shape = c(image_size, 3))
x <- inputs |>
  backbone() |>
  layer_conv_2d(512, c(3, 3), strides = c(2, 2)) |>                             # <1>
  layer_flatten() |>                                                            # <1>
  layer_dense(2048, activation = "relu",
              kernel_initializer = "glorot_normal") |>                          # <2>
  layer_dropout(0.5) |>                                                         # <2>
  layer_dense(grid_size * grid_size * (num_labels + 5)) |>                      # <2>
  layer_reshape(c(grid_size, grid_size, num_labels + 5))                        # <3>

box_predictions <- x@r[.., 1:5]                                                 # <4>
class_predictions <- layer_activation_softmax(x@r[.., 6:NA])                    # <4>
outputs <- list(box = box_predictions, class = class_predictions)
model <- keras_model(inputs, outputs)


# ----------------------------------------------------------------------
model


# ----------------------------------------------------------------------
to_grid <- function(box) {
  .[x, y, w, h] <- box
  .[cx, cy] <- c(x + w / 2, y + h / 2) * grid_size
  .[ix, iy] <- as.integer(c(cx, cy))
  grid_box <- c(cx - ix, cy - iy, w, h)
  list(cell = c(ix, iy), box = grid_box)
}

from_grid <- function(cell, box) {
  .[xi, yi] <- cell
  .[x, y, w, h] <- box
  x <- (xi + x) / grid_size - w / 2
  y <- (yi + y) / grid_size - h / 2
  cbind(left = x, top = y, width = w, height = h)
}


# ----------------------------------------------------------------------
#| lst-cap: Creating the YOLO targets
class_array <- array(0L, c(nrow(metadata), grid_size, grid_size))
box_array <- array(0, c(nrow(metadata), grid_size, grid_size, 5))

clamp_to_grid <- \(val) val |> pmax(1L) |> pmin(grid_size)

for (img_i in seq_len(nrow(metadata))) {
  sample <- metadata[img_i, ] |> lapply(`[[`, 1)
  for (box_i in seq_len(nrow(sample$boxes))) {
    box <- sample$boxes[box_i, ]
    label <- sample$labels[box_i]
    .[x, y, w, h] <- box
    .[left, bottom] <- clamp_to_grid(floor(c(x, y) * grid_size) + 1L)
    .[right, top] <- clamp_to_grid(ceiling(c(x + w, y + h) * grid_size))
    class_array[img_i, bottom:top, left:right] <- label                         # <1>
  }
}


for (img_i in seq_len(nrow(metadata))) {
  sample <- metadata[img_i, ] |> lapply(`[[`, 1)
  for (box_i in seq_len(nrow(sample$boxes))) {
    box <- sample$boxes[box_i, ]
    label <- sample$labels[box_i]
    .[.[xi, yi], grid_box] <- to_grid(box)                                      # <2>
    box_array[img_i, yi + 1, xi + 1, ] <- c(grid_box, 1)
    class_array[img_i, yi + 1, xi + 1] <- label                                 # <3>
  }
}


# ----------------------------------------------------------------------
draw_prediction <- function(image, boxes, classes, cutoff = NULL) {
  draw_image(image)

  for (yi in seq_len(grid_size)) {                                              # <1>
    for (xi in seq_len(grid_size)) {
      label <- classes[yi, xi]
      col  <- if (label == 0) NA else label_to_color(label, alpha = 0.4)
      .[x0, y0] <- (c(xi, yi) - 1) / grid_size
      rect(
        xleft = x0, xright = x0 + 1 / grid_size,
        ytop = 1 - (y0 + 1 / grid_size), ybottom = 1 - y0,
        col = col, border = "black", lwd = 2
      )
    }
  }

  for (yi in seq_len(grid_size)) {                                              # <2>
    for (xi in seq_len(grid_size)) {
      cell       <- boxes[yi, xi, ]
      confidence <- cell[5]
      if (is.null(cutoff) || confidence >= cutoff) {
        grid_box <- cell[1:4]
        box <- from_grid(c(xi - 1, yi - 1), grid_box)
        label <- classes[yi, xi]
        color <- label_to_color(label)
        name <- keras_hub$utils$coco_id_to_name(label)
        draw_boxes(box, sprintf("%s %.2f", name, max(confidence, 0)), color)
      }
    }
  }
}


# ----------------------------------------------------------------------
i <- 1
draw_prediction(
  metadata$path[i],
  box_array[i, , , ],
  class_array[i, , ],
  cutoff = 1
)


# ----------------------------------------------------------------------
#| eval: false
# for(i in 1:200) {
#   draw_prediction(
#     metadata$path[i],
#     box_array[i, , , ],
#     class_array[i, , ],
#     cutoff = 1
#   )
#   title(i)
# }


# ----------------------------------------------------------------------
#| lst-cap: Creating a dataset to train on
library(tfdatasets, exclude = c("shape"))

images <- metadata$path |> normalizePath() |>
  tensor_slices_dataset() |>
  dataset_map(\(path) {
    path |>
      tf$io$read_file() |>                                                      # <1>
      tf$image$decode_jpeg(channels = 3L) |>                                    # <1>
      preprocessor()                                                            # <1>
  }, num_parallel_calls = 8)

labels <-  tensor_slices_dataset(list(
  box = box_array, class = class_array
))

dataset <- zip_datasets(images, labels) |>                                      # <2>
  dataset_batch(16) |> dataset_prefetch(2)

val_dataset <- dataset |> dataset_take(500)                                     # <3>
train_dataset <- dataset |> dataset_skip(500)                                   # <3>


# ----------------------------------------------------------------------
#| lst-cap: Computing IoU for two boxes
intersection <- function(box1, box2) {                                          # <1>
  .[cx1, cy1, w1, h1, conf] <- op_unstack(box1, 5, axis = -1)                   # <2>
  .[cx2, cy2, w2, h2, conf] <- op_unstack(box2, 5, axis = -1)                   # <2>

  left   <- op_maximum(cx1 - w1 / 2, cx2 - w2 / 2)
  bottom <- op_maximum(cy1 - h1 / 2, cy2 - h2 / 2)
  right  <- op_minimum(cx1 + w1 / 2, cx2 + w2 / 2)
  top    <- op_minimum(cy1 + h1 / 2, cy2 + h2 / 2)

  op_maximum(0.0, right - left) * op_maximum(0.0, top - bottom)
}

intersection_over_union <- function(box1, box2) {                               # <3>
  .[cx1, cy1, w1, h1, conf] <- op_unstack(box1, 5, axis = -1)
  .[cx2, cy2, w2, h2, conf] <- op_unstack(box2, 5, axis = -1)

  inter <- intersection(box1, box2)
  a1    <- op_maximum(w1, 0.0) * op_maximum(h1, 0.0)
  a2    <- op_maximum(w2, 0.0) * op_maximum(h2, 0.0)
  union <- a1 + a2 - inter

  op_divide_no_nan(inter, union)
}


# ----------------------------------------------------------------------
#| lst-cap: Defining the YOLO bounding box loss
signed_sqrt <- function(x) {
  op_sign(x) * op_sqrt(op_abs(x) + config_epsilon())
}

box_loss <- function(true, pred) {
  unpack <- \(x) list(x[.., 1:2], x[.., 3:4], x[.., 5:NA])                      # <1>
  .[xy_true, wh_true, conf_true] <- unpack(true)                                # <1>
  .[xy_pred, wh_pred, conf_pred] <- unpack(pred)                                # <1>

  no_object <- conf_true == 0                                                   # <2>

  xy_error <- op_square(xy_true - xy_pred)                                      # <3>
  wh_error <- op_square(signed_sqrt(wh_true) - signed_sqrt(wh_pred))            # <3>

  iou <- intersection_over_union(true, pred)                                    # <4>
  conf_target <- op_where(no_object, 0, op_expand_dims(iou, -1))                # <4>
  conf_error <- op_square(conf_target - conf_pred)                              # <4>

  error <- op_concatenate(axis = -1, list(                                      # <5>
    op_where(no_object, 0, xy_error  * 5),                                      # <5>
    op_where(no_object, 0, wh_error  * 5),                                      # <5>
    op_where(no_object, conf_error * 0.5, conf_error)                           # <5>
  ))

  op_sum(error, axis = c(2, 3, 4))                                              # <6>
}


# ----------------------------------------------------------------------
#| lst-cap: Training the YOLO model
model |> compile(
  optimizer = optimizer_adam(2e-4),
  loss = list(box = box_loss, class = "sparse_categorical_crossentropy")
)
model |> fit(
  train_dataset,
  validation_data = val_dataset,
  epochs = 4
)


# ----------------------------------------------------------------------
.[x, y] <- val_dataset |> dataset_rebatch(1) |>                                 # <1>
  as_iterator() |> iter_next()
preds <- model(x)
boxes <- preds$box@r[1, ..] |> as.array()
classes <- preds$class@r[1, ..] |>
  op_argmax(axis = -1, zero_indexed = TRUE) |>                                  # <2>
  as.array()
path <- metadata[1,]$path                                                       # <3>
draw_prediction(path, boxes, classes, cutoff = 0.1)


# ----------------------------------------------------------------------
draw_prediction(path, boxes, classes, cutoff = NULL)



# ----------------------------------------------------------------------
library(keras3)
reticulate::py_require(c("keras-hub", "matplotlib"))
keras_hub <- import("keras_hub")
py_require("keras-hub")
library(dplyr, warn.conflicts = FALSE)
library(tfdatasets, exclude = c("shape"))


# ----------------------------------------------------------------------
image_size <- c(448, 448)
grid_size <- 6L
num_labels <- 91L
i <- 1


# ----------------------------------------------------------------------
scale_boxes <- function(boxes, height, width) {                                 # <3>
  if (width > height) {                                                         # <4>
    boxes[, "top"] <- boxes[, "top"] + (width - height) / 2
    scale <- width
  } else if (height > width) {
    boxes[, "left"] <- boxes[, "left"] + (height - width) / 2
    scale <- height
  } else {
    scale <- width
  }

  boxes / scale
}

label_to_color <- function(label, alpha = 1) {
  ifelse(label == 0, "gray", hsv(
    h = (label * 0.618) %% 1,                                                   # <1>
    s = 0.5,
    v = 0.9,
    alpha = alpha
  ))
}

draw_image <- function(image_path, show_padding = FALSE) {                      # <2>
  img <- jpeg::readJPEG(image_path, native = TRUE)
  par(mar = rep(1.1, 4), xaxs = "i", yaxs = "i")
  plot.new()
  if (nrow(img) > ncol(img)) {                                                  # <3>
    x_pad <- (nrow(img) - ncol(img)) / nrow(img) / 2                            # <3>
    plot.window(
      xlim = if (show_padding) 0:1 else c(x_pad, 1 - x_pad),                    # <3>
      ylim = 0:1,
      asp = 1
    )
    rasterImage(img, x_pad, 0, 1 - x_pad, 1)
  } else if (ncol(img) > nrow(img)) {
    y_pad <- (ncol(img) - nrow(img)) / ncol(img) / 2                            # <3>
    plot.window(
      xlim = 0:1,
      ylim = if (show_padding) 0:1 else c(y_pad, 1 - y_pad),                    # <3>
      asp = 1
    )
    rasterImage(img, 0, y_pad, 1, 1 - y_pad)
  } else {
    plot.window(0:1, 0:1, asp = 1)
    rasterImage(img, 0, 0, 1, 1)
  }
}

draw_boxes <- function(boxes, text, color) {
  boxes <- as.data.frame(as.matrix(boxes))
  stopifnot(c("left", "top", "width", "height") %in% names(boxes))
  rect(                                                                         # <4>
    xleft = boxes$left, xright = boxes$left + boxes$width,
    ytop = 1 - boxes$top, ybottom = 1 - boxes$top - boxes$height,
    border = color, lwd = 3
  )
  rect(                                                                         # <5>
    xleft = boxes$left, xright = boxes$left + strwidth(text, cex = 1.4),
    ytop = 1 - boxes$top + strheight(text, cex = 1.4), ybottom = 1 - boxes$top,
    col = color, border = color, lwd = 3
  )
  text(boxes$left, 1 - boxes$top, text,                                         # <6>
       adj = c(0, 0), col = "black", cex = 1.4, xpd = NA)
}

to_grid <- function(box) {
  .[x, y, w, h] <- box
  .[cx, cy] <- c(x + w / 2, y + h / 2) * grid_size
  .[ix, iy] <- as.integer(c(cx, cy))
  grid_box <- c(cx - ix, cy - iy, w, h)
  list(cell = c(ix, iy), box = grid_box)
}

from_grid <- function(cell, box) {
  .[xi, yi] <- cell
  .[x, y, w, h] <- box
  x <- (xi + x) / grid_size - w / 2
  y <- (yi + y) / grid_size - h / 2
  cbind(left = x, top = y, width = w, height = h)
}

draw_prediction <- function(image, boxes, classes, cutoff = NULL) {
  draw_image(image)

  for (yi in seq_len(grid_size)) {                                              # <1>
    for (xi in seq_len(grid_size)) {
      label <- classes[yi, xi]
      col  <- if (label == 0) NA else label_to_color(label, alpha = 0.4)
      .[x0, y0] <- (c(xi, yi) - 1) / grid_size
      rect(
        xleft = x0, xright = x0 + 1 / grid_size,
        ytop = 1 - (y0 + 1 / grid_size), ybottom = 1 - y0,
        col = col, border = "black", lwd = 2
      )
    }
  }

  for (yi in seq_len(grid_size)) {                                              # <2>
    for (xi in seq_len(grid_size)) {
      cell       <- boxes[yi, xi, ]
      confidence <- cell[5]
      if (is.null(cutoff) || confidence >= cutoff) {
        grid_box <- cell[1:4]
        box <- from_grid(c(xi - 1, yi - 1), grid_box)
        label <- classes[yi, xi]
        color <- label_to_color(label)
        name <- keras_hub$utils$coco_id_to_name(label)
        draw_boxes(box, sprintf("%s %.2f", name, max(confidence, 0)), color)
      }
    }
  }
}

intersection <- function(box1, box2) {                                          # <1>
  .[cx1, cy1, w1, h1, conf] <- op_unstack(box1, 5, axis = -1)                   # <2>
  .[cx2, cy2, w2, h2, conf] <- op_unstack(box2, 5, axis = -1)                   # <2>

  left   <- op_maximum(cx1 - w1 / 2, cx2 - w2 / 2)
  bottom <- op_maximum(cy1 - h1 / 2, cy2 - h2 / 2)
  right  <- op_minimum(cx1 + w1 / 2, cx2 + w2 / 2)
  top    <- op_minimum(cy1 + h1 / 2, cy2 + h2 / 2)

  op_maximum(0.0, right - left) * op_maximum(0.0, top - bottom)
}

intersection_over_union <- function(box1, box2) {                               # <3>
  .[cx1, cy1, w1, h1, conf] <- op_unstack(box1, 5, axis = -1)
  .[cx2, cy2, w2, h2, conf] <- op_unstack(box2, 5, axis = -1)

  inter <- intersection(box1, box2)
  a1    <- op_maximum(w1, 0.0) * op_maximum(h1, 0.0)
  a2    <- op_maximum(w2, 0.0) * op_maximum(h2, 0.0)
  union <- a1 + a2 - inter

  op_divide_no_nan(inter, union)
}

signed_sqrt <- function(x) {
  op_sign(x) * op_sqrt(op_abs(x) + config_epsilon())
}

box_loss <- function(true, pred) {
  unpack <- \(x) list(x[.., 1:2], x[.., 3:4], x[.., 5:NA])                      # <1>
  .[xy_true, wh_true, conf_true] <- unpack(true)                                # <1>
  .[xy_pred, wh_pred, conf_pred] <- unpack(pred)                                # <1>

  no_object <- conf_true == 0                                                   # <2>

  xy_error <- op_square(xy_true - xy_pred)                                      # <3>
  wh_error <- op_square(signed_sqrt(wh_true) - signed_sqrt(wh_pred))            # <3>

  iou <- intersection_over_union(true, pred)                                    # <4>
  conf_target <- op_where(no_object, 0, op_expand_dims(iou, -1))                # <4>
  conf_error <- op_square(conf_target - conf_pred)                              # <4>

  error <- op_concatenate(axis = -1, list(                                      # <5>
    op_where(no_object, 0, xy_error  * 5),                                      # <5>
    op_where(no_object, 0, wh_error  * 5),                                      # <5>
    op_where(no_object, conf_error * 0.5, conf_error)                           # <5>
  ))

  op_sum(error, axis = c(2, 3, 4))                                              # <6>
}


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
url <- paste0(
  "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/",
  "A_Sunday_on_La_Grande_Jatte%2C_Georges_Seurat%2C_1884.jpg/",
  "1280px-A_Sunday_on_La_Grande_Jatte%2C_Georges_Seurat%2C_1884.jpg"
)
path <- get_file("la_grande_jatte.jpg", origin = url)
image <- image_load(path) |> image_to_array(dtype = "float32")


# ----------------------------------------------------------------------
#| lst-cap: "Creating the `ObjectDetector` model"
detector <- keras_hub$models$ObjectDetector$from_preset(
  "retinanet_resnet50_fpn_v2_coco",
  bounding_box_format =  "rel_xywh"
)
predictions <- predict(detector, list(image))                                   # <1>


# ----------------------------------------------------------------------
str(predictions)


# ----------------------------------------------------------------------
#| fig-cap: "RetinaNet detections for _A Sunday Afternoon on the Island of La Grande Jatte_.^[Image: Georges Seurat, public domain, via Wikimedia Commons, <https://commons.wikimedia.org/wiki/File:A_Sunday_on_La_Grande_Jatte,_Georges_Seurat,_1884.jpg>.]"
#| lst-cap: Running inference with RetinaNet
draw_image(path, show_padding = FALSE)
for(i in seq_len(predictions$num_detections)) {
  box <- predictions$boxes[1, i, ] |> matrix(ncol = 4)
  colnames(box) <- c("left", "top", "width", "height")
  label <- predictions$labels[1, i]
  label_name <- keras_hub$utils$coco_id_to_name(label)
  draw_boxes(box, label_name, label_to_color(label))
}



