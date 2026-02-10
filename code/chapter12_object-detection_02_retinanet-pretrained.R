# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "fs", "httr2", "jpeg", "jsonlite", "tfdatasets", "yyjsonr")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


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


