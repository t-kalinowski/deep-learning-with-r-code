library(keras3)


library(keras3)
model <- load_model("convnet_from_scratch_with_augmentation.keras")


img_path <- get_file(                                                           # <1>
  fname="cat.jpg",                                                              # <1>
  origin="https://img-datasets.s3.amazonaws.com/cat.jpg"                        # <1>
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


display_image <- function(x, ..., max = 255, margin = 0) {
  par(mar = rep(margin, 4))

  x |> as.array() |> drop() |>
    as.raster(max = max) |>
    plot(..., interpolate = FALSE)
}

img |> display_image()


is_conv_layer <- function(x) inherits(x, keras$layers$Conv2D)
is_pooling_layer <- function(x) inherits(x, keras$layers$MaxPool2D)

layer_outputs <- list()
for (layer in model$layers)                                                     # <1>
  if (is_conv_layer(layer) || is_pooling_layer(layer))                          # <1>
    layer_outputs[[layer$name]] <- layer$output                                 # <1>

activation_model <-                                                             # <2>
  keras_model(inputs = model$input,
              outputs = layer_outputs)


activations <- predict(activation_model, img)                                   # <1>
str(activations)


first_layer_activation <- activations[[ names(layer_outputs)[1] ]]
dim(first_layer_activation)


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

plot_activations(first_layer_activation[, , , 1])
plot_activations(first_layer_activation[, , , 2])
plot_activations(first_layer_activation[, , , 3])
plot_activations(first_layer_activation[, , , 4])
plot_activations(first_layer_activation[, , , 5])
plot_activations(first_layer_activation[, , , 6])
plot_activations(first_layer_activation[, , , 7])
plot_activations(first_layer_activation[, , , 8])


for (layer_name in names(activations)) {                                        # <1>
  layer_activation <- activations[[layer_name]]                                 # <1>

  n_features <- dim(layer_activation)[[4]]                                      # <2>

  par(mfrow = n2mfrow(n_features, asp = 1.75),                                  # <3>
      mar = rep(.1, 4), oma = c(0, 0, 1.5, 0))                                  # <3>

  for (j in 1:n_features)                                                       # <4>
    plot_activations(layer_activation[, , , j])                                 # <4>
  title(main = layer_name, outer = TRUE)                                        # <5>
}


model <- application_xception(
  weights = "imagenet",
  include_top = FALSE                                                           # <1>
)


unlist(lapply(model$layers, \(layer) {
  if (inherits(layer, keras$layers$Conv2D) ||
      inherits(layer, keras$layers$SeparableConv2D))
    layer$name
}))


layer_name <- "block3_sepconv1"                                                 # <1>
layer <- model |> get_layer(name = layer_name)                                  # <2>
feature_extractor <-
  keras_model(inputs = model$input,                                             # <3>
              outputs = layer$output)


activation <- img |>
  application_preprocess_inputs(model = model) |>
  feature_extractor()


compute_loss <- function(image, filter_index) {                                 # <1>
  activation <- feature_extractor(image)
  filter_activation <- activation@r[, 3:-3, 3:-3, filter_index]                 # <2>
  op_mean(filter_activation)                                                    # <3>
}


model <- application_xception(weights = "imagenet", include_top = FALSE)

layer_name <- "block3_sepconv1"
layer <- model |> get_layer(name = layer_name)
feature_extractor <- keras_model(inputs = model$input, outputs = layer$output)

compute_loss <- function(image, filter_index) {
  activation <- feature_extractor(image)
  filter_activation <- activation@r[, 3:-3, 3:-3, filter_index]
  op_mean(filter_activation)
}


jax <- reticulate::import("jax")

grad_fn <- jax$grad(compute_loss)

gradient_ascent_step <- jax$jit(function(image, filter_index, learning_rate) {
  grads <- grad_fn(image, filter_index)
  grads <- op_normalize(grads)
  image + (learning_rate * grads)
})


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


generate_filter_pattern(filter_index = 2L) |>
  deprocess_image() |>
  display_image()


par(mfrow = c(8, 8))
for (i in seq_len(64)) {
  generate_filter_pattern(filter_index = i) |>
    deprocess_image() |>
    display_image(margin = .1)
}


model <- application_xception(weights = "imagenet")                             # <1>
preprocess_input <- application_preprocess_inputs(model)


img_path <- get_file(
  fname = "elephant.jpg",                                                       # <1>
  origin = "https://img-datasets.s3.amazonaws.com/elephant.jpg"                 # <1>
)

get_image_tensor <- function(img_path, target_size = NULL) {
  img_path |>
    image_load(target_size = target_size) |>                                    # <2>
    image_to_array(dtype = "float32") |>                                        # <3>
    op_expand_dims(1) |>                                                        # <4>
    preprocess_input()                                                          # <5>
}

image_tensor <- get_image_tensor(img_path, target_size = c(299, 299))


preds <- predict(model, image_tensor)
application_decode_predictions(model, preds, top = 3)


which.max(preds[1, ])


last_conv_layer_name <- "block14_sepconv2_act"
classifier_layer_names <- c("avg_pool", "predictions")
last_conv_layer <- model |> get_layer(last_conv_layer_name)
last_conv_layer_model <- keras_model(model$inputs,
                                     last_conv_layer$output)


classifier_input <- keras_input(batch_shape = last_conv_layer$output$shape)

x <- classifier_input
for (layer_name in classifier_layer_names)
  x <- get_layer(model, layer_name)(x)

classifier_model <- keras_model(classifier_input, x)


torch <- reticulate::import("jax")

loss_fn <- function(last_conv_layer_output) {                                   # <1>
  preds <- classifier_model(last_conv_layer_output)
  top_pred_index <- op_argmax(preds@r[1])
  top_class_channel <- preds[, top_pred_index]
  top_class_channel@r[1]                                                        # <2>
}
grad_fn <- jax$grad(loss_fn)                                                    # <3>

get_top_class_gradients <- function(image_tensor) {
  last_conv_layer_output <- last_conv_layer_model(image_tensor)
  grads <- grad_fn(last_conv_layer_output)                                      # <4>
  list(grads, last_conv_layer_output)
}


.[grads, last_conv_layer_output] <- get_top_class_gradients(image_tensor)

pooled_grads <- op_mean(grads, axis = c(1, 2, 3), keepdims = TRUE)              # <1>
output <- last_conv_layer_output * pooled_grads                                 # <2>
heatmap <- op_mean(output@r[1], axis = -1)                                      # <3>


plot_activations(heatmap)


palette <- hcl.colors(256, palette = "Spectral", alpha = .4, rev = TRUE)
heatmap <- as.array(heatmap)
heatmap[] <- palette[cut(heatmap, 256)]
heatmap <- as.raster(heatmap)

img <- image_load(img_path) |> image_to_array()
display_image(img)
rasterImage(heatmap, 0, 0, ncol(img), nrow(img), interpolate = FALSE)



