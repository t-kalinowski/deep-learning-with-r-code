# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "fs", "jpeg", "tfdatasets", "vctrs")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
py_require("keras-hub")
library(tfdatasets, exclude = "shape")


# ----------------------------------------------------------------------
digit_size <- 28
batch_size <- 32
image_size <- 128


# ----------------------------------------------------------------------
layer_sampler <- new_layer_class(
  classname = "Sampler",
  initialize = function(...) {
    super$initialize(...)
    self$seed_generator <- random_seed_generator()                              # <1>
    self$built <- TRUE
  },
  call = function(z_mean, z_log_var) {
    .[batch_size, z_size] <- op_shape(z_mean)
    epsilon <- random_normal(shape = op_shape(z_mean),                          # <2>
                             seed = self$seed_generator)                        # <2>
    z_mean + (op_exp(0.5 * z_log_var) * epsilon)                                # <3>
  }
)

residual_block <- function(x, width) {                                          # <1>
  .[.., n_features] <- op_shape(x)

  if (n_features == width) {
    residual <- x
  } else {
    residual <- x |> layer_conv_2d(filters = width, kernel_size = 1)
  }

  x <- x |>
    layer_batch_normalization(center = FALSE, scale = FALSE) |>
    layer_conv_2d(width, kernel_size = 3, padding = "same",
                  activation = "swish") |>
    layer_conv_2d(width, kernel_size = 3, padding = "same")

  x + residual
}

get_model <- function(image_size, widths, block_depth) {
  noisy_images <- keras_input(shape = c(image_size, image_size, 3))
  noise_rates <- keras_input(shape = c(1, 1, 1))

  x <- noisy_images |> layer_conv_2d(filters = widths[1], kernel_size = 1)
  n <- noise_rates |>
    layer_upsampling_2d(size = image_size, interpolation = "nearest")
  x <- layer_concatenate(c(x, n))

  skips <- list()

  for (width in head(widths, -1)) {                                             # <2>
    for (i in seq_len(block_depth)) {                                           # <2>
      x <- x |> residual_block(width)                                           # <2>
      skips <- c(skips, x)
    }
    x <- x |> layer_average_pooling_2d(pool_size = 2)
  }

  for (i in seq_len(block_depth)) {                                             # <3>
    x <- x |> residual_block(tail(widths, 1))                                   # <3>
  }

  for (width in rev(head(widths, -1))) {                                        # <4>
    x <- x |> layer_upsampling_2d(size = 2, interpolation = "bilinear")         # <4>
    for (i in seq_len(block_depth)) {                                           # <4>
      x <- x |> layer_concatenate(tail(skips, 1)[[1]])                          # <4>
      skips <- head(skips, -1)
      x <- x |> residual_block(width)
    }
  }

  pred_noise_masks <- x |> layer_conv_2d(                                       # <5>
    filters = 3, kernel_size = 1, kernel_initializer = "zeros"                  # <5>
  )                                                                             # <5>

  model <- keras_model(inputs = list(noisy_images, noise_rates),                # <6>
                       outputs = pred_noise_masks)                              # <6>
  model
}

diffusion_schedule <- function(diffusion_times, min_signal_rate = 0.02,
                               max_signal_rate = 0.95) {
  start_angle <- op_arccos(max_signal_rate) |> op_cast(dtype = "float32")
  end_angle <- op_arccos(min_signal_rate) |> op_cast(dtype = "float32")

  diffusion_angles <-
    start_angle + diffusion_times * (end_angle - start_angle)
  signal_rates <- op_cos(diffusion_angles)
  noise_rates <- op_sin(diffusion_angles)
  list(noise_rates = noise_rates, signal_rates = signal_rates)
}

callback_visualization <- new_callback_class(
  classname = "VisualizationCallback",
  initialize = function(diffusion_steps = 20, num_rows = 3, num_cols = 6) {
    self$diffusion_steps <- diffusion_steps
    self$num_rows <- num_rows
    self$num_cols <- num_cols
  },

  on_epoch_end = function(epoch = NULL, logs = NULL) {
    generated_images <- self$model$generate(
      num_images = self$num_rows * self$num_cols,
      diffusion_steps = self$diffusion_steps
    ) |> as.array()

    par(mfrow = c(self$num_rows, self$num_cols),
        mar = c(0, 0, 0, 0))

    for (i in seq_len(self$num_rows * self$num_cols))
      plot(as.raster(generated_images[i, , , ], max = 255))

  }
)


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
rm(list = ls()); gc(); import("gc")$collect(); gc()


# ----------------------------------------------------------------------
#| lst-cap: Creating a Stable Diffusion text-to-image model
library(keras3)
py_require("keras-hub")
keras_hub <- import("keras_hub")

.[height, width] <- c(512, 512)
task <- keras_hub$models$TextToImage$from_preset(
  "stable_diffusion_3_medium",
  image_shape = shape(height, width, 3),
  dtype = "float16"                                                             # <1>
)
prompt <- "A NASA astronaut riding an origami elephant in New York City"


# ----------------------------------------------------------------------
image <- task$generate(prompt)
par(mar = c(0, 0, 0, 0))
plot(as.raster(image, max = max(image)))


# ----------------------------------------------------------------------
image <- task$generate(list(
  prompts = prompt,
  negative_prompts = "blue color"
))
par(mar = c(0, 0, 0, 0))
plot(as.raster(image, max = max(image)))


# ----------------------------------------------------------------------
par(mfrow = c(1, 5), mar = c(0, 0, 0, 0))
for (num_steps in c(5, 10, 15, 20, 25)) {
  image <- task$generate(prompt, num_steps = num_steps)
  plot(as.raster(image, max = max(image)))
}


# ----------------------------------------------------------------------
#| output: false
#| eval: true
#| lst-cap: "Breaking down the `generate()` function"
get_text_embeddings <- function(prompt) {
  token_ids <- task$preprocessor$generate_preprocess(list(prompt))
  negative_token_ids <- task$preprocessor$generate_preprocess(list(""))         # <1>
  task$backbone$encode_text_step(token_ids, negative_token_ids)
}

denoise_with_text_embeddings <- function(
  embeddings,
  num_steps = 28L,
  guidance_scale = 7
) {
  latents <- random_normal(c(1, height %/% 8, width %/% 8, 16))                 # <2>
  for (step in seq_len(num_steps)) {
    latents <- latents |>
      task$backbone$denoise_step(
        embeddings, step, num_steps, guidance_scale
      )
  }
  task$backbone$decode_step(latents)@r[1]
}


scale_output <- function(x) {
  op_clip((x + 1) / 2, 0, 1)                                                    # <3>
}

embeddings <- get_text_embeddings(prompt)
image <- denoise_with_text_embeddings(embeddings)
image <- scale_output(image)


# ----------------------------------------------------------------------
#| eval: false
# par(mar = c(0,0,0,0))
# plot(as.raster(as.array(image)))


# ----------------------------------------------------------------------
str(embeddings)


# ----------------------------------------------------------------------
#| lst-cap: Function to interpolate text embeddings
slerp <- function(t, v1, v2) {
  .[v1, v2] <- list(v1, v2) |> lapply(op_cast, "float32")
  v1_norm  <- op_norm(op_ravel(v1))
  v2_norm  <- op_norm(op_ravel(v2))
  dot <- op_sum(v1 * v2 / (v1_norm * v2_norm))
  theta_0 <- op_arccos(dot)
  sin_theta_0 <- op_sin(theta_0)
  theta_t <- theta_0 * t
  sin_theta_t <- op_sin(theta_t)
  s0 <- op_sin(theta_0 - theta_t) / sin_theta_0
  s1 <- sin_theta_t / sin_theta_0
  s0 * v1 + s1 * v2
}

interpolate_text_embeddings <- function(e1, e2, start=0, end=1, num=10) {
  lapply(seq(start, end, length.out = num), \(t) {
    list(
      slerp(t, e1[[1]], e2[[1]]),
      e1[[2]],                                                                  # <1>
      slerp(t, e1[[3]], e2[[3]]),
      e1[[4]]                                                                   # <1>
    )
  })

}


# ----------------------------------------------------------------------
par(mar = c(0,0,0,0))
prompt1 <- "A friendly dog looking up in a field of flowers"
prompt2 <- "A horrifying, tentacled creature hovering over a field of flowers"
e1 <- get_text_embeddings(prompt1)
e2 <- get_text_embeddings(prompt2)

for (et in interpolate_text_embeddings(e1, e2, start=0.5, end=0.6, num=9)) {    # <1>
  image <- denoise_with_text_embeddings(et)
  plot(as.raster(as.array(scale_output(image))))
}


