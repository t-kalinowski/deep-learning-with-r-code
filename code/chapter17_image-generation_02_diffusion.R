# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "fs", "jpeg", "tfdatasets", "vctrs")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
py_require("keras-hub")


# ----------------------------------------------------------------------
digit_size <- 28


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


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
flowers_tgz <- get_file(
  origin = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
)


# ----------------------------------------------------------------------
untar(flowers_tgz, exdir = "flowers")


# ----------------------------------------------------------------------
library(tfdatasets, exclude = "shape")
batch_size <- 32
image_size <- 128
images_dir <- fs::path("flowers", "jpg")
dataset <- image_dataset_from_directory(
  images_dir,
  labels = NULL,                                                                # <1>
  image_size = c(image_size, image_size),
  crop_to_aspect_ratio = TRUE                                                   # <2>
)
dataset <- dataset |> dataset_rebatch(
  batch_size,
  drop_remainder = TRUE                                                         # <3>
)


# ----------------------------------------------------------------------
#| fig-cap: Example images from the Oxford Flowers dataset
par(mar = rep(.1, 4), mfrow = c(3, 6))
batch <- dataset |> as_array_iterator() |> iter_next()
for (i in 1:18) {
  img <- batch[i, , ,]
  plot(as.raster(img, max = 255))
}


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
#| lst-cap: The diffusion schedule
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


# ----------------------------------------------------------------------
#| fig-cap: Cosine relationship between noise rates and signal rates
noise_rates <- seq(0, 1, length.out = 100)
signal_rates <- sqrt(1 - noise_rates^2)

# Plot the relationship
par(pty = "s", bty = "l", mar = c( 5.1, 4.1, 4.1, 2.1))
plot(
  noise_rates,
  signal_rates,
  type = "l",
  asp = 1,
  col = "blue",
  lwd = 2,
  bty = "n",
  pty = "s",
  main = "Cosine Relationship between Noise and Signal Rates",
  xlab = "Noise Rates",
  ylab = "Signal Rates",
  xlim = c(0, 1),
  ylim = c(0, 1),
  panel.first = grid()
)

legend(
  "topright",
  inset = c(-0.1, -0.06),
  legend = expression(noise_rates^2 + signal_rates^2 == 1),
  bty = "n",
  xpd = TRUE,
  text.col = "black"
)


# ----------------------------------------------------------------------
#| fig-cap: Our cosine diffusion schedule
diffusion_times <- op_arange(0, 1, 0.01)                                        # <1>
schedule <- diffusion_schedule(diffusion_times)                                 # <1>

diffusion_times <- as.array(diffusion_times)                                    # <2>
noise_rates <- as.array(schedule$noise_rates)                                   # <2>
signal_rates <- as.array(schedule$signal_rates)                                 # <2>

plot(NULL, type = "n", main = "Diffusion Schedule",                             # <3>
     ylab = "Rate",  ylim = c(0, 1),
     xlab = "Diffusion time", xlim = c(0, 1))
lines(diffusion_times, noise_rates, col = "blue", lty = 1)                      # <4>
lines(diffusion_times, signal_rates, col = "red", lty = 2)                      # <4>

legend("bottomleft",                                                            # <5>
       legend = c("Noise rate", "Signal rate"),                                 # <5>
       col = c("blue", "red"), lty = c(1, 2))


# ----------------------------------------------------------------------
new_diffusion_model <- new_model_class(
  classname = "DiffusionModel",

  initialize = function(image_size, widths, block_depth, ...) {
    super$initialize(...)
    self$image_size <- shape(image_size)
    self$denoising_model <- get_model(image_size, widths, block_depth)
    self$seed_generator <- random_seed_generator()
    self$loss <- loss_mean_absolute_error()                                     # <1>
    self$normalizer <- layer_normalization()                                    # <2>
  },

# ----------------------------------------------------------------------

  denoise = function(noisy_images, noise_rates, signal_rates) {
    pred_noise_masks <-
      self$denoising_model(list(noisy_images, noise_rates))                     # <1>
    pred_images <-
      (noisy_images - noise_rates * pred_noise_masks) /
      signal_rates                                                              # <2>
    list(pred_images = pred_images, pred_noise_masks = pred_noise_masks)
  },

# ----------------------------------------------------------------------

  call = function(images) {
    images <- self$normalizer(images)
    .[batch_size, ..] <- op_shape(images)

    noise_masks <- random_normal(                                               # <1>
      shape = c(batch_size, self$image_size, self$image_size, 3),
      seed = self$seed_generator
    )

    diffusion_times <- random_uniform(                                          # <2>
      shape = c(batch_size, 1, 1, 1),
      minval = 0.0, maxval = 1.0,
      seed = self$seed_generator
    )

    .[noise_rates, signal_rates] <- diffusion_schedule(diffusion_times)
    noisy_images <- signal_rates * images + noise_rates * noise_masks           # <3>

    .[pred_images, pred_noise_masks] <-
      self$denoise(noisy_images, noise_rates, signal_rates)                     # <4>

    list(pred_images, pred_noise_masks, noise_masks)
  },

  compute_loss = function(x, y, y_pred,
                          sample_weight = NULL,
                          training = TRUE) {
    .[.., pred_noise_masks, noise_masks] <- y_pred
    self$loss(noise_masks, pred_noise_masks)
  },

# ----------------------------------------------------------------------

  generate = function(num_images, diffusion_steps) {
    noisy_images <- random_normal(                                              # <1>
      shape = c(num_images, self$image_size, self$image_size, 3),
      seed = self$seed_generator
    )

    diffusion_times <- seq(1, 0, length.out = diffusion_steps)

    for (i in seq_len(diffusion_steps - 1)) {
      diffusion_time <- diffusion_times[i]
      next_diffusion_time <- diffusion_times[i + 1]

      .[noise_rates, signal_rates] <- diffusion_time |>
        op_broadcast_to(c(num_images, 1, 1, 1)) |>
        diffusion_schedule()                                                    # <2>

      .[pred_images, pred_noises] <-                                            # <3>
        self$denoise(noisy_images, noise_rates, signal_rates)                   # <3>

      .[next_noise_rates, next_signal_rates] <- next_diffusion_time |>
        op_broadcast_to(c(num_images, 1, 1, 1)) |>
        diffusion_schedule()                                                    # <4>

      noisy_images <-                                                           # <4>
        (next_signal_rates * pred_images) +
        (next_noise_rates * pred_noises)
    }

    images <-
      self$normalizer$mean + pred_images * op_sqrt(self$normalizer$variance)    # <5>

    op_clip(images, 0, 255)                                                     # <5>
  }
)


# ----------------------------------------------------------------------
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
model <- new_diffusion_model(
  image_size,
  widths = c(32, 64, 96, 128),
  block_depth = 2
)
model$normalizer$adapt(dataset)                                                 # <1>


# ----------------------------------------------------------------------
model |> compile(
  optimizer = optimizer_adam_w(                                                 # <1>
    learning_rate = learning_rate_schedule_inverse_time_decay(                  # <1>
      initial_learning_rate = 1e-3,                                             # <1>
      decay_steps = 1000,                                                       # <1>
      decay_rate = 0.1                                                          # <1>
    ),                                                                          # <1>
    use_ema = TRUE,                                                             # <2>
    ema_overwrite_frequency = 100                                               # <3>
  )
)


# ----------------------------------------------------------------------
model |> fit(
  dataset,
  epochs = 100,
  callbacks = list(
    callback_visualization(),
    callback_model_checkpoint(
      filepath = "diffusion_model.weights.h5",
      save_weights_only = TRUE,
      save_best_only = TRUE,
      monitor = "loss"
    )
  )
)


