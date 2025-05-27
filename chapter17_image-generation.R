library(keras3)
reticulate::py_require("keras-hub==0.18.1")


latent_dim <- 2                                                                 # <1>

encoder_inputs <- keras_input(shape = c(28, 28, 1))
x <- encoder_inputs |>
  layer_conv_2d(32, 3, activation = "relu", strides = 2, padding = "same") |>
  layer_conv_2d(64, 3, activation = "relu", strides = 2, padding = "same") |>
  layer_flatten() |>
  layer_dense(16, activation = "relu")
z_mean    <- x |> layer_dense(latent_dim, name="z_mean")                        # <2>
z_log_var <- x |> layer_dense(latent_dim, name="z_log_var")                     # <2>
encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var),
                       name="encoder")


encoder


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


latent_inputs <- keras_input(shape = c(latent_dim))                             # <1>
decoder_outputs <- latent_inputs |>
  layer_dense(7 * 7 * 64, activation = "relu") |>                               # <2>
  layer_reshape(c(7, 7, 64)) |>                                                 # <3>
  layer_conv_2d_transpose(64, 3, activation = "relu",                           # <4>
                          strides = 2, padding = "same") |>                     # <4>
  layer_conv_2d_transpose(32, 3, activation = "relu",                           # <4>
                          strides = 2, padding = "same") |>                     # <4>
  layer_conv_2d(1, 3, activation = "sigmoid", padding = "same")                 # <5>
decoder <- keras_model(latent_inputs, decoder_outputs,
                       name = "decoder")


decoder


model_vae <- new_model_class(
  classname = "VAE",

  initialize = function(encoder, decoder, ...) {
    super$initialize(...)
    self$encoder <- encoder
    self$decoder <- decoder
    self$sampler <- layer_sampler()
    self$reconstruction_loss_tracker <-                                         # <1>
      metric_mean(name = "reconstruction_loss")                                 # <1>
    self$kl_loss_tracker <- metric_mean(name = "kl_loss")                       # <1>
  },

  call = function(inputs) {
    self$encoder(inputs)
  },

  compute_loss = function(x, y, y_pred, sample_weight = NULL, training = TRUE) {
    original <- x                                                               # <2>
    .[z_mean, z_log_var] <- y_pred                                              # <3>


    z <- self$sampler(z_mean, z_log_var)
    reconstruction <- self$decoder(z)                                           # <4>

    reconstruction_loss <-                                                      # <5>
      loss_binary_crossentropy(original, reconstruction) |>                     # <5>
      op_sum(axis = c(2, 3)) |>                                                 # <5>
      op_mean()                                                                 # <5>

    kl_loss <- -0.5 * (                                                         # <6>
      1 + z_log_var - op_square(z_mean) - op_exp(z_log_var)                     # <6>
    )                                                                           # <6>
    total_loss <- reconstruction_loss + op_mean(kl_loss)                        # <6>

    self$reconstruction_loss_tracker$update_state(reconstruction_loss)          # <7>
    self$kl_loss_tracker$update_state(kl_loss)                                  # <7>

    total_loss
  }
)


.[.[x_train, ..], .[x_test, ..]] <- dataset_mnist()
mnist_digits <- vctrs::vec_c(x_train, x_test)                                   # <1>
mnist_digits <- mnist_digits / 255
dim(mnist_digits) <- c(dim(mnist_digits), 1)

str(mnist_digits)

vae <- model_vae(encoder, decoder)
vae |> compile(optimizer = optimizer_adam())                                    # <2>
vae |> fit(mnist_digits, epochs = 30, batch_size = 128)                         # <3>


n <- 30                                                                         # <1>
digit_size <- 28

z <- seq(-1, 1, length.out = n)                                                 # <2>
z_grid <- as.matrix(expand.grid(z, z))                                          # <2>

decoded <- predict(vae$decoder, z_grid, verbose = 0)                            # <3>

z_grid_i <- expand.grid(x = seq_len(n), y = seq_len(n))                         # <4>
figure <- array(0, c(digit_size * n, digit_size * n))                           # <4>

for (i in 1:nrow(z_grid_i)) {
  .[xi, yi] <- z_grid_i[i, ]
  digit <- decoded[i, , , ]
  figure[seq(to = (n + 1 - xi) * digit_size, length.out = digit_size),
         seq(to = yi * digit_size, length.out = digit_size)] <-
    digit
}

par(pty = "s")                                                                  # <8>
lim <- extendrange(r = c(-1, 1), f = 1 - (n / (n+.5)))                          # <5>
plot(NULL, frame.plot = FALSE,
     ylim = lim, xlim = lim,
     xlab = ~z[1], ylab = ~z[2])                                                # <6>
rasterImage(as.raster(1 - figure, max = 1),                                     # <7>
            lim[1], lim[1], lim[2], lim[2],
            interpolate = FALSE)


fpath <- "flowers"


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


img <- dataset |> as_array_iterator() |> iter_next() |> _[1, , ,]
par(mar = c(0, 0, 0, 0))
plot(as.raster(img, max = 255))


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
  n <- noise_rates |> layer_upsampling_2d(size = image_size, interpolation = "nearest")
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

  diffusion_angles <- start_angle + diffusion_times * (end_angle - start_angle)
  signal_rates <- op_cos(diffusion_angles)
  noise_rates <- op_sin(diffusion_angles)

  list(noise_rates = noise_rates, signal_rates = signal_rates)
}


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

  denoise = function(noisy_images, noise_rates, signal_rates) {
    pred_noise_masks <- self$denoising_model(list(noisy_images, noise_rates))   # <3>
    pred_images <-
      (noisy_images - noise_rates * pred_noise_masks) /
      signal_rates                                                              # <4>
    list(pred_images = pred_images, pred_noise_masks = pred_noise_masks)
  },

  call = function(images) {
    images <- self$normalizer(images)
    .[batch_size, ..] <- op_shape(images)

    noise_masks <- random_normal(                                               # <5>
      shape = c(batch_size, self$image_size, self$image_size, 3),
      seed = self$seed_generator
    )

    diffusion_times <- random_uniform(                                          # <6>
      shape = c(batch_size, 1, 1, 1),
      minval = 0.0, maxval = 1.0,
      seed = self$seed_generator
    )

    .[noise_rates, signal_rates] <- diffusion_schedule(diffusion_times)
    noisy_images <- signal_rates * images + noise_rates * noise_masks           # <7>

    .[pred_images, pred_noise_masks] <-
      self$denoise(noisy_images, noise_rates, signal_rates)                     # <8>

    list(pred_images, pred_noise_masks, noise_masks)
  },

  compute_loss = function(x, y, y_pred, sample_weight = NULL, training = TRUE) {
    .[.., pred_noise_masks, noise_masks] <- y_pred
    self$loss(noise_masks, pred_noise_masks)
  },

  generate = function(num_images, diffusion_steps) {
    noisy_images <- random_normal(                                              # <9>
      shape = c(num_images, self$image_size, self$image_size, 3),
      seed = self$seed_generator
    )

    diffusion_times <- seq(1, 0, length.out = diffusion_steps)

    for (i in seq_len(diffusion_steps - 1)) {
      diffusion_time <- diffusion_times[i]
      next_diffusion_time <- diffusion_times[i + 1]

      .[noise_rates, signal_rates] <- diffusion_time |>
        op_broadcast_to(c(num_images, 1, 1, 1)) |>
        diffusion_schedule()                                                    # <10>

      .[pred_images, pred_noises] <-
        self$denoise(noisy_images, noise_rates, signal_rates)                   # <11>

      .[next_noise_rates, next_signal_rates] <- next_diffusion_time |>
        op_broadcast_to(c(num_images, 1, 1, 1)) |>
        diffusion_schedule()                                                    # <12>

      noisy_images <-                                                           # <13>
        (next_signal_rates * pred_images) +
        (next_noise_rates * pred_noises)
    }

    images <-                                                                   # <14>
      self$normalizer$mean + pred_images *
      op_sqrt(self$normalizer$variance)

    op_clip(images, 0, 255)                                                     # <15>
  }
)


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

    for (i in seq_len(self$num_rows * self$num_cols)) {
      plot(as.raster(generated_images[i, , , ], max = 255))
    }
  }
)


model <- new_diffusion_model(image_size, widths = c(32, 64, 96, 128),
                             block_depth = 2)
model$normalizer$adapt(dataset)                                                 # <1>


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


library(keras3)
reticulate::py_require("keras-hub==0.18.1")
keras_hub <- reticulate::import("keras_hub")

model <- keras_hub$models$TextToImage$from_preset(
  "stable_diffusion_3_medium",
  image_shape = shape(512, 512, 3),
  dtype = "float16"
)

image <- model$generate(
  "photograph of an astronaut riding a horse, detailed, 8k",
  guidance_scale = 1
)


par(mar = c(0, 0, 0, 0))
plot(as.raster(image, max = max(image)))


prompts <- c(
  "A photograph of a cat wearing a top hat, photorealistic",
  "A neon sci-fi skyline at night, illustration"
)

images <- model$generate(prompts, num_steps = 25L, guidance_scale = 7.5)

for(i in seq_len(nrow(images))) {
  image <- images[i, , , ]
  plot(as.raster(image, max = 255L))
  image_array_save(image, sprintf("generated_image_%i.png", i), scale = FALSE)
}



