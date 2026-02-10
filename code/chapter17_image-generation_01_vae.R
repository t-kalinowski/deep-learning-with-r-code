# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "fs", "jpeg", "tfdatasets", "vctrs")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
py_require("keras-hub")


# ----------------------------------------------------------------------
#| eval: false
# .[z_mean, z_log_var] <- encoder(input_img)                                      # <1>
# z <- z_mean + exp(0.5 * z_log_var) * epsilon                                    # <2>
# reconstructed_img <- decoder(z)                                                 # <3>
# model <- keras_model(input_img, reconstructed_img)                              # <4>


# ----------------------------------------------------------------------
#| lst-cap: VAE encoder network
latent_dim <- 2                                                                 # <1>

encoder_inputs <- keras_input(shape = c(28, 28, 1))
x <- encoder_inputs |>
  layer_conv_2d(32, 3, activation = "relu",
                strides = 2, padding = "same") |>
  layer_conv_2d(64, 3, activation = "relu",
                strides = 2, padding = "same") |>
  layer_flatten() |>
  layer_dense(16, activation = "relu")
z_mean    <- x |> layer_dense(latent_dim, name="z_mean")                        # <2>
z_log_var <- x |> layer_dense(latent_dim, name="z_log_var")                     # <2>
encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var),
                       name="encoder")


# ----------------------------------------------------------------------
encoder


# ----------------------------------------------------------------------
#| lst-cap: Latent-space-sampling layer
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
#| lst-cap: "VAE decoder network, mapping latent space points to images"
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


# ----------------------------------------------------------------------
decoder


# ----------------------------------------------------------------------
#| lst-cap: "VAE model with custom `compute_loss()` method"
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

  compute_loss = function(x, y, y_pred,
                          sample_weight = NULL,
                          training = TRUE) {
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


# ----------------------------------------------------------------------
#| lst-cap: Training the VAE
.[.[x_train, ..], .[x_test, ..]] <- dataset_mnist()
mnist_digits <- vctrs::vec_c(x_train, x_test)                                   # <1>
mnist_digits <- mnist_digits / 255
dim(mnist_digits) <- c(dim(mnist_digits), 1)
str(mnist_digits)

vae <- model_vae(encoder, decoder)
vae |> compile(optimizer = optimizer_adam())                                    # <2>


# ----------------------------------------------------------------------
#| results: false
vae |> fit(mnist_digits, epochs = 30, batch_size = 128)                         # <1>


# ----------------------------------------------------------------------
#| lst-cap: Sampling a grid of points from the 2D latent space and decoding them to images
#| fig-cap: Grid of digits decoded from the latent space
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


