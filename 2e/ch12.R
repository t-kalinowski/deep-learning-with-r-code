## ----setup----------------------------------------------------------------
library(keras3)
library(tensorflow)
library(tfautograph)
tensorflow::as_tensor(1)


## -------------------------------------------------------------------------
reweight_distribution <-
  function(original_distribution, temperature = 0.5) {
    original_distribution %>%
      { exp(log(.) / temperature) } %>%
      { . / sum(.) }
  }


## ---- eval = FALSE--------------------------------------------------------
## url <- "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
## filename <- basename(url)
## options(timeout = 60*10) # 10 minute timeout
## download.file(url, destfile = filename)
## untar(filename)


## -------------------------------------------------------------------------
library(tensorflow)
library(tfdatasets, exclude = "shape")
library(keras3)
dataset <- text_dataset_from_directory(directory = "aclImdb",
                                       label_mode = NULL,
                                       batch_size = 256)
dataset <- dataset %>%
  dataset_map( ~ tf$strings$regex_replace(.x, "<br />", " "))


## -------------------------------------------------------------------------
sequence_length <- 100
vocab_size <- 15000
text_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length
)
adapt(text_vectorization, dataset)


## -------------------------------------------------------------------------
prepare_lm_dataset <- function(text_batch) {
  vectorized_sequences <- text_vectorization(text_batch)
  x <- vectorized_sequences[, NA:-2]
  y <- vectorized_sequences[, 2:NA]
  list(x, y)
}

lm_dataset <- dataset %>%
  dataset_map(prepare_lm_dataset, num_parallel_calls = 4)


## ---- include=FALSE-------------------------------------------------------
layer_transformer_decoder <- new_layer_class(
  classname = "TransformerDecoder",

  initialize = function(embed_dim, dense_dim, num_heads, ...) {
    super$initialize(...)
    self$embed_dim <- embed_dim
    self$dense_dim <- dense_dim
    self$num_heads <- num_heads
    self$attention_1 <- layer_multi_head_attention(num_heads = num_heads,
                                                   key_dim = embed_dim)
    self$attention_2 <- layer_multi_head_attention(num_heads = num_heads,
                                                   key_dim = embed_dim)
    self$dense_proj <- keras_model_sequential() %>%
      layer_dense(dense_dim, activation = "relu") %>%
      layer_dense(embed_dim)

    self$layernorm_1 <- layer_layer_normalization()
    self$layernorm_2 <- layer_layer_normalization()
    self$layernorm_3 <- layer_layer_normalization()
    self$supports_masking <- TRUE
  },

  get_config = function() {
    config <- super$get_config()
    for (name in c("embed_dim", "num_heads", "dense_dim"))
      config[[name]] <- self[[name]]
    config
  },

  get_causal_attention_mask = function(inputs) {
    c(batch_size, sequence_length, encoding_length) %<-%
      tf$unstack(tf$shape(inputs))

    x <- tf$range(sequence_length)
    i <- x[, tf$newaxis]
    j <- x[tf$newaxis, ]
    mask <- tf$cast(i >= j, "int32")

    tf$tile(mask[tf$newaxis, , ],
            tf$stack(c(batch_size, 1L, 1L)))
  },

  call = function(inputs, encoder_outputs, mask = NULL) {

    causal_mask <- self$get_causal_attention_mask(inputs)

    if (is.null(mask))
      mask <- causal_mask
    else
      mask %<>% { tf$minimum(tf$cast(.[, tf$newaxis, ], "int32"),
                             causal_mask) }

    inputs %>%
      { self$attention_1(query = ., value = ., key = .,
                         attention_mask = causal_mask) + . } %>%
      self$layernorm_1() %>%

      { self$attention_2(query = .,
                         value = encoder_outputs,
                         key = encoder_outputs,
                         attention_mask = mask) + . } %>%
      self$layernorm_2() %>%

      { self$dense_proj(.) + . } %>%
      self$layernorm_3()

  }
)

layer_positional_embedding <- new_layer_class(
  classname = "PositionalEmbedding",

  initialize = function(sequence_length, input_dim, output_dim, ...) {
    super$initialize(...)
    self$token_embeddings <-
      layer_embedding(input_dim = input_dim,
                      output_dim = output_dim)
    self$position_embeddings <-
      layer_embedding(input_dim = sequence_length,
                      output_dim = output_dim)
    self$sequence_length <- sequence_length
    self$input_dim <- input_dim
    self$output_dim <- output_dim
  },

  call = function(inputs) {
    length <- tf$shape(inputs)[-1]
    positions <- tf$range(start = 0L, limit = length, delta = 1L)
    embedded_tokens <- self$token_embeddings(inputs)
    embedded_positions <- self$position_embeddings(positions)
    embedded_tokens + embedded_positions
  },

  compute_mask = function(inputs, mask = NULL) {
    inputs != 0
  },

  get_config = function() {
    config <- super$get_config()
    for(name in c("output_dim", "sequence_length", "input_dim"))
      config[[name]] <- self[[name]]
    config
  }
)


## -------------------------------------------------------------------------
embed_dim <- 256
latent_dim <- 2048
num_heads <- 2

transformer_decoder <-
  layer_transformer_decoder(NULL, embed_dim, latent_dim, num_heads)

inputs <- layer_input(shape(NA), dtype = "int64")
outputs <- inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  transformer_decoder(., .) %>%
  layer_dense(vocab_size, activation = "softmax")

model <-
  keras_model(inputs, outputs) %>%
  compile(loss = "sparse_categorical_crossentropy",
          optimizer = "rmsprop")


## -------------------------------------------------------------------------
vocab <- get_vocabulary(text_vectorization)

sample_next <- function(predictions, temperature = 1.0) {
  predictions %>%
    reweight_distribution(temperature) %>%
    sample.int(length(.), 1, prob = .)
}


generate_sentence <-
  function(model, prompt, generate_length, temperature) {

    sentence <- prompt
    for (i in seq(generate_length)) {

      model_preds <- sentence %>%
        array(dim = c(1, 1)) %>%
        text_vectorization() %>%
        predict(model, .)

      sampled_word <- model_preds %>%
        .[1, i, ] %>%
        sample_next(temperature) %>%
        vocab[.]

      sentence <- paste(sentence, sampled_word)

    }

    sentence
  }


## -------------------------------------------------------------------------
tf_sample_next <- function(predictions, temperature = 1.0) {
    predictions %>%
      reweight_distribution(temperature) %>%
      { log(.[tf$newaxis, ]) } %>%
      tf$random$categorical(1L) %>%
      tf$reshape(shape())
  }

library(tfautograph)

tf_generate_sentence <- tf_function(
  function(model, prompt, generate_length, temperature) {

    withr::local_options(tensorflow.extract.style = "python")

    vocab <- as_tensor(vocab)

    sentence <- prompt %>% as_tensor(shape = c(1, 1))

    ag_loop_vars(sentence)
    for (i in tf$range(generate_length)) {

      model_preds <- sentence %>%
        text_vectorization() %>%
        model()

      sampled_word <- model_preds %>%
        .[0, i, ] %>%
        tf_sample_next(temperature) %>%
        vocab[.]

      sentence <- sampled_word %>%
        { tf$strings$join(c(sentence, .), " ") }

    }

    sentence %>% tf$reshape(shape())
  }
)


## -------------------------------------------------------------------------
library(tfautograph)
autograph({
  for(i in tf$range(3L))
    print(i)
})


## -------------------------------------------------------------------------
fn <- function(x) {
  for(i in x) print(i)
}
ag_fn <- autograph(fn)
ag_fn(tf$range(3))


## -------------------------------------------------------------------------
callback_text_generator <- new_callback_class(
  classname = "TextGenerator",

  initialize = function(prompt, generate_length,
                        temperatures = 1,
                        print_freq = 1L) {
    private$prompt <- as_tensor(prompt, "string")
    private$generate_length <- as_tensor(generate_length, "int32")
    private$temperatures <- as.numeric(temperatures)
    private$print_freq <- as.integer(print_freq)
  },

  on_epoch_end = function(epoch, logs = NULL) {
    if ((epoch %% private$print_freq) != 0 )
      return()

    for (temperature in private$temperatures) {
      cat("== Generating with temperature", temperature, "\n")

      sentence <- tf_generate_sentence(
        self$model,
        as_tensor(private$prompt, "string"),
        as_tensor(private$generate_length, "int32"),
        as_tensor(temperature, "float32")
      )
      cat(as.character(sentence), "\n")
    }
  }
)


text_gen_callback <- callback_text_generator(
  prompt = "This movie",
  generate_length = 50,
  temperatures = c(0.2, 0.5, 0.7, 1., 1.5)
)


## -------------------------------------------------------------------------
model %>%
  fit(lm_dataset,
      epochs = 200,
      callbacks = list(text_gen_callback))


## ---- include = FALSE-----------------------------------------------------
model <- keras::load_model(
  "text_generator.keras",
  custom_objects = list(layer_transformer_decoder,
                        layer_positional_embedding)
)


## -------------------------------------------------------------------------
base_image_path <- get_file(
  "coast.jpg", origin = "https://img-datasets.s3.amazonaws.com/coast.jpg")

plot(as.raster(jpeg::readJPEG(base_image_path)))


## -------------------------------------------------------------------------
model <- application_inception_v3(weights = "imagenet", include_top = FALSE)


## -------------------------------------------------------------------------
layer_settings <- c(
  "mixed4" = 1.0,
  "mixed5" = 1.5,
  "mixed6" = 2.0,
  "mixed7" = 2.5
)

outputs <- list()
for(layer_name in names(layer_settings))
  outputs[[layer_name]] <- get_layer(model, layer_name)$output

feature_extractor <- keras_model(inputs = model$inputs,
                                 outputs = outputs)


## -------------------------------------------------------------------------
compute_loss <- function(input_image) {
  features <- feature_extractor(input_image)

  feature_losses <- names(features) %>%
    lapply(function(name) {
      coeff <- layer_settings[[name]]
      activation <- features[[name]]
      coeff * mean(activation[, 3:-3, 3:-3, ] ^ 2)
    })

  Reduce(`+`, feature_losses)
}


## -------------------------------------------------------------------------
gradient_ascent_step <- tf_function(
  function(image, learning_rate) {

    with(tf$GradientTape() %as% tape, {
      tape$watch(image)
      loss <- compute_loss(image)
    })

    grads <- tape$gradient(loss, image) %>%
      tf$math$l2_normalize()

    image %<>% `+`(learning_rate * grads)

    list(loss, image)
  })


gradient_ascent_loop <-
  function(image, iterations, learning_rate, max_loss = -Inf) {

    learning_rate %<>% as_tensor("float32")

    for(i in seq(iterations)) {

      c(loss, image) %<-% gradient_ascent_step(image, learning_rate)

      loss %<>% as.numeric()
      if(loss > max_loss)
        break

      writeLines(sprintf(
        "... Loss value at step %i: %.2f", i, loss))
    }

    image
  }


## -------------------------------------------------------------------------
step <- 20
num_octaves <- 3L
octave_scale <- 1.4
iterations <- 30
max_loss <- 15


## -------------------------------------------------------------------------
preprocess_image <- tf_function(function(image_path) {
  image_path %>%
    tf$io$read_file() %>%
    tf$io$decode_image() %>%
    tf$expand_dims(axis = 0L) %>%
    tf$cast("float32") %>%
    inception_v3_preprocess_input()
})

deprocess_image <- tf_function(function(img) {
  img %>%
    tf$squeeze(axis = 0L) %>%
    { (. * 127.5) + 127.5 } %>%
    tf$saturate_cast("uint8")
})


display_image_tensor <- function(x, ..., max = 255,
                                 plot_margins = c(0, 0, 0, 0)) {

  if (!is.null(plot_margins))
    withr::local_par(mar = plot_margins)

  x %>%
    as.array() %>%
    drop() %>%
    as.raster(max = max) %>%
    plot(..., interpolate = FALSE)
}


## ---- eval = FALSE--------------------------------------------------------
## display_image_tensor <- function()
##   <...>
##   opar <- par(mar = plot_margins)
##   on.exit(par(opar))
##   <...>
## }


## -------------------------------------------------------------------------
original_img <- preprocess_image(base_image_path)
original_HxW <- dim(original_img)[2:3]

calc_octave_HxW <- function(octave) {
  as.integer(round(original_HxW / (octave_scale ^ octave)))
}

octaves <- seq(num_octaves - 1, 0) %>%
  { zip_lists(num = .,
              HxW = lapply(., calc_octave_HxW)) }

str(octaves)

shrunk_original_img <- original_img %>% tf$image$resize(octaves[[1]]$HxW)

img <- original_img
for (octave in octaves) {
  cat(sprintf("Processing octave %i with shape (%s)\n",
              octave$num, paste(octave$HxW, collapse = ", ")))

  img <- img %>%
    tf$image$resize(octave$HxW) %>%
    gradient_ascent_loop(iterations = iterations,
                         learning_rate = step,
                         max_loss = max_loss)

  upscaled_shrunk_original_img <-
    shrunk_original_img %>%
    tf$image$resize(octave$HxW)

  same_size_original <-
    original_img %>%
    tf$image$resize(octave$HxW)

  lost_detail <-
    same_size_original - upscaled_shrunk_original_img

  img %<>% `+`(lost_detail)

  shrunk_original_img <- original_img %>% tf$image$resize(octave$HxW)
}

img <- deprocess_image(img)

img %>% display_image_tensor()

img %>%
  tf$io$encode_png() %>%
  tf$io$write_file("dream.png", .)


## ---- eval = FALSE--------------------------------------------------------
## loss <- distance(style(reference_image) - style(combination_image)) +
##         distance(content(original_image) - content(combination_image))


## -------------------------------------------------------------------------
base_image_path <- get_file(
  "sf.jpg",  origin = "https://img-datasets.s3.amazonaws.com/sf.jpg")

style_reference_image_path <- get_file(
  "starry_night.jpg",
  origin = "https://img-datasets.s3.amazonaws.com/starry_night.jpg")

c(original_height, original_width) %<-% {
  base_image_path %>%
    tf$io$read_file() %>%
    tf$io$decode_image() %>%
    dim() %>% .[1:2]
}
img_height <- 400
img_width <- round(img_height * (original_width /
                                   original_height))


## -------------------------------------------------------------------------
preprocess_image <- function(image_path) {
  image_path %>%
    tf$io$read_file() %>%
    tf$io$decode_image() %>%
    tf$image$resize(as.integer(c(img_height, img_width))) %>%
    k_expand_dims(axis = 1) %>%
    imagenet_preprocess_input()
}

deprocess_image <- tf_function(function(img) {
  if (length(dim(img)) == 4)
    img <- k_squeeze(img, axis = 1)

  c(b, g, r) %<-% {
    img %>%
      k_reshape(c(img_height, img_width, 3)) %>%
      k_unstack(axis = 3)
  }

  r %<>% `+`(123.68)
  g %<>% `+`(103.939)
  b %<>% `+`(116.779)

  k_stack(c(r, g, b), axis = 3) %>%
    k_clip(0, 255) %>%
    k_cast("uint8")
})


## -------------------------------------------------------------------------
model <- application_vgg19(weights = "imagenet", include_top = FALSE)

outputs <- list()
for (layer in model$layers)
  outputs[[layer$name]] <- layer$output

feature_extractor <- keras_model(inputs = model$inputs,
                                 outputs = outputs)


## -------------------------------------------------------------------------
content_loss <- function(base_img, combination_img)
    sum((combination_img - base_img)^2)


## -------------------------------------------------------------------------
gram_matrix <- function(x) {
  n_features <- tf$shape(x)[3]
  x %>%
    tf$reshape(c(-1L, n_features)) %>%
    tf$matmul(., ., transpose_a = TRUE)
}

style_loss <- function(style_img, combination_img) {
  S <- gram_matrix(style_img)
  C <- gram_matrix(combination_img)
  channels <- 3
  size <- img_height * img_width
  sum((S - C) ^ 2) /
    (4 * (channels ^ 2) * (size ^ 2))
}


## -------------------------------------------------------------------------
total_variation_loss <- function(x) {
  a <- k_square(x[, NA:(img_height-1), NA:(img_width-1), ] -
                x[, 2:NA             , NA:(img_width-1), ])
  b <- k_square(x[, NA:(img_height-1), NA:(img_width-1), ] -
                x[, NA:(img_height-1), 2:NA            , ])
  sum((a + b) ^ 1.25)
}


## -------------------------------------------------------------------------
style_layer_names <- c(
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
)
content_layer_name <- "block5_conv2"
total_variation_weight <- 1e-6
content_weight <- 2.5e-8
style_weight <- 1e-6

compute_loss <-
  function(combination_image, base_image, style_reference_image) {

    input_tensor <-
      list(base_image,
           style_reference_image,
           combination_image) %>%
      k_concatenate(axis = 1)

    features <- feature_extractor(input_tensor)
    layer_features <- features[[content_layer_name]]
    base_image_features <- layer_features[1, , , ]
    combination_features <- layer_features[3, , , ]

    loss <- 0
    loss %<>% `+`(
      content_loss(base_image_features, combination_features) *
        content_weight
    )

    for (layer_name in style_layer_names) {
      layer_features <- features[[layer_name]]
      style_reference_features <- layer_features[2, , , ]
      combination_features <- layer_features[3, , , ]

      loss %<>% `+`(
        style_loss(style_reference_features, combination_features) *
          style_weight / length(style_layer_names)
      )
    }

    loss %<>% `+`(
      total_variation_loss(combination_image) *
        total_variation_weight
    )

    loss
  }


## -------------------------------------------------------------------------
compute_loss_and_grads <- tf_function(
  function(combination_image, base_image, style_reference_image) {
    with(tf$GradientTape() %as% tape, {
      loss <- compute_loss(combination_image,
                           base_image,
                           style_reference_image)
    })
    grads <- tape$gradient(loss, combination_image)
    list(loss, grads)
  })

optimizer <- optimizer_sgd(
  learning_rate_schedule_exponential_decay(
    initial_learning_rate = 100, decay_steps = 100, decay_rate = 0.96))



optimizer <-
  optimizer_sgd(learning_rate = learning_rate_schedule_exponential_decay(
    initial_learning_rate = 100,
    decay_steps = 100,
    decay_rate = 0.96
  ))


base_image <- preprocess_image(base_image_path)
style_reference_image <- preprocess_image(style_reference_image_path)
combination_image <- tf$Variable(preprocess_image(base_image_path))


## -------------------------------------------------------------------------
output_dir <- fs::path("style-transfer-generated-images")
iterations <- 4000
for (i in seq(iterations)) {
  c(loss, grads) %<-% compute_loss_and_grads(
    combination_image, base_image, style_reference_image)

  optimizer$apply_gradients(list(
    tuple(grads, combination_image)))

  if ((i %% 100) == 0) {
    cat(sprintf("Iteration %i: loss = %.2f\n", i, loss))
    img <- deprocess_image(combination_image)
    display_image_tensor(img)
    fname <- sprintf("combination_image_at_iteration_%04i.png", i)
    tf$io$write_file(filename = output_dir / fname,
                     contents = tf$io$encode_png(img))
  }
}


## ---- eval = FALSE--------------------------------------------------------
## c(z_mean, z_log_variance) %<-% encoder(input_img)
## z <- z_mean + exp(z_log_variance) * epsilon
## reconstructed_img <- decoder(z)
## model <- keras_model(input_img, reconstructed_img)


## -------------------------------------------------------------------------
latent_dim <- 2

encoder_inputs <-  layer_input(shape=c(28, 28, 1))
x <- encoder_inputs %>%
  layer_conv_2d(32, 3, activation = "relu", strides = 2, padding = "same") %>%
  layer_conv_2d(64, 3, activation = "relu", strides = 2, padding = "same") %>%
  layer_flatten() %>%
  layer_dense(16, activation = "relu")
z_mean    <- x %>% layer_dense(latent_dim, name="z_mean")
z_log_var <- x %>% layer_dense(latent_dim, name="z_log_var")
encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var),
                       name="encoder")


## -------------------------------------------------------------------------
encoder


## -------------------------------------------------------------------------
layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(self, z_mean, z_log_var) {
    epsilon <- tf$random$normal(shape = tf$shape(z_mean))
    z_mean + exp(0.5 * z_log_var) * epsilon
  }
)


## -------------------------------------------------------------------------
latent_inputs <- layer_input(shape = c(latent_dim))
decoder_outputs <- latent_inputs %>%
  layer_dense(7 * 7 * 64, activation = "relu") %>%
  layer_reshape(c(7, 7, 64)) %>%
  layer_conv_2d_transpose(64, 3, activation = "relu",
                          strides = 2, padding = "same") %>%
  layer_conv_2d_transpose(32, 3, activation = "relu",
                          strides = 2, padding = "same") %>%
  layer_conv_2d(1, 3, activation = "sigmoid", padding = "same")
decoder <- keras_model(latent_inputs, decoder_outputs,
                       name = "decoder")


## -------------------------------------------------------------------------
decoder


## -------------------------------------------------------------------------
model_vae <- new_model_class(
  classname = "VAE",

  initialize = function(encoder, decoder, ...) {
    super$initialize(...)
    self$encoder <- encoder
    self$decoder <- decoder
    self$sampler <- layer_sampler()
    self$total_loss_tracker <-
      metric_mean(name = "total_loss")
    self$reconstruction_loss_tracker <-
      metric_mean(name = "reconstruction_loss")
    self$kl_loss_tracker <-
      metric_mean(name = "kl_loss")
  },

  metrics = mark_active(function() {
    list(
      self$total_loss_tracker,
      self$reconstruction_loss_tracker,
      self$kl_loss_tracker
    )
  }),

  train_step = function(data) {
    with(tf$GradientTape() %as% tape, {

      c(z_mean, z_log_var) %<-% self$encoder(data)
      z <- self$sampler(z_mean, z_log_var)

      reconstruction <- decoder(z)
      reconstruction_loss <-
        loss_binary_crossentropy(data, reconstruction) %>%
          sum(axis = c(2, 3)) %>%
          mean()

      kl_loss <- -0.5 * (1 + z_log_var - z_mean^2 - exp(z_log_var))
      total_loss <- reconstruction_loss + mean(kl_loss)
    })

    grads <- tape$gradient(total_loss, self$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))

    self$total_loss_tracker$update_state(total_loss)
    self$reconstruction_loss_tracker$update_state(reconstruction_loss)
    self$kl_loss_tracker$update_state(kl_loss)

    list(total_loss = self$total_loss_tracker$result(),
         reconstruction_loss = self$reconstruction_loss_tracker$result(),
         kl_loss = self$kl_loss_tracker$result())
  }
)


## -------------------------------------------------------------------------
library(listarrays)
c(c(x_train, .), c(x_test, .)) %<-% dataset_mnist()

mnist_digits <-
  bind_on_rows(x_train, x_test) %>%
  expand_dims(-1) %>%
  { . / 255 }

str(mnist_digits)

vae <- model_vae(encoder, decoder)
vae %>% compile(optimizer = optimizer_adam())


## -------------------------------------------------------------------------
vae %>% fit(mnist_digits, epochs = 30, batch_size = 128)


## -------------------------------------------------------------------------
n <- 30
digit_size <- 28

z_grid <-
  seq(-1, 1, length.out = n) %>%
  expand.grid(., .) %>%
  as.matrix()

decoded <- predict(vae$decoder, z_grid)

z_grid_i <- seq(n) %>% expand.grid(x = ., y = .)
figure <- array(0, c(digit_size * n, digit_size * n))
for (i in 1:nrow(z_grid_i)) {
  c(xi, yi) %<-% z_grid_i[i, ]
  digit <- decoded[i, , , ]
  figure[seq(to = (n + 1 - xi) * digit_size, length.out = digit_size),
         seq(to = yi * digit_size, length.out = digit_size)] <-
    digit
}

par(pty = "s")
lim <- extendrange(r = c(-1, 1),
                   f = 1 - (n / (n+.5)))
plot(NULL, frame.plot = FALSE,
     ylim = lim, xlim = lim,
     xlab = ~z[1], ylab = ~z[2])
rasterImage(as.raster(1 - figure, max = 1),
            lim[1], lim[1], lim[2], lim[2],
            interpolate = FALSE)


## ---- eval = FALSE--------------------------------------------------------
## reticulate::py_install("gdown", pip = TRUE)
## system("gdown 1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684")


## ---- eval = FALSE--------------------------------------------------------
## zip::unzip("img_align_celeba.zip", exdir = "celeba_gan")


## -------------------------------------------------------------------------
dataset <- image_dataset_from_directory(
  "celeba_gan",
  label_mode = NULL,
  image_size = c(64, 64),
  batch_size = 32,
  crop_to_aspect_ratio = TRUE
)


## -------------------------------------------------------------------------
library(tfdatasets, exclude = "shape")
dataset %<>% dataset_map(~ .x / 255)


## -------------------------------------------------------------------------
x <- dataset %>% as_iterator() %>% iter_next()
display_image_tensor(x[1, , , ], max = 1)


## -------------------------------------------------------------------------
discriminator <-
  keras_model_sequential(name = "discriminator",
                         input_shape = c(64, 64, 3)) %>%
  layer_conv_2d(64, kernel_size = 4, strides = 2, padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d(128, kernel_size = 4, strides = 2, padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d(128, kernel_size = 4, strides = 2, padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_flatten() %>%
  layer_dropout(0.2) %>%
  layer_dense(1, activation = "sigmoid")


## -------------------------------------------------------------------------
discriminator


## -------------------------------------------------------------------------
latent_dim <- 128

generator <-
  keras_model_sequential(name = "generator",
                         input_shape = c(latent_dim)) %>%
  layer_dense(8 * 8 * 128) %>%
  layer_reshape(c(8, 8, 128)) %>%
  layer_conv_2d_transpose(128, kernel_size = 4, strides = 2,
                          padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d_transpose(256, kernel_size = 4, strides = 2,
                          padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d_transpose(512, kernel_size = 4, strides = 2,
                          padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d(3, kernel_size = 5, padding = "same",
                activation = "sigmoid")


## -------------------------------------------------------------------------
generator


## -------------------------------------------------------------------------
model_gan <- new_model_class(
  classname = "GAN",

  initialize = function(discriminator, generator, latent_dim) {
    super$initialize()
    self$discriminator  <- discriminator
    self$generator      <- generator
    self$latent_dim     <- as.integer(latent_dim)
    self$d_loss_metric  <- metric_mean(name = "d_loss")
    self$g_loss_metric  <- metric_mean(name = "g_loss")
  },

  compile = function(d_optimizer, g_optimizer, loss_fn) {
    super$compile()
    self$d_optimizer <- d_optimizer
    self$g_optimizer <- g_optimizer
    self$loss_fn <- loss_fn
  },

   metrics = mark_active(function() {
      list(self$d_loss_metric,
           self$g_loss_metric)
   }),

  train_step = function(real_images) {
    batch_size <- tf$shape(real_images)[1]
    random_latent_vectors <-
      tf$random$normal(shape = c(batch_size, self$latent_dim))
    generated_images <- self$generator(random_latent_vectors)

    combined_images <-
      tf$concat(list(generated_images,
                     real_images),
                axis = 0L)

    labels <-
      tf$concat(list(tf$ones(tuple(batch_size, 1L)),
                     tf$zeros(tuple(batch_size, 1L))),
                axis = 0L)

    labels %<>% `+`(
      tf$random$uniform(tf$shape(.), maxval = 0.05))

    with(tf$GradientTape() %as% tape, {
      predictions <- self$discriminator(combined_images)
      d_loss <- self$loss_fn(labels, predictions)
    })

    grads <- tape$gradient(d_loss, self$discriminator$trainable_weights)
    self$d_optimizer$apply_gradients(
      zip_lists(grads, self$discriminator$trainable_weights))

    random_latent_vectors <-
      tf$random$normal(shape = c(batch_size, self$latent_dim))

    misleading_labels <- tf$zeros(tuple(batch_size, 1L))

    with(tf$GradientTape() %as% tape, {
      predictions <- random_latent_vectors %>%
        self$generator() %>%
        self$discriminator()
      g_loss <- self$loss_fn(misleading_labels, predictions)
    })
    grads <- tape$gradient(g_loss, self$generator$trainable_weights)
    self$g_optimizer$apply_gradients(
      zip_lists(grads, self$generator$trainable_weights))

    self$d_loss_metric$update_state(d_loss)
    self$g_loss_metric$update_state(g_loss)

    list(d_loss = self$d_loss_metric$result(),
         g_loss = self$g_loss_metric$result())
  })


## -------------------------------------------------------------------------
callback_gan_monitor <- new_callback_class(
  classname = "GANMonitor",

  initialize = function(num_img = 3, latent_dim = 128,
                        dirpath = "gan_generated_images") {
    private$num_img <- as.integer(num_img)
    private$latent_dim <- as.integer(latent_dim)
    private$dirpath <- fs::path(dirpath)
    fs::dir_create(dirpath)
  },

  on_epoch_end = function(epoch, logs = NULL) {
    random_latent_vectors <-
      tf$random$normal(shape = c(private$num_img, private$latent_dim))

    generated_images <- random_latent_vectors %>%
      self$model$generator() %>%
      { tf$saturate_cast(. * 255, "uint8") }

    for (i in seq(private$num_img))
      tf$io$write_file(
        filename = private$dirpath / sprintf("img_%03i_%02i.png", epoch, i),
        contents = tf$io$encode_png(generated_images[i, , , ])
      )
  }
)


## -------------------------------------------------------------------------
epochs <- 100

gan <- model_gan(discriminator = discriminator,
                 generator = generator,
                 latent_dim = latent_dim)

gan %>% compile(
  d_optimizer = optimizer_adam(learning_rate = 0.0001),
  g_optimizer = optimizer_adam(learning_rate = 0.0001),
  loss_fn = loss_binary_crossentropy()
)


## -------------------------------------------------------------------------
gan %>% fit(
  dataset,
  epochs = epochs,
  callbacks = callback_gan_monitor(num_img = 10, latent_dim = latent_dim)
)
