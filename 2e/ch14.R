## ----setup, include=FALSE-------------------------------------------------
library(tensorflow)
library(keras3)
tensorflow::as_tensor(1)


## ---- include = FALSE-----------------------------------------------------
## Define objects so all the visible code is runnable
num_classes <- 10
num_values <- 4
num_inputs_features <- 20
num_timesteps <- 20
height <- width <- 200
channels <- 3
num_features <- 20
sequence_length <- 200
embed_dim <- 256
dense_dim <- 32
num_heads <- 4
vocab_size <- 15000

layer_transformer_encoder <- new_layer_class(
  classname = "TransformerEncoder",
  initialize = function(embed_dim, dense_dim, num_heads, ...) {
    super$initialize(...)
    self$embed_dim <- embed_dim
    self$dense_dim <- dense_dim
    self$num_heads <- num_heads
    self$attention <-
      layer_multi_head_attention(num_heads = num_heads,
                                 key_dim = embed_dim)

    self$dense_proj <- keras_model_sequential() %>%
      layer_dense(dense_dim, activation = "relu") %>%
      layer_dense(embed_dim)

    self$layernorm_1 <- layer_layer_normalization()
    self$layernorm_2 <- layer_layer_normalization()
  },

  call = function(inputs, mask = NULL) {
    if (!is.null(mask))
      mask <- mask[, tf$newaxis, ]

    inputs %>%
      { self$attention(., ., attention_mask = mask) + . } %>%
      self$layernorm_1() %>%
      { self$dense_proj(.) + . } %>%
      self$layernorm_2()
  },

  get_config = function() {
    config <- super$get_config()
    for(name in c("embed_dim", "num_heads", "dense_dim"))
      config[[name]] <- self[[name]]
    config
  }
)

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
inputs <- layer_input(shape = c(num_inputs_features))
outputs <- inputs %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "binary_crossentropy")


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(num_inputs_features))
outputs <- inputs %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(num_classes, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy")


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(num_inputs_features))
outputs <- inputs %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(num_classes, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "binary_crossentropy")


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(num_inputs_features))
outputs <- inputs %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(num_values)
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "mse")


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(height, width, channels))
outputs <- inputs %>%
  layer_separable_conv_2d(32, 3, activation = "relu") %>%
  layer_separable_conv_2d(64, 3, activation = "relu") %>%
  layer_max_pooling_2d(2) %>%
  layer_separable_conv_2d(64, 3, activation = "relu") %>%
  layer_separable_conv_2d(128, 3, activation = "relu") %>%
  layer_max_pooling_2d(2) %>%
  layer_separable_conv_2d(64, 3, activation = "relu") %>%
  layer_separable_conv_2d(128, 3, activation = "relu") %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(num_classes, activation = "softmax")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy")


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(num_timesteps, num_features))
outputs <- inputs %>%
  layer_lstm(32) %>%
  layer_dense(num_classes, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "binary_crossentropy")


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(num_timesteps, num_features))
outputs <- inputs %>%
  layer_lstm(32, return_sequences = TRUE) %>%
  layer_lstm(32, return_sequences = TRUE) %>%
  layer_lstm(32) %>%
  layer_dense(num_classes, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "binary_crossentropy")


## -------------------------------------------------------------------------
encoder_inputs <- layer_input(shape = c(sequence_length), dtype = "int64")
encoder_outputs <- encoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  layer_transformer_encoder(embed_dim, dense_dim, num_heads)

decoder <- layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)
decoder_inputs <- layer_input(shape = c(NA), dtype = "int64")
decoder_outputs <- decoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  decoder(., encoder_outputs) %>%
  layer_dense(vocab_size, activation = "softmax")

transformer <- keras_model(list(encoder_inputs, decoder_inputs),
                           decoder_outputs)
transformer %>%
  compile(optimizer = "rmsprop", loss = "categorical_crossentropy")


## -------------------------------------------------------------------------
inputs <- layer_input(shape=c(sequence_length), dtype="int64")
outputs <- inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  layer_transformer_encoder(embed_dim, dense_dim, num_heads) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer="rmsprop", loss="binary_crossentropy")



## ---- eval = FALSE--------------------------------------------------------
## # Instance
## l <- as.list(obj)
## #----
## l_sum <- 0
## l_entries <- 0
## for (e in l) {
##   if (!is.null(e)) {
##     l_sum <- l_sum + e
##     l_entries <- l_entries + 1
##   }
## }
## avg <- l_sum / l_entries
## #-----
## print('avg:', avg)
##
##
## # Instance
## my_list <- get_data()
## ----
## total <- 0
## num_elems <- 0
## for (n in my_list) {
##   if (!is.null(n)) {
##     total <- total + e
##     num_elems <- num_elems + 1
##   }
## }
## mean <- total / num_elems
## ----
## update_mean(mean)
##
##
## # Shared abstraction
## compute_mean <- function(x) {
##   num_elems <- total <- 0
##   for (e in x)
##     if (!is.null(e)) {
##       total <- total + e
##       num_elems <- num_elems + 1
##     }
##   total / num_elems
## }
