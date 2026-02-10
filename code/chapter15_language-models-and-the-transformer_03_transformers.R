# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "fs", "readr", "stringr", "tfdatasets")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
Sys.setenv("KERAS_BACKEND"="jax")
library(tfdatasets, exclude = c("shape"))
library(stringr)
library(keras3)
reticulate::py_require("keras-hub")
library(tensorflow, exclude = c("set_random_seed", "shape"))
library(tfdatasets, exclude = "shape")


# ----------------------------------------------------------------------
embedding_dim <- 256L
hidden_dim <- 1024L
max_length <- 250
punctuation_regex <- r"---([!"#$%&'()*+,./:;<=>?@\\^_`{|}~¡¿-])---"
vocab_size <- 15000
sequence_length <- 20
batch_size <- 64
embed_dim <- 256
hidden_dim <- 1024


# ----------------------------------------------------------------------
split_input <- function(text, sequence_length) {
  starts <- seq.int(1, str_length(text), by = sequence_length)
  str_sub(text, cbind(starts, length = sequence_length))
}

custom_standardization <- function(input_string) {
  input_string |>
    tf$strings$lower() |>
    tf$strings$regex_replace(punctuation_regex, "")
}

format_pair <- function(pair) {
  eng <- pair$english |> english_tokenizer()
  spa <- pair$spanish |> spanish_tokenizer()

  spa_feature <- spa@r[NA:-2]                                                   # <1>
  spa_target <- spa@r[2:NA]                                                     # <2>

  features <- list(english = eng, spanish = spa_feature)
  labels <- spa_target
  sample_weight <- labels != 0

  tuple(features, labels, sample_weight)
}

make_dataset <- function(pairs) {
  tensor_slices_dataset(pairs) |>
    dataset_map(format_pair, num_parallel_calls = 4) |>
    dataset_cache() |>
    dataset_shuffle(2048) |>
    dataset_batch(batch_size) |>
    dataset_prefetch(16)
}

generate_translation <- function(input_sentence) {

  tokenized_input_sentence <- english_tokenizer(list(input_sentence))
  decoded_sentence <- "[start]"

  for (i in seq_len(sequence_length)) {
    tokenized_target_sentence <- spanish_tokenizer(list(decoded_sentence))
    inputs <- list(english = tokenized_input_sentence,
                   spanish = tokenized_target_sentence)
    next_token_predictions <- predict(seq2seq_rnn, inputs, verbose = 0)
    sampled_token_index <- which.max(next_token_predictions[, i, ])
    sampled_token <- spa_vocab[sampled_token_index]
    decoded_sentence <- str_c(decoded_sentence, " ", sampled_token)
    if (sampled_token == "[end]")
      break
  }
  decoded_sentence
}


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
#| eval: false
# op_einsum("ij->ji")                                                             # <1>
# op_einsum("ij,jk->ik")                                                          # <2>
# op_einsum("hij,jk->hik")                                                        # <3>
# op_einsum("i,i->")                                                              # <4>
# op_einsum("ijk,ijk->ijk")                                                       # <5>
# op_einsum("ijk,ijk->")                                                          # <6>


# ----------------------------------------------------------------------
#| eval: false
# scores <- sources |>
#   sapply(\(source) score(target, source)) |>
#   softmax()
# combined <- sum(scores * sources)


# ----------------------------------------------------------------------
#| eval: false
# dot_product_attention <- function(target, source) {
#   scores <- op_einsum("btd,bsd->bts", target, source)                           # <1>
#   scores <- op_softmax(scores, axis = -1)
#   op_einsum("bts,bsd->btd", scores, source)                                     # <2>
# }
# 
# dot_product_attention(target, source)


# ----------------------------------------------------------------------
#| eval: false
# query_dense  <- layer_dense(, dim)
# key_dense    <- layer_dense(, dim)
# value_dense  <- layer_dense(, dim)
# output_dense <- layer_dense(, dim)
# 
# parameterized_attention <- function(query, key, value) {
#   query <- query |> query_dense()
#   key   <- key   |> key_dense()
#   value <- value |> value_dense()
# 
#   scores  <-
#     op_einsum("btd,bsd->bts", query, key) |>
#     op_softmax(axis = -1)
# 
#   outputs <- op_einsum("bts,bsd->btd", scores, value)
#   outputs |> output_dense()
# }
# 
# parameterized_attention(query = target, key = source, value = source)


# ----------------------------------------------------------------------
#| eval: false
# query_dense   <- replicate(num_heads, layer_dense(, head_dim))
# key_dense     <- replicate(num_heads, layer_dense(, head_dim))
# value_dense   <- replicate(num_heads, layer_dense(, head_dim))
# output_dense  <- layer_dense(, head_dim * num_heads)
# 
# multi_head_attention <- function(query, key, value) {
#   head_outputs <- lapply(seq_len(num_heads), function(i) {
#     query <- query |> query_dense[[i]]()
#     key   <- key   |> key_dense[[i]]()
#     value <- value |> value_dense[[i]]()
# 
#     scores <- op_einsum("btd,bsd->bts", query, key)
#     scores <- op_softmax(scores / op_sqrt(head_dim), axis = -1)
#     op_einsum("bts,bsd->btd", scores, value)
#   })
# 
#   head_outputs |> op_concatenate(axis = -1) |> output_dense()
# }
# 
# multi_head_attention(query = target, key = source, value = source)


# ----------------------------------------------------------------------
#| eval: false
# multi_head_attention <- layer_multi_head_attention(
#   num_heads = num_heads,
#   head_dim = head_dim
# )
# multi_head_attention(query = target, key = source, value = source)


# ----------------------------------------------------------------------
#| lst-cap: A Transformer encoder block
layer_transformer_encoder <- new_layer_class(
  "TransformerEncoder",
  initialize = function(hidden_dim, intermediate_dim, num_heads) {
    super$initialize()
    key_dim <- hidden_dim %/% num_heads
    self$self_attention <- layer_multi_head_attention(                          # <1>
      num_heads = num_heads,
      key_dim = key_dim
    )
    self$self_attention_layernorm <- layer_layer_normalization()                # <1>
    self$feed_forward_1 <- layer_dense(, intermediate_dim,                      # <2>
                                       activation = "relu")                     # <2>
    self$feed_forward_2 <- layer_dense(, hidden_dim)                            # <2>
    self$feed_forward_layernorm <- layer_layer_normalization()                  # <2>
  },
  call = function(source, source_mask) {
    residual <- x <- source                                                     # <3>
    mask <- source_mask@r[, newaxis, ]                                          # <3>
    x <- self$self_attention(                                                   # <3>
      query = x,                                                                # <3>
      key = x,                                                                  # <3>
      value = x,                                                                # <3>
      attention_mask = mask                                                     # <3>
    )                                                                           # <3>
    x <- x + residual                                                           # <3>
    x <- x |> self$self_attention_layernorm()                                   # <3>

    residual <- x                                                               # <4>
    x <- x |>                                                                   # <4>
      self$feed_forward_1() |>                                                  # <4>
      self$feed_forward_2()                                                     # <4>
    x <- x + residual                                                           # <4>
    x <- x |> self$feed_forward_layernorm()                                     # <4>
    x
  }
)


# ----------------------------------------------------------------------
layer_normalization <- function(batch_of_sequences) {
  mean <- op_mean(batch_of_sequences, axis = -1, keepdims = TRUE)               # <1>
  variance <- op_var(batch_of_sequences, axis = -1, keepdims = TRUE)            # <2>
  (batch_of_sequences - mean) / op_sqrt(variance)                               # <2>
}


# ----------------------------------------------------------------------
batch_normalization <- function(batch_of_images) {
  mean <- op_mean(batch_of_images, axis = c(1, 2, 3), keepdims = TRUE)          # <1>
  variance <- op_var(batch_of_images, axis = c(1, 2, 3), keepdims = TRUE)       # <2>
  (batch_of_images - mean) / op_sqrt(variance)                                  # <2>
}


# ----------------------------------------------------------------------
#| lst-cap: A Transformer decoder block
layer_transformer_decoder <- new_layer_class(
  "TransformerDecoder",
  initialize = function(hidden_dim, intermediate_dim, num_heads) {
    super$initialize()
    key_dim <- hidden_dim %/% num_heads
    self$self_attention <- layer_multi_head_attention(
      num_heads = num_heads,
      key_dim = key_dim
    )                                                                           # <1>
    self$cross_attention <- layer_multi_head_attention(
      num_heads = num_heads,
      key_dim = key_dim
    )                                                                           # <2>
    self$self_attention_layernorm <- layer_layer_normalization()                # <1>
    self$cross_attention_layernorm <- layer_layer_normalization()               # <2>
    self$feed_forward_1 <- layer_dense(, intermediate_dim,                      # <3>
                                       activation = "relu")                     # <3>
    self$feed_forward_2 <- layer_dense(, hidden_dim)                            # <3>
    self$feed_forward_layernorm <- layer_layer_normalization()                  # <3>
  },
  call = function(target, source, source_mask) {
    residual <- x <- target                                                     # <4>
    x <- self$self_attention(query = x, key = x, value = x,
                             use_causal_mask = TRUE)
    x <- x + residual                                                           # <4>
    x <- x |> self$self_attention_layernorm()                                   # <4>

    residual <- x                                                               # <5>
    mask <- source_mask@r[, newaxis, ]                                          # <5>
    x <- self$cross_attention(
      query = x, key = source, value = source,
      attention_mask = mask
    )
    x <- x + residual                                                           # <5>
    x <- x |> self$cross_attention_layernorm()                                  # <5>

    residual <- x                                                               # <6>
    x <- x |>
      self$feed_forward_1() |>
      self$feed_forward_2()                                                     # <6>
    x <- x + residual                                                           # <6>
    x <- x |> self$feed_forward_layernorm()                                     # <6>

    x
  }
)


# ----------------------------------------------------------------------
mat <- matrix(0, 5, 5)
mat[lower.tri(mat, diag = TRUE)] <- 1
write.table(mat, row.names = rep(" ", 5),
            col.names = FALSE, quote = FALSE, sep = " ")


# ----------------------------------------------------------------------
#| lst-cap: Building a Transformer model
hidden_dim <- 256
intermediate_dim <- 2048
num_heads <- 8

encoder <- layer_transformer_encoder(
  hidden_dim = hidden_dim,
  intermediate_dim = intermediate_dim,
  num_heads = num_heads
)

decoder <- layer_transformer_decoder(
  hidden_dim = hidden_dim,
  intermediate_dim = intermediate_dim,
  num_heads = num_heads
)

source <- keras_input(shape = NA, dtype = "int32", name = "english")
encoder_output <- source |>
  layer_embedding(input_dim = vocab_size, output_dim = hidden_dim) |>
  encoder(source_mask = source != 0L)

target <- keras_input(shape = list(NULL), dtype = "int32", name = "spanish")
target_predictions <- target |>
  layer_embedding(input_dim = vocab_size, output_dim = hidden_dim) |>
  decoder(source = encoder_output, source_mask = source != 0L) |>
  layer_dropout(0.5) |>
  layer_dense(units = vocab_size, activation = "softmax")

transformer <- keras_model(
  inputs = list(source, target),
  outputs = target_predictions
)


# ----------------------------------------------------------------------
transformer


# ----------------------------------------------------------------------
transformer |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  weighted_metrics = "accuracy"
)

transformer |> fit(train_ds, epochs = 15, validation_data = val_ds)


# ----------------------------------------------------------------------
#| lst-cap: A learned position embedding layer
layer_positional_embedding <- new_layer_class(
  "PositionalEmbedding",
  initialize = function(sequence_length, input_dim, output_dim) {
    super$initialize()
    self$token_embeddings <- layer_embedding(input_dim = input_dim,
                                             output_dim = output_dim)
    self$position_embeddings <- layer_embedding(input_dim = sequence_length,
                                                output_dim = output_dim)
  },
  call = function(inputs) {
    .[.., sequence_length] <- op_shape(inputs)
    positions <-
      op_arange(0, sequence_length - 1, dtype = "int32") |>                     # <1>
      op_expand_dims(1)
    embedded_tokens <- self$token_embeddings(inputs)
    embedded_positions <- self$position_embeddings(positions)
    embedded_tokens + embedded_positions
  }
)


# ----------------------------------------------------------------------
#| lst-cap: Building a Transformer model with position embeddings
hidden_dim <- 256
intermediate_dim <- 2056
num_heads <- 8

encoder <- layer_transformer_encoder(
  hidden_dim = hidden_dim,
  intermediate_dim = intermediate_dim,
  num_heads = num_heads
)

decoder <- layer_transformer_decoder(
  hidden_dim = hidden_dim,
  intermediate_dim = intermediate_dim,
  num_heads = num_heads
)

source <- keras_input(shape = NA, dtype = "int32", name = "english")

encoder_output <- source |>
  layer_positional_embedding(sequence_length, vocab_size, hidden_dim) |>
  encoder(source_mask = source != 0L)

target <- keras_input(shape = list(NULL), dtype = "int32", name = "spanish")

target_predictions <- target |>
  layer_positional_embedding(sequence_length, vocab_size, hidden_dim) |>
  decoder(source = encoder_output, source_mask = source != 0L) |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = vocab_size, activation = "softmax")

transformer <- keras_model(
  inputs = list(source, target),
  outputs = target_predictions
)


# ----------------------------------------------------------------------
transformer |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  weighted_metrics = "accuracy"
)
transformer |> fit(train_ds, epochs = 30, validation_data = val_ds)


# ----------------------------------------------------------------------
#| lst-cap: Generating translations with a Transformer
spa_vocab <- get_vocabulary(spanish_tokenizer)

generate_translation <- function(input_sentence) {

  tokenized_input_sentence <- english_tokenizer(list(input_sentence))
  decoded_sentence <- "[start]"

  for (i in seq_len(sequence_length)) {
    tokenized_target_sentence <- spanish_tokenizer(list(decoded_sentence))
    tokenized_target_sentence <- tokenized_target_sentence@r[, NA:-2]           # <1>

    inputs <- list(english = tokenized_input_sentence,
                   spanish = tokenized_target_sentence)
    next_token_predictions <- predict(transformer, inputs, verbose = 0)

    sampled_token_index <- which.max(next_token_predictions[1, i, ])
    sampled_token <- spa_vocab[sampled_token_index]

    decoded_sentence <- paste(decoded_sentence, sampled_token)

    if (sampled_token == "[end]")
      break
  }

  decoded_sentence
}


# ----------------------------------------------------------------------
for (i in sample.int(nrow(test_pairs), 5)) {
  input_sentence <- test_pairs$english[i]
  writeLines(c(
    "-",
    input_sentence,
    generate_translation(input_sentence)
  ))
}


# ----------------------------------------------------------------------
#| eval: false
# for (i in sample.int(nrow(test_pairs), 5)) {
#   .[english, spanish] <- test_pairs[i, ]
#   input_sentence = english
#   translated <- generate_translation(english)
#   cat("-- example", i, "--\n",
#       "  english:", english, "\n",
#       "  spanish:", spanish, "\n",
#       "predicted:", translated, "\n")
# }


# ----------------------------------------------------------------------
#| lst-cap: Loading the RoBERTa pretrained model with KerasHub
py_require("keras-hub")
keras_hub <- import("keras_hub")

tokenizer <- keras_hub$models$Tokenizer$from_preset("roberta_base_en")
backbone <- keras_hub$models$Backbone$from_preset("roberta_base_en")


# ----------------------------------------------------------------------
tokenizer("The quick brown fox")


# ----------------------------------------------------------------------
backbone


# ----------------------------------------------------------------------
batch_size <- 8
train_ds <- text_dataset_from_directory(
  "aclImdb/train", batch_size = batch_size,
   class_names = c("neg", "pos")
)
val_ds <- text_dataset_from_directory(
  "aclImdb/val", batch_size = batch_size
)
test_ds <- text_dataset_from_directory(
  "aclImdb/test", batch_size = batch_size
)


# ----------------------------------------------------------------------
#| eval: false
# list(
#   c("<s>", "the", "quick", "brown", "fox", "jumped", ".", "</s>"),
#   c("<s>", "the", "panda", "slept", ".", "</s>", "<pad>", "<pad>")
# )


# ----------------------------------------------------------------------
#| lst-cap: Preprocessing IMDb movie reviews with RoBERTa’s tokenizer
library(tfdatasets, exclude = "shape")

packer <- keras_hub$layers$StartEndPacker(
  sequence_length = 512L,
  start_value = tokenizer$start_token_id,
  end_value = tokenizer$end_token_id,
  pad_value = tokenizer$pad_token_id,
  return_padding_mask = TRUE,
)

preprocess <- function(text, label) {
  .[token_ids, padding_mask] <- text |> tokenizer() |> packer()
  list(
    named_list(token_ids, padding_mask),
    label
  )
}

preprocessed_train_ds <- train_ds |> dataset_map(preprocess)
preprocessed_val_ds   <- val_ds   |> dataset_map(preprocess)
preprocessed_test_ds  <- test_ds  |> dataset_map(preprocess)


# ----------------------------------------------------------------------
preprocessed_train_ds |> as_iterator() |> iter_next() |> str()


# ----------------------------------------------------------------------
#| lst-cap: Extending the base RoBERTa model for classification
inputs <- backbone$input
outputs <- inputs |>
  backbone() |>
  op_subset(, 1, ) |>                                                           # <1>
  layer_dropout(0.1) |>
  layer_dense(768, activation = "relu") |>
  layer_dropout(0.1) |>
  layer_dense(1, activation = "sigmoid")

classifier <- keras_model(inputs, outputs)


# ----------------------------------------------------------------------
#| lst-cap: Training the RoBERTa classification model
classifier |> compile(
  optimizer = optimizer_adam(5e-5),
  loss = "binary_crossentropy",
  metrics = "accuracy"
)
classifier |> fit(
  preprocessed_train_ds,
  validation_data = preprocessed_val_ds
)


# ----------------------------------------------------------------------
evaluate(classifier, preprocessed_test_ds)


