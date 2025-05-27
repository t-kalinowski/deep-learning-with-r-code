library(tfdatasets, exclude = c("shape"))
library(stringr)
library(keras3)
reticulate::py_require("keras-hub==0.18.1")
reticulate::import("keras")
reticulate::import("keras_hub")


library(keras3)

filename = get_file(
  origin = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)
shakespeare <- readLines(filename)


writeLines(head(shakespeare, 14))


sequence_length <- 100                                                          # <1>

split_input <- function(text, sequence_length) {
  starts <- seq.int(1, str_length(text), by = sequence_length)
  str_sub(text, cbind(starts, length = sequence_length))
}

shakespeare <- shakespeare |> str_flatten("\n")
features <- shakespeare |> str_sub(end = -2)  |> split_input(sequence_length)
labels   <- shakespeare |> str_sub(start = 2) |> split_input(sequence_length)

dataset <- tensor_slices_dataset(tuple(features, labels))


dataset |>
  as_iterator() |> iter_next() |>
  lapply(tf$strings$substr, 0L, len = 20L)


tokenizer <- layer_text_vectorization(
  standardize = NULL,
  split = "character",
  output_sequence_length = sequence_length
)
features_only_dataset <- dataset |> dataset_map(\(text, labels) text)
adapt(tokenizer, features_only_dataset)


vocabulary_size <- tokenizer$vocabulary_size()
vocabulary_size


dataset <- dataset |>
  dataset_map(\(features, labels) {
    tuple(tokenizer(features), tokenizer(labels))
  }, num_parallel_calls = 8)

training_data <-  dataset |>
  dataset_cache() |>
  dataset_shuffle(10000) |>
  dataset_batch(64)


embedding_dim <- 256L
hidden_dim <- 1024L

inputs <- keras_input(shape = c(sequence_length), dtype = "int",
                      name = "token_ids")

outputs <- inputs |>
  layer_embedding(vocabulary_size, embedding_dim) |>
  layer_gru(hidden_dim, return_sequences = TRUE) |>
  layer_dropout(0.1) |>
  layer_dense(vocabulary_size, activation = "softmax")                          # <1>

model <- keras_model(inputs, outputs)


model


model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "sparse_categorical_accuracy"
)
model |> fit(training_data, epochs = 20)


inputs <- keras_input(shape = c(1), dtype = "int", name = "token_ids")          # <1>
input_state <- keras_input(shape = c(hidden_dim), name = "state")

x <- inputs |> layer_embedding(vocabulary_size, embedding_dim)
.[x, output_state] <- layer_gru(units = hidden_dim, return_state = TRUE)(
  x, initial_state = input_state
)
outputs <- x |> layer_dense(vocabulary_size, activation="softmax")
generation_model <- keras_model(
  inputs = list(inputs, input_state),
  outputs = list(outputs, output_state)
)
set_weights(generation_model, get_weights(model))                               # <2>


vocab <- get_vocabulary(tokenizer)
token_ids <- seq_along(vocabulary_size)

chars_to_ids = function(chars) match(chars, vocab, nomatch = 2L) - 1L
ids_to_chars = function(ids) vocab[ids + 1L]

prompt = r"--(
KING RICHARD III:
)--"


input_ids <- chars_to_ids(str_split_1(prompt, ""))
state <- op_zeros(shape = c(1, hidden_dim))
for (token_id in input_ids) {
  inputs <- op_expand_dims(token_id, axis = 1)
  .[predictions, state] <- generation_model(tuple(inputs, state))               # <1>
}


max_length <- 250
generated_ids <- integer(max_length)

for (i in seq_len(max_length)) {                                                # <1>
  next_char_id <- op_argmax(predictions, axis = -1,                             # <2>
                            zero_indexed = TRUE, keepdims = TRUE)
  generated_ids[i] <- as.array(next_char_id)
  .[predictions, state] <- generation_model(list(next_char_id, state))
}


output <- generated_ids |> ids_to_chars() |> str_flatten("")
writeLines(c(prompt, output))


zip_path <-
  "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip" |>
  get_file(origin = _, extract = TRUE)

fs::dir_tree(zip_path, recurse = 1)


text_path <- fs::path(zip_path, "spa-eng/spa.txt")


text_pairs <- text_path |>
  readr::read_tsv(col_names = c("english", "spanish"),
                  col_types = c("cc")) |>
  dplyr::mutate(spanish = str_c("[start] ", spanish, " [end]"))


set.seed(1)


text_pairs |> dplyr::slice_sample(n = 1) |> dplyr::glimpse()


num_test_samples <- num_val_samples <-
  round(0.15 * nrow(text_pairs))
num_train_samples <- nrow(text_pairs) - num_val_samples - num_test_samples

pair_group <- sample(c(
  rep("train", num_train_samples),
  rep("test", num_test_samples),
  rep("val", num_val_samples)
))

train_pairs <- text_pairs[pair_group == "train", ]
test_pairs <- text_pairs[pair_group == "test", ]
val_pairs <- text_pairs[pair_group == "val", ]


punctuation_regex <- r"---([!"#$%&'()*+,./:;<=>?@\\^_`{|}~¡¿-])---"

library(tensorflow, exclude = c("set_random_seed", "shape"))
custom_standardization <- function(input_string) {
  input_string |>
    tf$strings$lower() |>
    tf$strings$regex_replace(punctuation_regex, "")
}

input_string <- as_tensor("[start] ¡corre! [end]")
custom_standardization(input_string)


vocab_size <- 15000
sequence_length <- 20

english_tokenizer <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length
)

spanish_tokenizer <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length + 1,
  standardize = custom_standardization
)

adapt(english_tokenizer, train_pairs$english)
adapt(spanish_tokenizer, train_pairs$spanish)


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

batch_size <- 64

library(tfdatasets)
make_dataset <- function(pairs) {
  tensor_slices_dataset(pairs) |>
    dataset_map(format_pair, num_parallel_calls = 4) |>
    dataset_cache() |>
    dataset_shuffle(2048) |>
    dataset_batch(batch_size) |>
    dataset_prefetch(16)
}

train_ds <- make_dataset(train_pairs)
val_ds <- make_dataset(val_pairs)


.[inputs, targets] <- iter_next(as_iterator(train_ds))
str(inputs)
str(targets)


inputs <- keras_input(shape = c(sequence_length), dtype = "int32")
outputs <- inputs |>
  layer_embedding(input_dim = vocab_size, output_dim = 128) |>
  layer_lstm(32, return_sequences = TRUE) |>
  layer_dense(vocab_size, activation = "softmax")

model <- keras_model(inputs, outputs)


embed_dim <- 256
hidden_dim <- 1024

source <- keras_input(shape = c(NA), dtype = "int32", name = "english")

encoder_output <- source |>
  layer_embedding(vocab_size, embed_dim, mask_zero = TRUE) |>
  bidirectional(layer_gru(units = hidden_dim), merge_mode = "sum")


target <- keras_input(shape = c(NA), dtype = "int32", name = "spanish")

rnn_layer <- layer_gru(units = hidden_dim, return_sequences = TRUE)

target_predictions <- target |>
  layer_embedding(vocab_size, embed_dim, mask_zero = TRUE) |>
  rnn_layer(initial_state = encoder_output) |>
  layer_dropout(0.5) |>
  layer_dense(vocab_size, activation = "softmax")                               # <1>

seq2seq_rnn <- keras_model(list(source, target), target_predictions)


seq2seq_rnn |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  weighted_metrics = "accuracy"
)

fit(seq2seq_rnn, train_ds, epochs = 15, validation_data = val_ds)


spa_vocab <- spanish_tokenizer$get_vocabulary()

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

generate_translation("hello")
generate_translation("You know that.")
generate_translation("Thanks.")
generate_translation("You're welcome.")
generate_translation("I think they're happy.")


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
    self$feed_forward_1 <- layer_dense(, intermediate_dim, activation = "relu") # <2>
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


layer_normalization <- function(batch_of_sequences) {
  mean <- op_mean(batch_of_sequences, axis = -1, keepdims = TRUE)               # <1>
  variance <- op_var(batch_of_sequences, axis = -1, keepdims = TRUE)            # <2>
  (batch_of_sequences - mean) / variance                                        # <2>
}


batch_normalization <- function(batch_of_images) {
  mean <- op_mean(batch_of_images, axis = c(1, 2, 3), keepdims = TRUE)          # <1>
  variance <- op_var(batch_of_images, axis = c(1, 2, 3), keepdims = TRUE)       # <2>
  (batch_of_images - mean) / variance                                           # <2>
}


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
    self$feed_forward_1 <- layer_dense(, intermediate_dim, activation = "relu") # <3>
    self$feed_forward_2 <- layer_dense(, hidden_dim)                            # <3>
    self$feed_forward_layernorm <- layer_layer_normalization()                  # <3>
  },
  call = function(target, source, source_mask) {
    residual <- x <- target                                                     # <4>
    x <- self$self_attention(query = x, key = x, value = x,
                             use_causal_mask = TRUE)
    x <- x + residual                                                           # <4>
    x <- self$self_attention_layernorm(x)                                       # <4>

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
    x <- self$feed_forward_layernorm(x)                                         # <6>

    x
  }
)


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


transformer


transformer |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  weighted_metrics = "accuracy"
)

transformer |> fit(train_ds, epochs = 15, validation_data = val_ds)


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


transformer |> compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    weighted_metrics="accuracy"
)
transformer |> fit(train_ds, epochs = 30, validation_data=val_ds)


spa_vocab <- get_vocabulary(spanish_tokenizer)

generate_translation <- function(input_sentence) {

  tokenized_input_sentence <- english_tokenizer(list(input_sentence))
  decoded_sentence <- "[start]"

  for (i in seq_len(sequence_length)) {
    tokenized_target_sentence <- spanish_tokenizer(list(decoded_sentence))
    tokenized_target_sentence <- tokenized_target_sentence@r[, NA:-2] # drop last token

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

for (i in sample.int(nrow(test_pairs), 5)) {
  .[english, spanish] <- test_pairs[i, ]
  input_sentence = english
  translated <- generate_translation(english)
  cat("-- example", i, "--\n",
      "  english:", english, "\n",
       " spanish:", spanish, "\n",
      "predicted:", translated, "\n")
}


reticulate::py_require("keras-hub==0.18.1")
keras_hub <- reticulate::import("keras_hub")


tokenizer <- keras_hub$models$Tokenizer$from_preset("roberta_base_en")
backbone <- keras_hub$models$Backbone$from_preset("roberta_base_en")


tokenizer("The quick brown fox")


backbone


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


library(tfdatasets, exclude = "shape")
preprocess <- function(text, label) {
  packer <- keras_hub$layers$StartEndPacker(
    sequence_length = 512L,
    start_value = tokenizer$start_token_id,
    end_value = tokenizer$end_token_id,
    pad_value = tokenizer$pad_token_id,
    return_padding_mask = TRUE,
  )
  .[token_ids, padding_mask] <- text |> tokenizer() |> packer()
  list(list(token_ids = token_ids, padding_mask = padding_mask),
       label)
}
preprocessed_train_ds <- train_ds |> dataset_map(preprocess)
preprocessed_val_ds <- val_ds |> dataset_map(preprocess)
preprocessed_test_ds <- test_ds |> dataset_map(preprocess)


preprocessed_train_ds |> as_iterator() |> iter_next() |> str()


inputs <- backbone$input
outputs <- inputs |>
  backbone() |>
  op_subset(, 1, ) |>                                                           # <1>
  layer_dropout(0.1) |>
  layer_dense(768, activation = "relu") |>
  layer_dropout(0.1) |>
  layer_dense(1, activation = "sigmoid")

classifier <- keras_model(inputs, outputs)


classifier |> compile(
  optimizer = optimizer_adam(5e-5),
  loss = "binary_crossentropy",
  metrics = "accuracy"
)
classifier |> fit(
  preprocessed_train_ds,
  validation_data = preprocessed_val_ds
)


evaluate(classifier, preprocessed_test_ds)



