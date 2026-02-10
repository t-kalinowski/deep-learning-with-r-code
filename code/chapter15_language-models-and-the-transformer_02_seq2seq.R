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


# ----------------------------------------------------------------------
embedding_dim <- 256L
hidden_dim <- 1024L
max_length <- 250


# ----------------------------------------------------------------------
split_input <- function(text, sequence_length) {
  starts <- seq.int(1, str_length(text), by = sequence_length)
  str_sub(text, cbind(starts, length = sequence_length))
}


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
zip_path <- get_file(
  origin = paste0(
    "http://storage.googleapis.com/download.tensorflow.org/",
    "data/spa-eng.zip"
  ),
  extract = TRUE
)

fs::dir_tree(zip_path, recurse = 1)


# ----------------------------------------------------------------------
text_path <- fs::path(zip_path, "spa-eng/spa.txt")


# ----------------------------------------------------------------------
text_pairs <- text_path |>
  readr::read_tsv(col_names = c("english", "spanish"),
                  col_types = c("cc")) |>
  dplyr::mutate(spanish = str_c("[start] ", spanish, " [end]"))


# ----------------------------------------------------------------------
text_pairs
text_pairs |> dplyr::slice_sample(n = 3)


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
#| lst-cap: Learning token vocabularies for English and Spanish text
punctuation_regex <- r"---([!"#$%&'()*+,./:;<=>?@\\^_`{|}~¡¿-])---"

library(tensorflow, exclude = c("set_random_seed", "shape"))
custom_standardization <- function(input_string) {
  input_string |>
    tf$strings$lower() |>
    tf$strings$regex_replace(punctuation_regex, "")
}

input_string <- as_tensor("[start] ¡corre! [end]")
custom_standardization(input_string)


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
#| lst-cap: Tokenizing and preparing the translation data
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

library(tfdatasets, exclude = "shape")
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


# ----------------------------------------------------------------------
.[inputs, targets] <- iter_next(as_iterator(train_ds))
str(inputs)
str(targets)


# ----------------------------------------------------------------------
inputs <- keras_input(shape = c(sequence_length), dtype = "int32")
outputs <- inputs |>
  layer_embedding(input_dim = vocab_size, output_dim = 128) |>
  layer_lstm(32, return_sequences = TRUE) |>
  layer_dense(vocab_size, activation = "softmax")

model <- keras_model(inputs, outputs)


# ----------------------------------------------------------------------
#| lst-cap: Building a sequence-to-sequence encoder
embed_dim <- 256
hidden_dim <- 1024

source <- keras_input(shape = c(NA), dtype = "int32", name = "english")

encoder_output <- source |>
  layer_embedding(vocab_size, embed_dim, mask_zero = TRUE) |>
  bidirectional(layer_gru(units = hidden_dim), merge_mode = "sum")


# ----------------------------------------------------------------------
#| lst-cap: Building a sequence-to-sequence decoder
target <- keras_input(shape = c(NA), dtype = "int32", name = "spanish")

rnn_layer <- layer_gru(units = hidden_dim, return_sequences = TRUE)

target_predictions <- target |>
  layer_embedding(vocab_size, embed_dim, mask_zero = TRUE) |>
  rnn_layer(initial_state = encoder_output) |>
  layer_dropout(0.5) |>
  layer_dense(vocab_size, activation = "softmax")                               # <1>

seq2seq_rnn <- keras_model(list(source, target), target_predictions)


# ----------------------------------------------------------------------
seq2seq_rnn


# ----------------------------------------------------------------------
seq2seq_rnn |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  weighted_metrics = "accuracy"
)

fit(seq2seq_rnn, train_ds, epochs = 15, validation_data = val_ds)


# ----------------------------------------------------------------------
#| lst-cap: Generating translations with a seq2seq RNN
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
# generate_translation("Hello")
# generate_translation("You know that.")
# generate_translation("Thanks.")
# generate_translation("You're welcome.")
# generate_translation("I think they're happy.")


