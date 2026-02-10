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
#| lst-cap: "Downloading some of Shakespeare's work"
library(keras3)

filename = get_file(origin = paste0(
  "https://storage.googleapis.com/download.tensorflow.org/",
  "data/shakespeare.txt"
))
shakespeare <- readLines(filename)


# ----------------------------------------------------------------------
writeLines(head(shakespeare, 14))


# ----------------------------------------------------------------------
#| lst-cap: Splitting text into chunks for language model training
sequence_length <- 100                                                          # <1>

split_input <- function(text, sequence_length) {
  starts <- seq.int(1, str_length(text), by = sequence_length)
  str_sub(text, cbind(starts, length = sequence_length))
}

shakespeare <- shakespeare |> str_flatten("\n")
features <- shakespeare |> str_sub(end = -2) |> split_input(sequence_length)
labels <- shakespeare |> str_sub(start = 2) |> split_input(sequence_length)

dataset <- tensor_slices_dataset(tuple(features, labels))


# ----------------------------------------------------------------------
dataset |>
  as_iterator() |> iter_next() |>
  lapply(tf$strings$substr, 0L, len = 20L)


# ----------------------------------------------------------------------
#| lst-cap: "Learning a character-level vocabulary with `TextVectorization`"
tokenizer <- layer_text_vectorization(
  standardize = NULL,
  split = "character",
  output_sequence_length = sequence_length
)
features_only_dataset <- dataset |> dataset_map(\(text, labels) text)
adapt(tokenizer, features_only_dataset)


# ----------------------------------------------------------------------
vocabulary_size <- tokenizer$vocabulary_size()
vocabulary_size


# ----------------------------------------------------------------------
dataset <- dataset |>
  dataset_map(\(features, labels) {
    tuple(tokenizer(features), tokenizer(labels))
  }, num_parallel_calls = 8)

training_data <-  dataset |>
  dataset_cache() |>
  dataset_shuffle(10000) |>
  dataset_batch(64)


# ----------------------------------------------------------------------
#| lst-cap: Building a miniature language model
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


# ----------------------------------------------------------------------
model


# ----------------------------------------------------------------------
#| lst-cap: Training a miniature language model
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "sparse_categorical_accuracy"
)
model |> fit(training_data, epochs = 20)


# ----------------------------------------------------------------------
#| lst-cap: Modifying the language model for autoregressive inference
inputs <- keras_input(shape = c(1), dtype = "int", name = "token_ids")          # <1>
input_state <- keras_input(shape = c(hidden_dim), name = "state")
gru <- layer_gru(units = hidden_dim, return_state = TRUE)

x <- inputs |> layer_embedding(vocabulary_size, embedding_dim)
.[x, output_state] <- gru(x, initial_state = input_state)
outputs <- x |> layer_dense(vocabulary_size, activation="softmax")
generation_model <- keras_model(
  inputs = list(inputs, input_state),
  outputs = list(outputs, output_state)
)
set_weights(generation_model, get_weights(model))                               # <2>


# ----------------------------------------------------------------------
vocab <- get_vocabulary(tokenizer)

chars_to_ids <- \(chars) match(chars, vocab, nomatch = 2L) - 1L                 # <1>
ids_to_chars <- \(ids) vocab[ids + 1L]                                          # <1>

prompt <- r"--(
KING RICHARD III:
)--"


# ----------------------------------------------------------------------
#| lst-cap: Computing a language model’s starting state
input_ids <- chars_to_ids(str_split_1(prompt, ""))
state <- op_zeros(shape = c(1, hidden_dim))
for (token_id in input_ids) {
  inputs <- op_expand_dims(token_id, axis = 1)
  .[predictions, state] <- generation_model(tuple(inputs, state))              # <1>
}


# ----------------------------------------------------------------------
#| lst-cap: Predicting with the language model a token at a time
max_length <- 250
generated_ids <- integer(max_length)

for (i in seq_len(max_length)) {                                                # <1>
  next_char_id <- op_argmax(predictions, axis = -1,                             # <2>
                            zero_indexed = TRUE, keepdims = TRUE)
  generated_ids[i] <- as.array(next_char_id)
  .[predictions, state] <- generation_model(list(next_char_id, state))
}


# ----------------------------------------------------------------------
output <- generated_ids |> ids_to_chars() |> str_flatten("")
writeLines(c(prompt, output))


