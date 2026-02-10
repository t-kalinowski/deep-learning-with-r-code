# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "fs", "glue", "jsonlite", "purrr", "readr", "stringr", "tfdatasets")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(fs)
library(stringr)
library(keras3)
use_backend("jax")
py_require("keras-hub")
Sys.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION" = "1.00")
# config_set_dtype_policy("float16")


# ----------------------------------------------------------------------
#| lst-cap: Downloading a portion of the C4 dataset
library(keras3)

zipfile <-
  "https://hf.co/datasets/mattdangerw/mini-c4/resolve/main/mini-c4.zip" |>
  get_file(origin = _)

unzip(zipfile, exdir = ".")
extract_dir <- fs::path("./mini-c4")


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
library(stringr)

fs::path(extract_dir, "shard0.txt") |> readLines(n = 1) |>
  str_replace_all(r"(\\n)", "\n") |> str_sub(1, 100) |> cat()


# ----------------------------------------------------------------------
#| lst-cap: Downloading a vocabulary and instantiating a tokenizer
py_require("keras_hub")
keras_hub <- import("keras_hub")

vocabulary_file <- get_file(
  origin = "https://hf.co/mattdangerw/spiece/resolve/main/vocabulary.proto"
)
tokenizer <- keras_hub$tokenizers$SentencePieceTokenizer(vocabulary_file)


# ----------------------------------------------------------------------
(tokenized <- tokenizer$tokenize("The quick brown fox."))
tokenizer$detokenize(tokenized)


# ----------------------------------------------------------------------
#| eval: false
# library(fs)
# library(readr)
# library(purrr)
# library(stringr)
# library(reticulate)
# py_require("sentencepiece")
# spm <- import("sentencepiece")
# 
# tmp_dir <- path_temp("spm_corpus")
# dir_create(tmp_dir)
# 
# files <- Sys.glob("./mini-c4/*.txt")
# new_files <- path(tmp_dir, basename(files))
# files <- map2_chr(files, new_files, \(f, f2) {
#   f |>
#     read_file() |>
#     str_replace_all(r"(\\n)", "\n") |>
#     write_file(f2)
#   f2
#   }, .progress = TRUE)
# 
# spm$SentencePieceTrainer$train(
#   input = files |> str_flatten(","),
#   model_prefix = "mini_gpt",
#   vocab_size = 2048,
#   input_sentence_size = 10000,
#   max_sentence_length = 50000,
#   shuffle_input_sentence = TRUE
# )


# ----------------------------------------------------------------------
#| eval: false
# mini_tokenizer <-
#   keras_hub$tokenizers$SentencePieceTokenizer("mini_gpt.model")
# mini_tokenizer$tokenize("The quick brown fox.")


# ----------------------------------------------------------------------
#| lst-cap: Preprocessing text input for Transformer pretraining
library(tfdatasets, exclude = "shape")
library(tensorflow, exclude = c("shape", "set_random_seed"))

batch_size <- 128
sequence_length <- 256
suffix <- tf$constant(
  tokenizer$token_to_id("<|endoftext|>"),
  shape = shape(1L)
)

files <- dir_ls(extract_dir, glob = "*.txt")

read_file <- function(filename) {
  text_line_dataset(filename) |>
    dataset_map(\(x) tf$strings$regex_replace(x, "\\\\n", "\n")) |>             # <1>
    dataset_map(tokenizer, num_parallel_calls = 8) |>                           # <2>
    dataset_map(\(x) tf$concat(list(x, suffix), -1L))                           # <3>
}


ds <- tensor_slices_dataset(files) |>
  dataset_interleave(read_file, cycle_length = 32,
                     num_parallel_calls = 32) |>                                # <4>
  dataset_rebatch(sequence_length + 1, drop_remainder = TRUE)  |>               # <5>
  dataset_map(\(x) list(x@r[NA:-2], x@r[2:NA]),                                 # <6>
              num_parallel_calls = 12) |>
  dataset_batch(batch_size) |>
  dataset_prefetch(buffer_size = 8)


# ----------------------------------------------------------------------
num_batches <- 29373
num_val_batches <- 500
num_train_batches <- num_batches - num_val_batches
val_ds <- ds |> dataset_take(num_val_batches) |> dataset_repeat()
train_ds <- ds |> dataset_skip(num_val_batches) |> dataset_repeat()


# ----------------------------------------------------------------------
#| lst-cap: Transformer decoder block without cross-attention
layer_transformer_decoder <- new_layer_class(
  "TransformerDecoder",
  initialize = function(hidden_dim, intermediate_dim, num_heads) {
    super$initialize()
    key_dim <- hidden_dim %/% num_heads
    self$self_attention <- layer_multi_head_attention(
      num_heads = num_heads,
      key_dim = key_dim,
      dropout = 0.1
    )                                                                           # <1>
    self$self_attention_layernorm <- layer_layer_normalization()                # <1>
    self$feed_forward_1 <- layer_dense(units = intermediate_dim,
                                       activation = "relu")                     # <2>
    self$feed_forward_2 <- layer_dense(units = hidden_dim)                      # <2>
    self$feed_forward_layernorm <- layer_layer_normalization()                  # <2>
    self$dropout <- layer_dropout(rate = 0.1)                                   # <2>
  },
  call = function(inputs) {
    residual <- x <- inputs                                                     # <3>
    x <- self$self_attention(query = x, key = x, value = x,
                             use_causal_mask = TRUE)                            # <3>
    x <- x |> self$dropout()                                                    # <3>
    x <- x + residual                                                           # <3>
    x <- x |> self$self_attention_layernorm()                                   # <3>

    residual <- x                                                               # <4>
    x <- x |>
      self$feed_forward_1() |>                                                  # <4>
      self$feed_forward_2() |>                                                  # <4>
      self$dropout()
    x <- x + residual                                                           # <4>
    x <- x |> self$feed_forward_layernorm()

    x
  }
)


# ----------------------------------------------------------------------
#| lst-cap: Positional embedding layer that can reverse a text embedding
layer_positional_embedding <- new_layer_class(
  "PositionalEmbedding",
  initialize = function(sequence_length, input_dim, output_dim) {
    super$initialize()
    self$token_embeddings <- layer_embedding(
      input_dim = input_dim, output_dim = output_dim
    )
    self$position_embeddings <- layer_embedding(
      input_dim = sequence_length, output_dim = output_dim
    )
  },
  call = function(inputs, reverse = FALSE) {
    if (reverse) {
      token_embeddings <- self$token_embeddings$embeddings
      return(inputs %*% t(token_embeddings))                                    # <1>
    }
    .[.., sequence_length] <- op_shape(inputs)
    positions <-
      op_arange(0, sequence_length - 1, dtype = "int32") |>
      op_expand_dims(1)
    embedded_tokens <- self$token_embeddings(inputs)
    embedded_positions <- self$position_embeddings(positions)
    embedded_tokens + embedded_positions
  }
)


# ----------------------------------------------------------------------
#| lst-cap: Creating a mini-GPT functional model
config_set_dtype_policy("mixed_float16")                                        # <1>
vocab_size <- tokenizer$vocabulary_size()
hidden_dim <- 512
intermediate_dim <- 2056
num_heads <- 8
num_layers <- 8

inputs <- keras_input(shape = c(NA), dtype = "int32", name = "inputs")
embedding <-
  layer_positional_embedding(, sequence_length, vocab_size, hidden_dim)

x <- inputs |>
  embedding() |>
  layer_layer_normalization()

for (i in seq_len(num_layers)) {
  x <- x |>
    layer_transformer_decoder(hidden_dim, intermediate_dim, num_heads)
}

outputs <- x |> embedding(reverse = TRUE)
mini_gpt <- keras_model(inputs, outputs)


# ----------------------------------------------------------------------
#| lst-cap: Defining a custom learning rate schedule
warmup_schedule <- new_learning_rate_schedule_class(
  classname = "WarmupSchedule",

  initialize = function() {
    self$rate <- 2e-4                                                           # <1>
    self$warmup_steps <- 1000
  },

  call = function(step) {
    step <- step |> op_cast(dtype = "float32")
    scale <- op_minimum(step / self$warmup_steps, 1)
    self$rate * scale
  }
)


# ----------------------------------------------------------------------
#| fig-cap: Warmup makes our updates to model parameters smaller at the beginning of training and can help with stability.
schedule <- warmup_schedule()
x <- seq(0, 5000, by = 10)
y <- sapply(x, \(step) as.array(schedule(step)))

par(mar = c(5, 7, 4, 2), bty = "n", ann = FALSE)
plot(x, y, type = "l", lwd = 2, panel.first = grid())
title(main = "Warmup Schedule", xlab = "Train Step")
title(ylab = "Learning Rate", line = 4.5)


# ----------------------------------------------------------------------
#| lst-cap: Training the mini-GPT model
num_epochs <- 8
steps_per_epoch <- num_train_batches %/% num_epochs                             # <1>
validation_steps <- num_val_batches                                             # <1>

mini_gpt |> compile(
  optimizer = optimizer_adam(schedule),
  loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
  metrics = "accuracy"
)

mini_gpt |> fit(
  train_ds,
  validation_data = val_ds,
  epochs = num_epochs,
  steps_per_epoch = steps_per_epoch,
  validation_steps = validation_steps
)


# ----------------------------------------------------------------------
#| lst-cap: Simple generation function for the mini-GPT model
generate <- function(prompt, max_length = 64) {
  tokens <- as.array(tokenizer(prompt))
  prompt_length <- length(tokens)
  for (i in seq(from = prompt_length + 1, to = max_length)) {
    prediction <- mini_gpt(matrix(tokens, nrow = 1))
    prediction <- prediction@r[1, -1]
    next_token <- op_argmax(prediction, zero_indexed = TRUE)
    tokens[i] <- as.array(next_token)
  }
  tokenizer$detokenize(tokens)
}


# ----------------------------------------------------------------------
prompt <- "A piece of advice"
cat(generate(prompt))


# ----------------------------------------------------------------------
#| lst-cap: Compiled generation function for the mini-GPT model
compiled_generate <- function(prompt, max_length = 64) {
  tokens <- as.array(tokenizer(prompt))
  prompt_length <- length(tokens)
  tokens[seq(prompt_length + 1, max_length)] <- 0L                              # <1>
  dim(tokens) <- c(1, max_length)
  storage.mode(tokens) <- "integer"
  for (i in seq(prompt_length, max_length - 1)) {
    prediction <- mini_gpt |> predict(tokens, verbose = 0)
    prediction <- prediction[, i, ]
    next_token <- which.max(prediction) - 1L
    tokens[, i + 1] <- next_token
  }
  tokenizer$detokenize(tokens)
}


# ----------------------------------------------------------------------
system.time(compiled_generate(prompt, 64))[["elapsed"]]


# ----------------------------------------------------------------------
compiled_generate <- function(prompt, sample_fn, max_length = 64) {
  tokens <- as.array(tokenizer(prompt))
  prompt_length <- length(tokens)
  tokens[seq(prompt_length + 1, max_length)] <- 0L
  dim(tokens) <- c(1, max_length)
  storage.mode(tokens) <- "integer"
  for (i in seq(prompt_length, max_length - 1)) {
    prediction <- predict(mini_gpt, tokens, verbose = 0)
    prediction <- prediction[, i, ]
    next_token <- sample_fn(prediction) - 1L
    tokens[, i + 1] <- as.array(next_token)
  }
  tokenizer$detokenize(tokens)
}


# ----------------------------------------------------------------------
#| results: hide
greedy_search <- function(preds) {
  op_argmax(preds)
}

compiled_generate(prompt, greedy_search)


# ----------------------------------------------------------------------
random_sample <- function(preds, temperature = 1) {
  preds <- preds / temperature
  preds <- op_reshape(preds, c(1, -1))
  random_categorical(preds, num_samples = 1) |> op_squeeze()
}


# ----------------------------------------------------------------------
cat(compiled_generate(prompt, random_sample))


# ----------------------------------------------------------------------
compiled_generate(prompt, \(x) random_sample(x, temperature = 2))


# ----------------------------------------------------------------------
compiled_generate(prompt, \(x) random_sample(x, temperature = 0.8))


# ----------------------------------------------------------------------
compiled_generate(prompt, \(x) random_sample(x, temperature = 0.2))


# ----------------------------------------------------------------------
#| results: hide
top_k <- function(preds, k = 5, temperature = 1) {
  preds <-  preds / temperature
  .[top_preds, top_indices] <- op_top_k(preds, k = k, sorted = FALSE)
  choice <- random_sample(top_preds)
  op_take(top_indices, choice)
}


# ----------------------------------------------------------------------
compiled_generate(prompt, \(preds) top_k(preds, k = 5))


# ----------------------------------------------------------------------
compiled_generate(prompt, \(preds) top_k(preds, k = 20))


# ----------------------------------------------------------------------
compiled_generate(prompt, \(preds) top_k(preds, k = 5, temperature=0.5))


