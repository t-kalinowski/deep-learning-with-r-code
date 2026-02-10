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
py_require("keras_hub")
keras_hub <- import("keras_hub")
library(tfdatasets, exclude = "shape")
library(tensorflow, exclude = c("shape", "set_random_seed"))


# ----------------------------------------------------------------------
batch_size <- 128
sequence_length <- 256
num_batches <- 29373
num_val_batches <- 500
hidden_dim <- 512
intermediate_dim <- 2056
num_heads <- 8
num_layers <- 8
num_epochs <- 8
prompt <- "A piece of advice"


# ----------------------------------------------------------------------
read_file <- function(filename) {
  text_line_dataset(filename) |>
    dataset_map(\(x) tf$strings$regex_replace(x, "\\\\n", "\n")) |>             # <1>
    dataset_map(tokenizer, num_parallel_calls = 8) |>                           # <2>
    dataset_map(\(x) tf$concat(list(x, suffix), -1L))                           # <3>
}

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

greedy_search <- function(preds) {
  op_argmax(preds)
}

random_sample <- function(preds, temperature = 1) {
  preds <- preds / temperature
  preds <- op_reshape(preds, c(1, -1))
  random_categorical(preds, num_samples = 1) |> op_squeeze()
}

top_k <- function(preds, k = 5, temperature = 1) {
  preds <-  preds / temperature
  .[top_preds, top_indices] <- op_top_k(preds, k = k, sorted = FALSE)
  choice <- random_sample(top_preds)
  op_take(top_indices, choice)
}


# ----------------------------------------------------------------------
# Split marker for notebook/code extraction.


# ----------------------------------------------------------------------
#| eval: false
# py_require("kagglehub")
# import("kagglehub")$login()


# ----------------------------------------------------------------------
rm(list = setdiff(ls(), c("keras_hub"))); gc()
import("gc")$collect()


# ----------------------------------------------------------------------
#| lst-cap: Instantiating a pretrained LLM with KerasHub
py_require("keras-hub")
keras_hub <- import("keras_hub")

gemma_lm <- keras_hub$models$CausalLM$from_preset(
  "gemma3_1b",
  dtype = "float32"
)


# ----------------------------------------------------------------------
gemma_lm


# ----------------------------------------------------------------------
gemma_lm$compile(sampler = "greedy")
cat(gemma_lm$generate("A piece of advice", max_length = 40L))
cat(gemma_lm$generate("How can I make brownies?", max_length = 40L))


# ----------------------------------------------------------------------
gemma_lm$generate(
  paste0(
    "The following brownie recipe is easy to make in just a few steps.",
    "\n\nYou can start by"
  ),
  max_length = 40L
) |> cat()


# ----------------------------------------------------------------------
gemma_lm$generate(
  "Tell me about the 542nd president of the United States.",
  max_length = 40L
) |> cat()


