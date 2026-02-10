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
#
# Ensure the fine-tuning notebook is standalone by (re)loading the base model.
keras_hub <- reticulate::import("keras_hub")
gemma_lm <- keras_hub$models$CausalLM$from_preset("gemma3_1b")


# ----------------------------------------------------------------------
#| lst-cap: Loading an instruction fine-tuning dataset
library(dplyr, warn.conflicts = FALSE)

format_prompt <-
  \(instruction) paste0("[instruction]\n", instruction, "[end]\n[response]\n")
format_response <-
  \(response) paste0(response, "[end]")


dataset_path <- get_file(origin = paste0(
  "https://hf.co/datasets/databricks/databricks-dolly-15k/",
  "resolve/main/databricks-dolly-15k.jsonl"
))

data <- readr::read_lines(dataset_path) |>
  lapply(jsonlite::parse_json) |>
  bind_rows()

glimpse(data)


# ----------------------------------------------------------------------
data <- data |>
  filter(context == "") |>
  mutate(
    prompts = format_prompt(instruction),
    responses = format_response(response),
    .keep = 'none'
  )


# ----------------------------------------------------------------------
str(data[2,])


# ----------------------------------------------------------------------
library(tfdatasets, exclude = "shape")

ds <- data |>
  tensor_slices_dataset() |>
  dataset_shuffle(2000) |>
  dataset_batch(2)

val_ds <- ds |> dataset_take(100)
train_ds <- ds |> dataset_skip(100)


# ----------------------------------------------------------------------
preprocessor <- gemma_lm$preprocessor
preprocessor$sequence_length <- 512L
batch <- iter_next(as_iterator(train_ds))
.[x, y, sample_weight] <- preprocessor(batch)
str(x)
str(y)
str(sample_weight)


# ----------------------------------------------------------------------
# bind
rbind(x$token_ids |> as.array() |> _[1, 1:5],
      y |> as.array() |> _[1, 1:5])


# ----------------------------------------------------------------------
layer_linear <- new_layer_class(
  classname = "Linear",
  initialize = function(input_dim, output_dim) {
    super$initialize()
    self$kernel <- self$add_weight(shape = shape(input_dim, output_dim))
  },
  call = function(inputs) {
    inputs %*% self$kernel                                                      # <1>
  }
)


# ----------------------------------------------------------------------
layer_lora_linear <- new_layer_class(
  classname = "LoraLinear",
  initialize = function(input_dim, output_dim, rank) {
    super$initialize()
    self$kernel <- self$add_weight(shape(input_dim, output_dim),
                                   trainable = FALSE)
    self$alpha <- self$add_weight(shape(input_dim, rank))
    self$beta <- self$add_weight(shape(rank, output_dim))
  },
  call = function(inputs) {
    frozen <- inputs %*% self$kernel
    update <- inputs %*% self$alpha %*% self$beta
    frozen + update
  }
)


# ----------------------------------------------------------------------
#| eval: true
#| lst-cap: Enabling LoRA training for a KerasHub model
gemma_lm$backbone$enable_lora(rank = 8L)


# ----------------------------------------------------------------------
#| eval: false
# gemma_lm$backbone$trainable <- FALSE                                            # <1>
# for (i in seq_len(gemma_lm$backbone$num_layers) - 1) {
#   layer <- get_layer(gemma_lm$backbone, sprintf("decoder_block_%d", i))         # <2>
#   layer$attention$key_dense$trainable <- TRUE
#   layer$attention$key_dense$enable_lora(rank = 8L)
#   layer$attention$query_dense$trainable <- TRUE
#   layer$attention$query_dense$enable_lora(rank = 8L)
# }


# ----------------------------------------------------------------------
gemma_lm


# ----------------------------------------------------------------------
#| lst-cap: Fine-tuning a pretrained LLM
gemma_lm |> compile(
  loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
  optimizer = optimizer_adam(5e-5),
  weighted_metrics = metric_sparse_categorical_accuracy()
)
gemma_lm |> fit(train_ds, validation_data = val_ds, epochs = 1)


# ----------------------------------------------------------------------
gemma_lm$generate(
  format_prompt("How can I make brownies?"),
  max_length = 512L
)


# ----------------------------------------------------------------------
gemma_lm$generate(
  format_prompt("What is a proper noun?"),
  max_length = 512L
) |> cat()


# ----------------------------------------------------------------------
gemma_lm$generate(
  format_prompt("Who is the 542nd president of the United States?"),
  max_length = 512L
) |> cat()


# ----------------------------------------------------------------------
#| eval: false
#| lst-cap: Pseudocode for the simplest possible RLHF algorithm
# for (prompts in dataset) {
#   responses <- model$generate(prompts)                                          # <1>
#   rewards <- reward_model |> predict(responses)                                 # <2>
#   good_responses <- responses[rewards > cutoff]
#   model |> fit(good_responses)                                                  # <3>
# }


# ----------------------------------------------------------------------
rm(list = setdiff(ls(), "keras_hub")); gc();
import("gc")$collect(); gc()


# ----------------------------------------------------------------------
#| lst-cap: Loading an instruction-tuned Gemma variant
gemma_lm <- keras_hub$models$CausalLM$from_preset(
  "gemma3_instruct_4b",
  dtype = "float16"
)


# ----------------------------------------------------------------------
template_format <- \(prompt) paste0(
  "<start_of_turn>user\n", prompt,
  "<end_of_turn>\n<start_of_turn>model"
)


# ----------------------------------------------------------------------
prompt <- "Why can't you assign values in Jax tensors? Be brief!"
cat(gemma_lm$generate(template_format(prompt), max_length = 512L))


# ----------------------------------------------------------------------
prompt <- "Who is the 542nd president of the United States?"
cat(gemma_lm$generate(template_format(prompt), max_length = 512L))


# ----------------------------------------------------------------------
#| results: hide
#| fig-cap: An image used to demonstrate multimodal prompting with Gemma.
image_url <- paste0("https://github.com/mattdangerw/keras-nlp-scripts/",
                    "blob/main/learned-python.png?raw=true")
image_path <- get_file(origin = image_url)

image <- image_path |> image_load() |> image_to_array()
par(mar = c(0, 0, 0, 0))
plot(as.raster(image, max = 255L))


# ----------------------------------------------------------------------
gemma_lm$preprocessor$max_images_per_prompt <- 1L                               # <1>
gemma_lm$preprocessor$sequence_length <- 512L                                   # <1>
prompt <- "What is going on in this image? Be concise!<start_of_image>"
gemma_lm$generate(list(
  prompts = template_format(prompt),
  images = list(image)
))


# ----------------------------------------------------------------------
prompt = "What is the snake wearing?<start_of_image>"
gemma_lm$generate(list(
  prompts = template_format(prompt),
  images = list(image)
))


# ----------------------------------------------------------------------
prompt = glue::trim(r"(
  Judy wrote a 2-page letter to 3 friends twice a week for 3 months.
  How many letters did she write?
  Be brief, and add "ANSWER:" before your final answer.
  )")

gemma_lm$compile(sampler = "random")                                            # <1>


# ----------------------------------------------------------------------
gemma_lm$generate(template_format(prompt))


# ----------------------------------------------------------------------
gemma_lm$generate(template_format(prompt))


