library(fs)
library(stringr)
library(keras3)
reticulate::py_require("keras-hub==0.18.1")
reticulate::py_require("tensorflow-text")
config_set_dtype_policy("float16")


zipfile <- get_file(
  origin = "https://huggingface.co/datasets/mattdangerw/mini-c4/resolve/main/mini-c4.zip"
)


unzip(zipfile, list = TRUE)                                                     # <1>


extract_dir <- fs::path("./mini-c4")


fs::dir_info(extract_dir)
fs::path(extract_dir, "shard0.txt") |>
  readLines(n = 1) |>
  str_replace_all(r"(\\n)", "\n") |>
  str_split_1("\n") |>
  str_wrap(width = 76) |>
  cat(sep = "\n--\n")


keras_hub <- reticulate::import("keras_hub")
tokenizer <- keras_hub$tokenizers$SentencePieceTokenizer("mini_gpt.model")


tokenized <- tokenizer$tokenize("The quick brown fox.")
tokenized
tokenizer$detokenize(tokenized)


library(tfdatasets, exclude = "shape")
library(tensorflow, exclude = c("shape", "set_random_seed"))

batch_size <- 128L
sequence_length <- 256L
suffix <-
  tokenizer$token_to_id("<|endoftext|>") |>
  tf$constant(shape = shape(1))

files <- dir_ls(extract_dir)
ds <-
  text_line_dataset(files, num_parallel_reads = 12) |>
  dataset_map(\(x) tf$strings$regex_replace(x, r"(\\n)", "\n"),                 # <1>
              num_parallel_calls = 12) |>
  dataset_map(tokenizer, num_parallel_calls = 12) |>                            # <2>
  dataset_map(\(x) tf$concat(c(x, suffix), -1L), num_parallel_calls = 12) |>    # <3>
  dataset_rebatch(sequence_length + 1, drop_remainder = TRUE) |>                # <4>
  dataset_map(\(x) list(x@r[NA:-2], x@r[2:NA]), num_parallel_calls = 12) |>     # <5>
  dataset_batch(batch_size, num_parallel_calls = 12) |>                         # <6>
  dataset_cache()


num_batches <- 38581
num_batches


num_val_batches <- 500
num_train_batches <- num_batches - num_val_batches
val_ds <- ds |> dataset_take(num_val_batches)
train_ds <- ds |> dataset_skip(num_val_batches) |> dataset_repeat()


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
      return(op_matmul(inputs, op_transpose(token_embeddings)))
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


vocab_size <- tokenizer$vocabulary_size()
hidden_dim <- 128
intermediate_dim <- 512
num_heads <- 4
num_layers <- 4

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


hidden_dim <- 512
intermediate_dim <- 2056
num_heads <- 8
num_layers <- 8


warmup_schedule <- new_learning_rate_schedule_class(
  classname = "WarmupSchedule",

  initialize = function() {
    self$rate <- 1e-4                                                           # <1>
    self$warmup_steps <- 1000
  },

  call = function(step) {
    step <- step |> op_cast(dtype = "float32")
    scale <- op_minimum(step / self$warmup_steps, 1)
    self$rate * scale
  }
)


schedule <- warmup_schedule()
x <- seq(0, 5000, by = 100)
y <- sapply(x, \(step) as.array(schedule(step)))
plot(x, y, type = "l",
     main = "Warmup Schedule",
     xlab = "Train Step", ylab = "Learning Rate",
     bty = "n", panel.first = grid())


load_model_weights(mini_gpt, "mini_gpt.weights.h5")


prompt <- "A piece of advice"
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
generate("A piece of advice")


compiled_generate <- function(prompt, max_length = 64) {
  tokens <- as.array(tokenizer(prompt))
  prompt_length <- length(tokens)
  tokens[seq(prompt_length + 1, max_length)] <- 0                               # <1>
  dim(tokens) <- c(1, max_length)
  for (i in seq(prompt_length, max_length - 1)) {
    prediction <- predict(mini_gpt, tokens, verbose = 0)
    prediction <- prediction[, i, ]
    next_token <- which.max(prediction) - 1L
    tokens[, i + 1] <- next_token
  }
  tokenizer$detokenize(tokens)
}


system.time(compiled_generate(prompt, 64))[["elapsed"]]


compiled_generate <- function(prompt, sample_fn, max_length = 64) {
  tokens <- as.array(tokenizer(prompt))
  prompt_length <- length(tokens)
  tokens[seq(prompt_length + 1, max_length)] <- 0
  dim(tokens) <- c(1, max_length)
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

compiled_generate(prompt, greedy_search)


random_sample <- function(preds, temperature = 1) {
  preds <- preds / temperature
  preds <- op_reshape(preds, c(1, -1))
  random_categorical(preds, num_samples = 1) |>
    op_squeeze()
}


compiled_generate(prompt, random_sample)


compiled_generate(prompt, \(x) random_sample(x, temperature = 2))
compiled_generate(prompt, \(x) random_sample(x, temperature = 0.8))
compiled_generate(prompt, \(x) random_sample(x, temperature = 0.2))


prompt <- "A piece of advice"
top_k <- function(preds, k = 5, temperature = 1) {
  preds <-  preds / temperature
  .[top_preds, top_indices] <- op_top_k(preds, k = k, sorted = FALSE)
  choice <- random_sample(top_preds)
  op_take(top_indices, choice)
}


compiled_generate(prompt, \(x) top_k(x, k = 5, temperature = 0.5))


rm(list = ls()); gc(TRUE); reticulate::import("gc")$collect()
rm(list = ls()); gc(TRUE); reticulate::import("gc")$collect()


library(keras3)
library(reticulate)
py_require("keras_hub")

config_set_dtype_policy("float16")
keras_hub <- import("keras_hub")
kaggle_credentials <- jsonlite::read_json("~/.kaggle/kaggle.json")
withr::with_envvar(c(
  KAGGLE_USERNAME = kaggle_credentials$username,
  KAGGLE_KEY = kaggle_credentials$key), {
    gemma_lm <- keras_hub$models$GemmaCausalLM$from_preset("gemma2_2b_en")
  }
)


gemma_lm


gemma_lm$compile(sampler = "greedy")
gemma_lm$generate("A piece of advice", max_length = 64L)
gemma_lm$generate("How can I make brownies?", max_length = 64L)


gemma_lm$generate(
  paste0(
    "The following brownie recipe is easy to make in just a few steps.",
    "\n\nYou can start by"
  ),
  max_length = 64L
)


gemma_lm$generate(
  "Tell me about the 61st president of the United States.",
  max_length = 64L
)


TEMPLATE = glue::trim(r"---(
  [instruction]
  {instruction}[end]
  [reponse]
  {response}[end]
  )---")

dataset_path <- get_file(origin = paste0(
  "https://huggingface.co/datasets/databricks/",
  "databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
))

data <- readr::read_lines(dataset_path) |>
  lapply(jsonlite::parse_json) |>
  dplyr::bind_rows()

data

data <- data |>
  dplyr::filter(context != "") |>
  glue::glue_data(TEMPLATE)


writeLines(data[[1]])


library(tfdatasets)
ds <- tensor_slices_dataset(data) |>
  dataset_shuffle(2000) |>
  dataset_batch(1)
val_ds <- ds |> dataset_take(100)
train_ds <- ds |> dataset_skip(100)


preprocessor <- gemma_lm$preprocessor
preprocessor$sequence_length <- 512L
batch <- iter_next(as_iterator(train_ds))
str(batch)
.[x, y, sample_weight] <- preprocessor(batch)
str(x)
str(y)
str(sample_weight)


x$token_ids |> as.array() |> _[1, 1:5]
y |> as.array() |> _[1, 1:5]


layer_linear <- new_layer_class(
  classname = "Linear",
  initialize = function(input_dim, output_dim) {
    super$initialize()
    self$kernel <- self$add_weight(shape = shape(input_dim, output_dim))
  },
  call = function(inputs) {
    op_matmul(inputs, self$kernel)
  }
)


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
    frozen <- inputs |> op_matmul(self$kernel)
    update <- inputs |> op_matmul(self$alpha) |> op_matmul(self$beta)
    frozen + update
  }
)


rank <- 2L
gemma_lm$backbone$trainable <- FALSE                                            # <1>

for (i in seq_len(gemma_lm$backbone$num_layers) - 1) {                          # <3>
  layer <- get_layer(gemma_lm$backbone, sprintf("decoder_block_%d", i))         # <2>

  layer$attention$key_dense$trainable <- TRUE
  layer$attention$key_dense$enable_lora(rank = rank)

  layer$attention$query_dense$trainable <- TRUE
  layer$attention$query_dense$enable_lora(rank = rank)
}


gemma_lm


gemma_lm |> compile(
  loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
  optimizer = optimizer_adam(5e-5),
  weighted_metrics = metric_sparse_categorical_accuracy()
)


rm(list = ls()); gc(); reticulate::import("gc")$collect()


image_url <- paste0("https://github.com/mattdangerw/keras-nlp-scripts/",
                    "blob/main/learned-python.png?raw=true")
image_path <- get_file(origin = image_url)

image <- image_path |> image_load() |> image_to_array()
par(mar = c(0, 0, 0, 0))
plot(as.raster(image, max = 255L))


library(keras3)
config_set_dtype_policy("float16")
keras_hub <- import("keras_hub")
pali_gemma_lm <- keras_hub$models$PaliGemmaCausalLM$from_preset(
  "pali_gemma_3b_mix_448"
)


pali_gemma_lm


pali_gemma_lm$generate(list(
  images = image,
  prompts = "cap en\n"
))
pali_gemma_lm$generate(list(
  images = image,
  prompts = "answer en where is the snake doing?\n"
))
pali_gemma_lm$generate(list(
  images = image,
  prompts = "detect glasses\n"
))


library(stringr)

response <- "<loc0280><loc0371><loc0380><loc0685> glasses"
box <- as.numeric(unlist(str_extract_all(response, "\\d+")))

.[height, ..] <- dim(image)
box <- box * height / 1024

.[ytop, xleft, ybottom, xright] <- box
ytop    <- height - ytop
ybottom <- height - ybottom

par(mar = c(0,0,0,0))
plot(as.raster(image, max = 255))
rect(xleft, ybottom, xright, ytop,
     border = "red", lwd = 4)

