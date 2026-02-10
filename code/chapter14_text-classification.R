# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "dplyr", "duckplyr", "envir", "fs", "glue", "stringr", "tfdatasets")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
Sys.setenv("KERAS_BACKEND"="jax")
library(stringr)
library(glue)
library(dplyr)
library(tfdatasets, exclude = "shape")
library(keras3)
library(duckplyr)
db_exec("SET enable_progress_bar TO false")


# ----------------------------------------------------------------------
text <- "The quick brown fox jumped over the lazy dog."


# ----------------------------------------------------------------------
library(stringr)
library(glue)
library(dplyr)
library(keras3)


# ----------------------------------------------------------------------
split_chars <- function(text) {
  unlist(str_split(text, boundary("character")))
}


# ----------------------------------------------------------------------
"The quick brown fox jumped over the lazy dog." |>
  split_chars() |> head(12)


# ----------------------------------------------------------------------
split_words <- function(text) {
  text |>
    str_split(boundary("word", skip_word_none = FALSE)) |>
    unlist() |>
    str_subset("\\S")
}


# ----------------------------------------------------------------------
split_words("The quick brown fox jumped over the dog.")


# ----------------------------------------------------------------------
vocabulary <- c("the", "quick", "brown", "fox", "jumped", "over", "dog", ".")
words <- split_words("The quick brown fox jumped over the lazy dog.")
indices <- match(words, vocabulary, nomatch = 0L)
indices


# ----------------------------------------------------------------------
#| lst-cap: A basic character-level tokenizer.
new_char_tokenizer <- function(vocabulary, nomatch = "[UNK]") {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "CharTokenizer"

  self$vocabulary <- vocabulary
  self$nomatch <- nomatch

  self$standardize <- function(strings) {
    str_to_lower(strings)
  }

  self$split <- function(strings) {
    split_chars(strings)
  }

  self$index <- function(tokens) {
    match(tokens, self$vocabulary, nomatch = 0L)
  }

  self$tokenize <- function(strings) {                                          # <1>
    strings |>
      self$standardize() |>
      self$split() |>
      self$index()
  }

  self$detokenize <- function(indices) {                                        # <2>
    indices[indices == 0] <- NA
    matches <- self$vocabulary[indices]
    matches[is.na(matches)] <- self$nomatch
    matches
  }

  self
}


# ----------------------------------------------------------------------
#| lst-cap: Computing a character-level vocabulary.
compute_char_vocabulary <- function(inputs, max_size = Inf) {
  tibble(chars = split_chars(inputs)) |>
    count(chars, sort = TRUE) |>
    slice_head(n = max_size) |>
    pull(chars)
}


# ----------------------------------------------------------------------
#| lst-cap: A basic word-level tokenizer.
new_word_tokenizer <- function(vocabulary, nomatch = "[UNK]") {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "WordTokenizer"

  vocabulary; nomatch;                                                          # <1>

  self$standardize <- function(string) {
    tolower(string)
  }

  self$split <- function(inputs) {
    split_words(inputs)
  }

  self$index <- function(tokens) {
    match(tokens, vocabulary, nomatch = 0)
  }

  self$tokenize <- function(string) {                                           # <2>
    string |>
      self$standardize() |>
      self$split() |>
      self$index()
  }

  self$detokenize <- function(indices) {                                        # <3>
    indices[indices == 0] <- NA
    matches <- vocabulary[indices]
    matches[is.na(matches)] <- nomatch
    matches
  }

  self
}


# ----------------------------------------------------------------------
#| lst-cap: Computing a word-level vocabulary.
compute_word_vocabulary <- function(inputs, max_size) {
  tibble(words = split_words(inputs)) |>
    count(words, sort = TRUE) |>
    slice_head(n = max_size) |>
    pull(words)
}


# ----------------------------------------------------------------------
filename <- get_file(
  origin = "https://www.gutenberg.org/files/2701/old/moby10b.txt"
)
moby_dick <- readLines(filename)

vocabulary <- compute_char_vocabulary(moby_dick, max_size = 100)
char_tokenizer <- new_char_tokenizer(vocabulary)


# ----------------------------------------------------------------------
str(vocabulary)
head(vocabulary, 10)
tail(vocabulary, 10)


# ----------------------------------------------------------------------
vocabulary <- compute_word_vocabulary(moby_dick, max_size = 2000)
word_tokenizer <- new_word_tokenizer(vocabulary)


# ----------------------------------------------------------------------
str(vocabulary)
tail(vocabulary)
string <- "Call me Ishmael. Some years ago--never mind how long precisely."
tokenized <- word_tokenizer$tokenize(string)
str(tokenized)
tokenized |>
  word_tokenizer$detokenize() |>
  str_flatten(collapse = " ")


# ----------------------------------------------------------------------
data <- c(
  "the quick brown fox",
  "the slow brown fox",
  "the quick brown foxhound"
)


# ----------------------------------------------------------------------
count_pairs <- function(tokens) {
  tibble(left = tokens, right = lead(tokens)) |>
    count(left, right, sort = TRUE) |>
    filter(left != " " & right != " ")
}

data |> split_chars() |> count_pairs()


# ----------------------------------------------------------------------
#| results: hold
get_most_common_pair <- function(tokens) {
  count_pairs(tokens) |>
    slice_max(n, with_ties = FALSE) |>
    select(left, right)
}

merge_pair <- function(tokens, pair) {
  matches <- which(
    tokens == pair$left & lead(tokens) == pair$right
  )

  tokens[matches] <- str_c(tokens[matches], tokens[matches + 1])
  tokens <- tokens[-(matches + 1)]
  tokens
}

show_tokens <- function(prefix, tokens) {
  tokens <- str_flatten(c("", unique(unlist(tokens)), ""), collapse = "_")
  cat(prefix, ": ", tokens, "\n", sep = "")
}

tokens <- data |> split_chars()
show_tokens(0, tokens)
for (i in seq_len(9)) {
  pair <- get_most_common_pair(tokens)
  tokens <- tokens |> merge_pair(pair)
  show_tokens(i, tokens)
}


# ----------------------------------------------------------------------
#| lst-cap: Computing a byte-pair encoding vocabulary.
compute_sub_word_vocabulary <- function(dataset, vocab_size) {
  dataset <- split_chars(dataset)
  vocab <- compute_char_vocabulary(dataset)
  merges <- list()
  while (length(vocab) < vocab_size) {
    pair <- get_most_common_pair(dataset)
    nrow(pair) || break
    dataset <- dataset |> merge_pair(pair)
    new_token <- str_flatten(pair)
    merges[[length(merges) + 1]] <- pair
    vocab[[length(vocab) + 1]] <- new_token
  }
  list(vocab = vocab, merges = merges)
}


# ----------------------------------------------------------------------
# `append<-` <- c
# dataset <- data
# compute_sub_word_vocabulary <- function(dataset, vocab_size) {
#   # word_counts <- count_and_split_words(dataset)
#   dataset <- unlist(split_chars(dataset))
#   vocab <- names(sort(table(dataset), decreasing = TRUE))
#   merges <- character()
#
#   while (length(vocab) < vocab_size) {
#     counts <- count_pairs(dataset)
#     if (!numhash(counts)) break
#     .[pair, ..] <- get_most_common_pair(counts)
#     dataset <- merge_pair(dataset, pair)
#     token <- paste0(pair, collapse = "")
#     vocab <- c(vocab, token)
#     merges <- c(merges, token)
#   }
#   list(vocab = vocab, merges = merges)
# }


# ----------------------------------------------------------------------
#| lst-cap: A byte-pair encoding tokenizer.
bpe_merge <- function(data, merges) {
  sep <- "|||SEP|||"                                                            # <1>
  data <- str_flatten(data, collapse = sep)                                     # <1>
  for (pair in merges) {                                                        # <2>
    .[left, right] <- pair
    data <- data |> str_replace_all(                                            # <3>
      pattern = fixed(str_c(sep, left, sep, right, sep)),
      replacement = str_c(sep, left, right, sep)
    )
  }
  str_split_1(data, fixed(sep))                                                 # <4>
}


# ----------------------------------------------------------------------
new_subword_tokenizer <- function(vocabulary, merges, nomatch = "[UNK]") {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "SubWordTokenizer"

  vocabulary; merges; nomatch

  self$standardize <- function(string) {
    tolower(string)
  }

  self$split <- function(string) {
    string |> split_chars() |> bpe_merge(merges)
  }

  self$index <- function(tokens) {
    match(tokens, vocabulary, nomatch = 0)
  }

  self$tokenize <- function(string, nomatch = 0) {
    string |>
      self$standardize() |>
      self$split() |>
      self$index()
  }

  self$detokenize <- function(indices) {
    indices[indices == 0] <- NA
    matches <- vocabulary[indices]
    matches[is.na(matches)] <- nomatch
    matches
  }

  self
}


# ----------------------------------------------------------------------
#| eval: false
# library(stringr)
# x <- str_locate_all(moby_dick, fixed("th"))
# x
# length(unlist(str_extract_all(moby_dick, fixed("th"))))
# 
# sub_word_tokenizer = SubWordTokenizer(vocabulary, merges)


# ----------------------------------------------------------------------
if (!file.exists("ch14-bpe-vocab.rds")) {
  .[vocabulary, merges] <- compute_sub_word_vocabulary(moby_dick, 2000)           # <1>
  sub_word_tokenizer <- new_subword_tokenizer(vocabulary, merges)
}


# ----------------------------------------------------------------------
str(vocabulary)
tail(vocabulary)
tokenized <- sub_word_tokenizer$tokenize(string)
str(tokenized)
tokenized |> sub_word_tokenizer$detokenize() |> str_flatten("_")


# ----------------------------------------------------------------------
#| eval: false
# glue(r"---(
#   Vocabulary length: { length(vocabulary) }
#   Vocabulary head: { str_flatten(double_quote(head(vocabulary)), " ") }
#   Vocabulary tail: { str_flatten(double_quote(tail(vocabulary)), " ") }
#   Line length: { length(word_tokenizer$tokenize(
#     "Call me Ishmael. Some years ago--never mind how long precisely.")) }
#   )---"
# )


# ----------------------------------------------------------------------
#| lst-cap: Downloading the IMDb movie review dataset.
library(keras3)
tar_path <- get_file(
  origin = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
)


# ----------------------------------------------------------------------
untar(tar_path)


# ----------------------------------------------------------------------
fs::dir_tree("aclImdb", type = "directory")


# ----------------------------------------------------------------------
#| lst-cap: Previewing a single IMDb review.
writeLines(strwrap(readLines("aclImdb/train/pos/4229_10.txt", warn = FALSE)))


# ----------------------------------------------------------------------
#| lst-cap: Splitting validation from the IMDb dataset.
library(fs)
set.seed(1337)
base_dir <- path("aclImdb")

for (category in c("neg", "pos")) {                                             # <1>
  filepaths <- dir_ls(base_dir / "train" / category)
  num_val_samples <- round(0.2 * length(filepaths))

  val_files <- sample(filepaths, num_val_samples)

  val_dir <- base_dir / "val" / category
  dir_create(val_dir)
  file_move(val_files, val_dir)
}


# ----------------------------------------------------------------------
#| lst-cap: Loading the IMDb dataset for use with Keras.
library(tfdatasets, exclude = c("shape"))

train_ds <- text_dataset_from_directory(
  "aclImdb/train",
  class_names = c("neg", "pos")                                                 # <1>
)
val_ds <- text_dataset_from_directory("aclImdb/val")
test_ds <- text_dataset_from_directory("aclImdb/test")


# ----------------------------------------------------------------------
.[inputs, targets] <- iter_next(as_iterator(train_ds))
str(inputs)
inputs@r[1]

str(targets)
targets@r[1]


# ----------------------------------------------------------------------
#| lst-cap: Applying a bag-of-words encoding to the IMDb reviews.
max_tokens <- 20000
text_vectorization <- layer_text_vectorization(
  max_tokens = max_tokens,
  split = "whitespace",                                                         # <1>
  output_mode = "multi_hot"
)

train_ds_no_labels <- train_ds |> dataset_map(\(x, y) x)
adapt(text_vectorization, train_ds_no_labels)

bag_of_words_train_ds <- train_ds |>
  dataset_map(\(x, y) tuple(text_vectorization(x), y),
              num_parallel_calls = 8)
bag_of_words_val_ds <- val_ds |>
  dataset_map(\(x, y) tuple(text_vectorization(x), y),
              num_parallel_calls = 8)
bag_of_words_test_ds <- test_ds |>
  dataset_map(\(x, y) tuple(text_vectorization(x), y),
              num_parallel_calls = 8)


# ----------------------------------------------------------------------
.[inputs, targets] <- bag_of_words_train_ds |>
  as_array_iterator() |> iter_next()
str(inputs)
str(targets)


# ----------------------------------------------------------------------
#| lst-cap: Building a bag-of-words regression model.
build_linear_classifier <- function(max_tokens, name) {
  inputs <- keras_input(shape = c(max_tokens))
  outputs <- inputs |>
    layer_dense(1, activation = "sigmoid")
  model <- keras_model(inputs, outputs, name = name)
  model |> compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )
  model
}
model <- build_linear_classifier(max_tokens, "bag_of_words_classifier")


# ----------------------------------------------------------------------
model


# ----------------------------------------------------------------------
#| lst-cap: Training the bag-of-words regression model.
early_stopping <- callback_early_stopping(
  monitor = "val_loss",
  restore_best_weights = TRUE,
  patience = 2
)
history <- model |> fit(
  bag_of_words_train_ds,
  validation_data = bag_of_words_val_ds,
  epochs = 10,
  callbacks = c(early_stopping)
)


# ----------------------------------------------------------------------
#| fig-cap: Training and validation metrics for our bag-of-words model.
plot(history, metrics = "accuracy")


# ----------------------------------------------------------------------
#| lst-cap: Evaluating the bag-of-words regression model.
test_result <- evaluate(model, bag_of_words_test_ds)
test_result$accuracy


# ----------------------------------------------------------------------
#| lst-cap: Applying a bigram encoding to the IMDb reviews.
max_tokens <- 30000
text_vectorization <- layer_text_vectorization(
  max_tokens = max_tokens,
  split = "whitespace",                                                         # <1>
  output_mode = "multi_hot",
  ngrams = 2,                                                                   # <2>
)
adapt(text_vectorization, train_ds_no_labels)

bigram_train_ds <- train_ds |>
  dataset_map(\(x, y) tuple(text_vectorization(x), y),
              num_parallel_calls = 8)
bigram_val_ds <- val_ds |>
  dataset_map(\(x, y) tuple(text_vectorization(x), y),
              num_parallel_calls = 8)
bigram_test_ds <- test_ds |>
  dataset_map(\(x, y) tuple(text_vectorization(x), y),
              num_parallel_calls = 8)


# ----------------------------------------------------------------------
.[inputs, targets] <- bigram_train_ds |>
  as_iterator() |> iter_next()
str(inputs)
str(targets)


# ----------------------------------------------------------------------
get_vocabulary(text_vectorization)[100:108]


# ----------------------------------------------------------------------
#| lst-cap: Training the bigram regression model.
model <- build_linear_classifier(max_tokens, "bigram_classifier")
model |> fit(
  bigram_train_ds,
  validation_data = bigram_val_ds,
  epochs = 10,
  callbacks = early_stopping
)


# ----------------------------------------------------------------------
#| lst-cap: Evaluating the bigram regression model.
result <- evaluate(model, bigram_test_ds)
result$accuracy



# ----------------------------------------------------------------------
Sys.setenv("KERAS_BACKEND"="jax")
library(stringr)
library(glue)
library(dplyr)
library(tfdatasets, exclude = "shape")
library(keras3)
library(duckplyr)
library(fs)
library(tfdatasets, exclude = c("shape"))


# ----------------------------------------------------------------------
text <- "The quick brown fox jumped over the lazy dog."
vocabulary <- c("the", "quick", "brown", "fox", "jumped", "over", "dog", ".")
string <- "Call me Ishmael. Some years ago--never mind how long precisely."
max_tokens <- 20000
max_tokens <- 30000


# ----------------------------------------------------------------------
split_chars <- function(text) {
  unlist(str_split(text, boundary("character")))
}

split_words <- function(text) {
  text |>
    str_split(boundary("word", skip_word_none = FALSE)) |>
    unlist() |>
    str_subset("\\S")
}

new_char_tokenizer <- function(vocabulary, nomatch = "[UNK]") {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "CharTokenizer"

  self$vocabulary <- vocabulary
  self$nomatch <- nomatch

  self$standardize <- function(strings) {
    str_to_lower(strings)
  }

  self$split <- function(strings) {
    split_chars(strings)
  }

  self$index <- function(tokens) {
    match(tokens, self$vocabulary, nomatch = 0L)
  }

  self$tokenize <- function(strings) {                                          # <1>
    strings |>
      self$standardize() |>
      self$split() |>
      self$index()
  }

  self$detokenize <- function(indices) {                                        # <2>
    indices[indices == 0] <- NA
    matches <- self$vocabulary[indices]
    matches[is.na(matches)] <- self$nomatch
    matches
  }

  self
}

compute_char_vocabulary <- function(inputs, max_size = Inf) {
  tibble(chars = split_chars(inputs)) |>
    count(chars, sort = TRUE) |>
    slice_head(n = max_size) |>
    pull(chars)
}

new_word_tokenizer <- function(vocabulary, nomatch = "[UNK]") {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "WordTokenizer"

  vocabulary; nomatch;                                                          # <1>

  self$standardize <- function(string) {
    tolower(string)
  }

  self$split <- function(inputs) {
    split_words(inputs)
  }

  self$index <- function(tokens) {
    match(tokens, vocabulary, nomatch = 0)
  }

  self$tokenize <- function(string) {                                           # <2>
    string |>
      self$standardize() |>
      self$split() |>
      self$index()
  }

  self$detokenize <- function(indices) {                                        # <3>
    indices[indices == 0] <- NA
    matches <- vocabulary[indices]
    matches[is.na(matches)] <- nomatch
    matches
  }

  self
}

compute_word_vocabulary <- function(inputs, max_size) {
  tibble(words = split_words(inputs)) |>
    count(words, sort = TRUE) |>
    slice_head(n = max_size) |>
    pull(words)
}

count_pairs <- function(tokens) {
  tibble(left = tokens, right = lead(tokens)) |>
    count(left, right, sort = TRUE) |>
    filter(left != " " & right != " ")
}

get_most_common_pair <- function(tokens) {
  count_pairs(tokens) |>
    slice_max(n, with_ties = FALSE) |>
    select(left, right)
}

merge_pair <- function(tokens, pair) {
  matches <- which(
    tokens == pair$left & lead(tokens) == pair$right
  )

  tokens[matches] <- str_c(tokens[matches], tokens[matches + 1])
  tokens <- tokens[-(matches + 1)]
  tokens
}

show_tokens <- function(prefix, tokens) {
  tokens <- str_flatten(c("", unique(unlist(tokens)), ""), collapse = "_")
  cat(prefix, ": ", tokens, "\n", sep = "")
}

compute_sub_word_vocabulary <- function(dataset, vocab_size) {
  dataset <- split_chars(dataset)
  vocab <- compute_char_vocabulary(dataset)
  merges <- list()
  while (length(vocab) < vocab_size) {
    pair <- get_most_common_pair(dataset)
    nrow(pair) || break
    dataset <- dataset |> merge_pair(pair)
    new_token <- str_flatten(pair)
    merges[[length(merges) + 1]] <- pair
    vocab[[length(vocab) + 1]] <- new_token
  }
  list(vocab = vocab, merges = merges)
}

bpe_merge <- function(data, merges) {
  sep <- "|||SEP|||"                                                            # <1>
  data <- str_flatten(data, collapse = sep)                                     # <1>
  for (pair in merges) {                                                        # <2>
    .[left, right] <- pair
    data <- data |> str_replace_all(                                            # <3>
      pattern = fixed(str_c(sep, left, sep, right, sep)),
      replacement = str_c(sep, left, right, sep)
    )
  }
  str_split_1(data, fixed(sep))                                                 # <4>
}

new_subword_tokenizer <- function(vocabulary, merges, nomatch = "[UNK]") {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "SubWordTokenizer"

  vocabulary; merges; nomatch

  self$standardize <- function(string) {
    tolower(string)
  }

  self$split <- function(string) {
    string |> split_chars() |> bpe_merge(merges)
  }

  self$index <- function(tokens) {
    match(tokens, vocabulary, nomatch = 0)
  }

  self$tokenize <- function(string, nomatch = 0) {
    string |>
      self$standardize() |>
      self$split() |>
      self$index()
  }

  self$detokenize <- function(indices) {
    indices[indices == 0] <- NA
    matches <- vocabulary[indices]
    matches[is.na(matches)] <- nomatch
    matches
  }

  self
}

build_linear_classifier <- function(max_tokens, name) {
  inputs <- keras_input(shape = c(max_tokens))
  outputs <- inputs |>
    layer_dense(1, activation = "sigmoid")
  model <- keras_model(inputs, outputs, name = name)
  model |> compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )
  model
}


# ----------------------------------------------------------------------
library(fs)
library(tfdatasets, exclude = c("shape"))

if (!dir_exists("aclImdb/train")) {
  stop(
    "Missing directory: 'aclImdb/train'. Run the earlier section that downloads/prepares the IMDb dataset first.",
    call. = FALSE
  )
}

train_ds <- text_dataset_from_directory(
  "aclImdb/train",
  class_names = c("neg", "pos")
)
val_ds <- text_dataset_from_directory("aclImdb/val")
test_ds <- text_dataset_from_directory("aclImdb/test")

train_ds_no_labels <- train_ds |> dataset_map(\(x, y) x)


# ----------------------------------------------------------------------
#| lst-cap: Padding IMDb reviews to a fixed sequence length.
max_length <- 600
max_tokens <- 30000
text_vectorization <- layer_text_vectorization(
  max_tokens = max_tokens,
  split = "whitespace",                                                         # <1>
  output_mode = "int",                                                          # <2>
  output_sequence_length = max_length                                           # <3>
)
text_vectorization |> adapt(train_ds_no_labels)

sequence_train_ds <- train_ds |>
  dataset_map(\(x, y) tuple(text_vectorization(x), y),
              num_parallel_calls = 8)
sequence_val_ds <- val_ds |>
  dataset_map(\(x, y) tuple(text_vectorization(x), y),
              num_parallel_calls = 8)
sequence_test_ds <- test_ds |>
  dataset_map(\(x, y) tuple(text_vectorization(x), y),
              num_parallel_calls = 8)


# ----------------------------------------------------------------------
.[x, y] <- sequence_test_ds |> as_array_iterator() |> iter_next()
str(x)
tail(x, c(5, 5))


# ----------------------------------------------------------------------
#| lst-cap: Building an LSTM sequence model.
hidden_dim <- 64
inputs <- keras_input(shape = c(max_length), dtype = "int32")
outputs <- inputs |>
  op_one_hot(num_classes = max_tokens, zero_indexed = TRUE) |>
  layer_bidirectional(layer_lstm(units = hidden_dim)) |>
  layer_dropout(0.5) |>
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs, name = "lstm_with_one_hot")
model |> compile(optimizer = "adam",
                 loss = "binary_crossentropy",
                 metrics = c("accuracy"))


# ----------------------------------------------------------------------
model


# ----------------------------------------------------------------------
#| lst-cap: Training the LSTM sequence model.
model |> fit(
  sequence_train_ds,
  validation_data = sequence_val_ds,
  epochs = 10,
  callbacks = c(early_stopping)
)


# ----------------------------------------------------------------------
#| lst-cap: Evaluating the LSTM sequence model.
evaluate(model, sequence_test_ds)$accuracy


# ----------------------------------------------------------------------
acc <- evaluate(model, sequence_test_ds)$accuracy
acc
envir::import_from(scales, percent)


# ----------------------------------------------------------------------
#| lst-cap: "Building an LSTM sequence model with an `Embedding` layer."
hidden_dim <- 64L
inputs <- keras_input(shape = c(max_length), dtype = "int32")
outputs <- inputs |>
  layer_embedding(input_dim = max_tokens,
                  output_dim = hidden_dim,
                  mask_zero = TRUE) |>
  layer_bidirectional(layer_lstm(units = hidden_dim)) |>
  layer_dropout(0.5) |>
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs, name = "lstm_with_embedding")
model |> compile(optimizer = "adam",
                 loss = "binary_crossentropy",
                 metrics = "accuracy")


# ----------------------------------------------------------------------
model


# ----------------------------------------------------------------------
#| lst-cap: "Training and evaluating the LSTM with an `Embedding` layer."
model |> fit(
  sequence_train_ds,
  validation_data = sequence_val_ds,
  epochs = 10,
  callbacks = early_stopping
)
result <- evaluate(model, sequence_test_ds)
result$accuracy


# ----------------------------------------------------------------------
#| lst-cap: "Removing padding from our `TextVectorization` preprocessing layer."
imdb_vocabulary <- text_vectorization |> get_vocabulary()
tokenize_no_padding <- layer_text_vectorization(
  vocabulary = imdb_vocabulary,
  split = "whitespace",
  output_mode = "int"
)


# ----------------------------------------------------------------------
#| lst-cap: Preprocessing our IMDb data for pretraining a CBOW model.
context_size <- 4L                                                              # <1>
window_size <- context_size + 1L + context_size                                 # <2>

window_data <- function(token_ids) {
  windows <- tf$signal$frame(                                                   # <1>
    token_ids,
    frame_length = window_size,
    frame_step = 1L
  )
  tensor_slices_dataset(windows)
}

split_label <- function(window) {
  .[left, label, right] <-
    tf$split(window, c(context_size, 1L, context_size))
  bag <- tf$concat(tuple(left, right), axis = 0L)
  tuple(bag, label)
}

dataset <- text_dataset_from_directory("aclImdb/train", batch_size = NULL)      # <3>

dataset <- dataset |>
  dataset_map(\(x, y) x, num_parallel_calls = 8) |>                             # <4>
  dataset_map(tokenize_no_padding, num_parallel_calls = 8) |>                   # <5>
  dataset_interleave(window_data) |>                                            # <6>
  dataset_map(split_label, num_parallel_calls = 8)                              # <7>


# ----------------------------------------------------------------------
#| lst-cap: Building a CBOW model.
hidden_dim <- 64

cbow_embedding <- layer_embedding(
  input_dim = max_tokens,
  output_dim = hidden_dim
)

inputs <- keras_input(shape = c(2 * context_size))

outputs <- inputs |>
  cbow_embedding() |>
  layer_global_average_pooling_1d() |>
  layer_dense(max_tokens, activation = "softmax")

cbow_model <- keras_model(inputs, outputs)
cbow_model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "sparse_categorical_accuracy"
)


# ----------------------------------------------------------------------
cbow_model


# ----------------------------------------------------------------------
#| lst-cap: Training the CBOW model.
dataset <- dataset |> dataset_batch(1024) |> dataset_cache()
cbow_model |> fit(dataset, epochs = 4)


# ----------------------------------------------------------------------
#| lst-cap: "Building another LSTM sequence model with an `Embedding` layer."
inputs <- keras_input(shape = c(max_length))
lstm_embedding <- layer_embedding(
  input_dim = max_tokens,
  output_dim = hidden_dim,
  mask_zero = TRUE
)
outputs <- inputs |>
  lstm_embedding() |>
  layer_bidirectional(layer_lstm(units = hidden_dim)) |>
  layer_dropout(0.5) |>
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs, name = "lstm_with_cbow")


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Reusing the CBOW embedding to prime the LSTM model.
lstm_embedding$embeddings$assign(cbow_embedding$embeddings)


# ----------------------------------------------------------------------
#| lst-cap: Training the LSTM model with a pretrained embedding.
model |> compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)
model |> fit(
  sequence_train_ds,
  validation_data = sequence_val_ds,
  epochs = 10,
  callbacks = c(early_stopping)
)


# ----------------------------------------------------------------------
#| lst-cap: Evaluating the LSTM model with a pretrained embedding.
result <- evaluate(model, sequence_test_ds)
result$accuracy


# ----------------------------------------------------------------------




