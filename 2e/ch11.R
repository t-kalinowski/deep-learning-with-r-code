## ----setup, include = FALSE-----------------------------------------------
library(keras3)
tensorflow::as_tensor(1)


## ---- eval = FALSE--------------------------------------------------------
## vocabulary <- character()
## for (string in text_dataset) {
##   tokens <- string %>%
##     standardize() %>%
##     tokenize()
##   vocabulary <- unique(c(vocabulary, tokens))
## }


## -------------------------------------------------------------------------
one_hot_encode_token <- function(token) {
  vector <- array(0, dim = length(vocabulary))
  token_index <- match(token, vocabulary)
  vector[token_index] <- 1
  vector
}


## -------------------------------------------------------------------------
rbind(c(5,  7, 124, 4, 89),
      c(8, 34,  21, 0,  0))


## -------------------------------------------------------------------------
new_vectorizer <- function() {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "Vectorizer"

  self$vocabulary <- c("[UNK]")

  self$standardize <- function(text) {
    text <- tolower(text)
    gsub("[[:punct:]]", "", text)
  }

  self$tokenize <- function(text) {
    unlist(strsplit(text, "[[:space:]]+"))
  }

  self$make_vocabulary <- function(text_dataset) {
    tokens <- text_dataset %>%
      self$standardize() %>%
      self$tokenize()
    self$vocabulary <- unique(c(self$vocabulary, tokens))
  }

  self$encode <- function(text) {
    tokens <- text %>%
      self$standardize() %>%
      self$tokenize()
    match(tokens, table = self$vocabulary, nomatch = 1)
  }

  self$decode <- function(int_sequence) {
    vocab_w_mask_token <- c("", self$vocabulary)
    vocab_w_mask_token[int_sequence + 1]
  }

  self
}

vectorizer <- new_vectorizer()

dataset <- c(
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms."
)

vectorizer$make_vocabulary(dataset)


## -------------------------------------------------------------------------
test_sentence <- "I write, rewrite, and still rewrite again"
encoded_sentence <- vectorizer$encode(test_sentence)
print(encoded_sentence)
decoded_sentence <- vectorizer$decode(encoded_sentence)
print(decoded_sentence)


## -------------------------------------------------------------------------
text_vectorization <- layer_text_vectorization(output_mode = "int")


## -------------------------------------------------------------------------
library(tensorflow)
custom_standardization_fn <- function(string_tensor) {
  string_tensor %>%
    tf$strings$lower() %>%
    tf$strings$regex_replace("[[:punct:]]", "")
}

custom_split_fn <- function(string_tensor) {
  tf$strings$split(string_tensor)
}

text_vectorization <- layer_text_vectorization(
  output_mode = "int",
  standardize = custom_standardization_fn,
  split = custom_split_fn
)


## -------------------------------------------------------------------------
dataset <- c("I write, erase, rewrite",
             "Erase again, and then",
             "A poppy blooms.")
adapt(text_vectorization, dataset)


## -------------------------------------------------------------------------
get_vocabulary(text_vectorization)


## -------------------------------------------------------------------------
vocabulary <- text_vectorization %>% get_vocabulary()
test_sentence <- "I write, rewrite, and still rewrite again"
encoded_sentence <- text_vectorization(test_sentence)
decoded_sentence <- paste(vocabulary[as.integer(encoded_sentence) + 1],
                          collapse = " ")

encoded_sentence
decoded_sentence


## ---- eval = FALSE--------------------------------------------------------
## int_sequence_dataset <- string_dataset %>%
##   dataset_map(text_vectorization, num_parallel_calls = 4)


## ---- eval = FALSE--------------------------------------------------------
## text_input <- layer_input(shape = shape(), dtype = "string")
## vectorized_text <- text_vectorization(text_input)
## embedded_input <- vectorized_text %>% layer_embedding(...)
## output <- embedded_input %>% ...
## model <- keras_model(text_input, output)


## ---- eval = FALSE--------------------------------------------------------
## url <- "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
## filename <- basename(url)
## options(timeout = 60*10)
## download.file(url, destfile = filename)
## untar(filename)


## ---- eval = TRUE, include = FALSE----------------------------------------
unlink("aclImdb", recursive = TRUE)
untar("aclImdb_v1.tar.gz")


## -------------------------------------------------------------------------
fs::dir_tree("aclImdb", recurse = 1)


## ---- eval = TRUE---------------------------------------------------------
fs::dir_delete("aclImdb/train/unsup/")


## -------------------------------------------------------------------------
writeLines(readLines("aclImdb/train/pos/4077_10.txt", warn = FALSE))


## ---- eval = TRUE---------------------------------------------------------
library(fs)
set.seed(1337)
base_dir <- path("aclImdb")

for (category in c("neg", "pos")) {
  filepaths <- dir_ls(base_dir / "train" / category)
  num_val_samples <- round(0.2 * length(filepaths))
  val_files <- sample(filepaths, num_val_samples)

  dir_create(base_dir / "val" / category)
  file_move(val_files,
            base_dir / "val" / category)
}


## -------------------------------------------------------------------------
library(keras3)
library(tfdatasets, exclude = "shape")

train_ds <- text_dataset_from_directory("aclImdb/train")
val_ds <- text_dataset_from_directory("aclImdb/val")
test_ds <- text_dataset_from_directory("aclImdb/test")


## -------------------------------------------------------------------------
c(inputs, targets) %<-% iter_next(as_iterator(train_ds))
str(inputs)
str(targets)

inputs[1]
targets[1]


## -------------------------------------------------------------------------
c("cat", "mat", "on", "sat", "the")


## -------------------------------------------------------------------------
text_vectorization <-
  layer_text_vectorization(max_tokens = 20000,
                           output_mode = "multi_hot")

text_only_train_ds <- train_ds %>%
  dataset_map(function(x, y) x)

adapt(text_vectorization, text_only_train_ds)

binary_1gram_train_ds <- train_ds %>%
  dataset_map( ~ list(text_vectorization(.x), .y),
               num_parallel_calls = 4)
binary_1gram_val_ds <- val_ds %>%
  dataset_map( ~ list(text_vectorization(.x), .y),
               num_parallel_calls = 4)
binary_1gram_test_ds <- test_ds %>%
  dataset_map( ~ list(text_vectorization(.x), .y),
               num_parallel_calls = 4)


## -------------------------------------------------------------------------
c(inputs, targets) %<-% iter_next(as_iterator(binary_1gram_train_ds))
str(inputs)
str(targets)
inputs[1, ]
targets[1]


## -------------------------------------------------------------------------
get_model <- function(max_tokens = 20000, hidden_dim = 16) {
  inputs <- layer_input(shape = c(max_tokens))
  outputs <- inputs %>%
    layer_dense(hidden_dim, activation = "relu") %>%
    layer_dropout(0.5) %>%
    layer_dense(1, activation = "sigmoid")
  model <- keras_model(inputs, outputs)
  model %>% compile(optimizer = "rmsprop",
                    loss = "binary_crossentropy",
                    metrics = "accuracy")
  model
}


## -------------------------------------------------------------------------
model <- get_model()
model
callbacks = list(
  callback_model_checkpoint("binary_1gram.keras", save_best_only = TRUE)
)


## -------------------------------------------------------------------------
model %>% fit(
  dataset_cache(binary_1gram_train_ds),
  validation_data = dataset_cache(binary_1gram_val_ds),
  epochs = 10,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
model <- load_model("binary_1gram.keras")
cat(sprintf(
  "Test acc: %.3f\n", evaluate(model, binary_1gram_test_ds)["accuracy"]))


## -------------------------------------------------------------------------
c("the", "the cat", "cat", "cat sat", "sat",
 "sat on", "on", "on the", "the mat", "mat")


## -------------------------------------------------------------------------
text_vectorization <-
  layer_text_vectorization(ngrams = 2,
                           max_tokens = 20000,
                           output_mode = "multi_hot")


## -------------------------------------------------------------------------
adapt(text_vectorization, text_only_train_ds)

dataset_vectorize <- function(dataset) {
  dataset %>%
    dataset_map(~ list(text_vectorization(.x), .y),
                num_parallel_calls = 4)
}

binary_2gram_train_ds <- train_ds %>% dataset_vectorize()
binary_2gram_val_ds <- val_ds %>% dataset_vectorize()
binary_2gram_test_ds <- test_ds %>% dataset_vectorize()

model <- get_model()
model
callbacks = list(callback_model_checkpoint("binary_2gram.keras",
                                           save_best_only = TRUE))


## -------------------------------------------------------------------------
model %>% fit(
  dataset_cache(binary_2gram_train_ds),
  validation_data = dataset_cache(binary_2gram_val_ds),
  epochs = 10,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
model <- load_model("binary_2gram.keras")
evaluate(model, binary_2gram_test_ds)["accuracy"] %>%
  sprintf("Test acc: %.3f\n", .) %>% cat()


## -------------------------------------------------------------------------
c("the" = 2, "the cat" = 1, "cat" = 1, "cat sat" = 1, "sat" = 1,
  "sat on" = 1, "on" = 1, "on the" = 1, "the mat" = 1, "mat" = 1)


## -------------------------------------------------------------------------
text_vectorization <-
  layer_text_vectorization(ngrams = 2,
                           max_tokens = 20000,
                           output_mode = "count")


## -------------------------------------------------------------------------
tf_idf <- function(term, document, dataset) {
  term_freq <- sum(document == term)
  doc_freqs <- sapply(dataset, function(doc) sum(doc == term))
  doc_freq <- log(1 + sum(doc_freqs))
  term_freq / doc_freq
}


## -------------------------------------------------------------------------
text_vectorization <-
  layer_text_vectorization(ngrams = 2,
                           max_tokens = 20000,
                           output_mode = "tf_idf")


## -------------------------------------------------------------------------
with(tf$device("CPU"), {
  adapt(text_vectorization, text_only_train_ds)
})

tfidf_2gram_train_ds <- train_ds %>% dataset_vectorize()
tfidf_2gram_val_ds <- val_ds %>% dataset_vectorize()
tfidf_2gram_test_ds <- test_ds %>% dataset_vectorize()

model <- get_model()
model
callbacks <- list(callback_model_checkpoint("tfidf_2gram.keras",
                                            save_best_only = TRUE))


## -------------------------------------------------------------------------
model %>% fit(
  dataset_cache(tfidf_2gram_train_ds),
  validation_data = dataset_cache(tfidf_2gram_val_ds),
  epochs = 10,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
model <- load_model("tfidf_2gram.keras")
evaluate(model, tfidf_2gram_test_ds)["accuracy"] %>%
  sprintf("Test acc: %.3f", .) %>% cat("\n")



## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(1), dtype = "string")
outputs <- inputs %>%
  text_vectorization() %>%
  model()
inference_model <- keras_model(inputs, outputs)


## -------------------------------------------------------------------------
raw_text_data <- "That was an excellent movie, I loved it." %>%
  as_tensor(shape = c(-1, 1))

predictions <- inference_model(raw_text_data)
str(predictions)
cat(sprintf("%.2f percent positive\n",
            as.numeric(predictions) * 100))


## -------------------------------------------------------------------------
max_length <- 600
max_tokens <- 20000

text_vectorization <- layer_text_vectorization(
  max_tokens = max_tokens,
  output_mode = "int",
  output_sequence_length = max_length
)

adapt(text_vectorization, text_only_train_ds)

int_train_ds <- train_ds %>% dataset_vectorize()
int_val_ds <- val_ds %>% dataset_vectorize()
int_test_ds <- test_ds %>% dataset_vectorize()


## -------------------------------------------------------------------------
inputs  <- layer_input(shape(NULL), dtype = "int64")
embedded <- tf$one_hot(inputs, depth = as.integer(max_tokens))
outputs <- embedded %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dropout(.5) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model


## -------------------------------------------------------------------------
callbacks <- list(
  callback_model_checkpoint("one_hot_bidir_lstm.keras",
                            save_best_only = TRUE))


## -------------------------------------------------------------------------
int_train_ds_smaller <- int_train_ds %>%
  dataset_unbatch() %>%
  dataset_batch(16)


## -------------------------------------------------------------------------
model %>% fit(int_train_ds_smaller, validation_data = int_val_ds,
              epochs = 10, callbacks = callbacks)


## -------------------------------------------------------------------------
model <- load_model("one_hot_bidir_lstm.keras")
sprintf("Test acc: %.3f", evaluate(model, int_test_ds)["accuracy"])


## -------------------------------------------------------------------------
embedding_layer <- layer_embedding(input_dim = max_tokens, output_dim = 256)


## -------------------------------------------------------------------------
inputs <- layer_input(shape(NA), dtype = "int64")
embedded <- inputs %>%
  layer_embedding(input_dim = max_tokens, output_dim = 256)
outputs <- embedded %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>%
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = "accuracy")
model

callbacks = list(callback_model_checkpoint("embeddings_bidir_lstm.keras",
                                           save_best_only = TRUE))


## -------------------------------------------------------------------------
model %>%
  fit(int_train_ds,
      validation_data = int_val_ds,
      epochs = 10,
      callbacks = callbacks)


## -------------------------------------------------------------------------
model <- load_model("embeddings_bidir_lstm.keras")
evaluate(model, int_test_ds)["accuracy"] %>%
  sprintf("Test acc: %.3f\n", .) %>% cat("\n")


## -------------------------------------------------------------------------
embedding_layer <- layer_embedding(input_dim = 10, output_dim = 256,
                                   mask_zero = TRUE)
some_input <- rbind(c(4, 3, 2, 1, 0, 0, 0),
                    c(5, 4, 3, 2, 1, 0, 0),
                    c(2, 1, 0, 0, 0, 0, 0))
mask <- embedding_layer$compute_mask(some_input)
mask


## -------------------------------------------------------------------------
inputs <- layer_input(c(NA), dtype = "int64")
embedded <- inputs %>%
  layer_embedding(input_dim = max_tokens,
                  output_dim = 256,
                  mask_zero = TRUE)

outputs <- embedded %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model

callbacks = list(
  callback_model_checkpoint("embeddings_bidir_lstm_with_masking.keras",
                            save_best_only = TRUE)
)


## -------------------------------------------------------------------------
model %>% fit(
  int_train_ds,
  validation_data = int_val_ds,
  epochs = 10,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
model <- load_model("embeddings_bidir_lstm_with_masking.keras")
cat(sprintf("Test acc: %.3f\n",
            evaluate(model, int_test_ds)["accuracy"]))


## ---- eval = FALSE--------------------------------------------------------
## download.file("http://nlp.stanford.edu/data/glove.6B.zip",
##               destfile = "glove.6B.zip")
## zip::unzip("glove.6B.zip")


## -------------------------------------------------------------------------
path_to_glove_file <- "glove.6B.100d.txt"
embedding_dim <- 100

df <- readr::read_table(
  path_to_glove_file,
  col_names = FALSE,
  col_types = paste0("c", strrep("n", 100))
)
embeddings_index <- as.matrix(df[, -1])
rownames(embeddings_index) <- df[[1]]
colnames(embeddings_index) <- NULL
rm(df)


## -------------------------------------------------------------------------
str(embeddings_index)


## -------------------------------------------------------------------------
vocabulary <- text_vectorization %>% get_vocabulary()
str(vocabulary)

tokens <- head(vocabulary[-1], max_tokens)

i <- match(vocabulary, rownames(embeddings_index),
           nomatch = 0)

embedding_matrix <- array(0, dim = c(max_tokens, embedding_dim))
embedding_matrix[i != 0, ] <- embeddings_index[i, ]
str(embedding_matrix)


## -------------------------------------------------------------------------
embedding_layer <- layer_embedding(
  input_dim = max_tokens,
  output_dim = embedding_dim,
  embeddings_initializer = initializer_constant(embedding_matrix),
  trainable = FALSE,
  mask_zero = TRUE
)


## -------------------------------------------------------------------------
inputs <- layer_input(shape(NA), dtype="int64")
embedded <- embedding_layer(inputs)
outputs <- embedded %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model


## -------------------------------------------------------------------------
callbacks <- list(
  callback_model_checkpoint("glove_embeddings_sequence_model.keras",
                            save_best_only = TRUE)
)
model %>%
  fit(int_train_ds, validation_data = int_val_ds,
      epochs = 10, callbacks = callbacks)


## -------------------------------------------------------------------------
model <- load_model("glove_embeddings_sequence_model.keras")
cat(sprintf(
  "Test acc: %.3f\n", evaluate(model, int_test_ds)["accuracy"]))


## -------------------------------------------------------------------------
self_attention <- function(input_sequence) {
  c(sequence_len, embedding_size) %<-% dim(input_sequence)

  output <- array(0, dim(input_sequence))

  for (i in 1:sequence_len) {

    pivot_vector <- input_sequence[i, ]

    scores <- sapply(1:sequence_len, function(j)
      pivot_vector %*% input_sequence[j, ])

    scores <- softmax(scores / sqrt(embedding_size))

    broadcast_scores <- as.matrix(scores)[,rep(1, embedding_size)]

    new_pivot_representation <- colSums(input_sequence * broadcast_scores)

    output[i, ] <- new_pivot_representation
  }

  output
}

softmax <- function(x) {
   e <- exp(x - max(x))
   e / sum(e)
}


## ---- include = FALSE-----------------------------------------------------
sequence_length <- 20
embed_dim <- 256
inputs <- layer_input(c(sequence_length, embed_dim))


## -------------------------------------------------------------------------
num_heads <- 4
embed_dim <- 256

mha_layer <- layer_multi_head_attention(num_heads = num_heads,
                                        key_dim = embed_dim)
outputs <- mha_layer(inputs, inputs, inputs)


## -------------------------------------------------------------------------
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


## ---- eval = FALSE--------------------------------------------------------
## x %>% { fn(., .) + . }


## ---- eval = FALSE--------------------------------------------------------
## fn(x, x) + x


## ---- eval = FALSE--------------------------------------------------------
##   call = function(inputs, mask = NULL) {
##     if (!is.null(mask))
##       mask <- mask[, tf$newaxis, ]
##     attention_output <- self$attention(inputs, inputs,
##                                        attention_mask = mask)
##     proj_input <- self$layernorm_1(inputs + attention_output)
##     proj_output <- self$dense_proj(proj_input)
##     self$layernorm_2(proj_input + proj_output)
##   }


## ---- eval = FALSE--------------------------------------------------------
## config <- layer$get_config()
## new_layer <- do.call(layer_<type>, config)


## -------------------------------------------------------------------------
layer <- layer_dense(units = 10)
config <- layer$get_config()
new_layer <- do.call(layer_dense, config)


## -------------------------------------------------------------------------
layer$`__class__`
new_layer <- layer$`__class__`$from_config(config)


## -------------------------------------------------------------------------
layer <- layer_transformer_encoder(embed_dim = 256, dense_dim = 32,
                                   num_heads = 2)
config <- layer$get_config()
new_layer <- do.call(layer_transformer_encoder, config)
# -- or --
new_layer <- layer$`__class__`$from_config(config)


## ---- include=FALSE-------------------------------------------------------
filename <- tempfile(fileext = ".keras")


## -------------------------------------------------------------------------
model <- save_model_tf(model, filename)
model <- load_model(filename,
                       custom_objects = list(layer_transformer_encoder))


## -------------------------------------------------------------------------
model <- load_model(
  filename,
  custom_objects = list(TransformerEncoder = layer_transformer_encoder))


## -------------------------------------------------------------------------
layer_normalization <- function(batch_of_sequences) {
  c(batch_size, sequence_length, embedding_dim) %<-% dim(batch_of_sequences)
  means <- variances <-
    array(0, dim = dim(batch_of_sequences))
  for (b in seq(batch_size))
    for (s in seq(sequence_length)) {
      embedding <- batch_of_sequences[b, s, ]
      means[b, s, ] <- mean(embedding)
      variances[b, s, ] <- var(embedding)
    }
  (batch_of_sequences - means) / variances
}


## -------------------------------------------------------------------------
batch_normalization <- function(batch_of_images) {
  c(batch_size, height, width, channels) %<-% dim(batch_of_images)
  means <- variances <-
    array(0, dim = dim(batch_of_images))
  for (ch in seq(channels)) {
    channel <- batch_of_images[, , , ch]
    means[, , , ch] <- mean(channel)
    variances[, , , ch] <- var(channel)
  }
  (batch_of_images - means) / variances
}


## -------------------------------------------------------------------------
vocab_size <- 20000
embed_dim <- 256
num_heads <- 2
dense_dim <- 32

inputs <- layer_input(shape(NA), dtype = "int64")
outputs <- inputs %>%
  layer_embedding(vocab_size, embed_dim) %>%
  layer_transformer_encoder(embed_dim, dense_dim, num_heads) %>%
  layer_global_average_pooling_1d() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")
model <-  keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model


## -------------------------------------------------------------------------
callbacks = list(callback_model_checkpoint("transformer_encoder.keras",
                                           save_best_only = TRUE))
model %>% fit(
  int_train_ds,
  validation_data = int_val_ds,
  epochs = 20,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
model <- load_model("transformer_encoder.keras",
                       custom_objects = layer_transformer_encoder)

sprintf("Test acc: %.3f", evaluate(model, int_test_ds)["accuracy"])


## -------------------------------------------------------------------------
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
vocab_size <- 20000
sequence_length <- 600
embed_dim <- 256
num_heads <- 2
dense_dim <- 32

inputs <- layer_input(shape(NULL), dtype = "int64")

outputs <- inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  layer_transformer_encoder(embed_dim, dense_dim, num_heads) %>%
  layer_global_average_pooling_1d() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model <-
  keras_model(inputs, outputs) %>%
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = "accuracy")

model


## -------------------------------------------------------------------------
callbacks <- list(
  callback_model_checkpoint("full_transformer_encoder.keras",
                            save_best_only = TRUE)
)

model %>% fit(
  int_train_ds,
  validation_data = int_val_ds,
  epochs = 20,
  callbacks = callbacks
)


## -------------------------------------------------------------------------
model <- load_model(
  "full_transformer_encoder.keras",
  custom_objects = list(layer_transformer_encoder,
                        layer_positional_embedding))

cat(sprintf(
  "Test acc: %.3f\n", evaluate(model, int_test_ds)["accuracy"]))


## ---- eval = FALSE--------------------------------------------------------
## download.file("http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
##               destfile = "spa-eng.zip")
## zip::unzip("spa-eng.zip")


## -------------------------------------------------------------------------
text_file <- "spa-eng/spa.txt"
text_pairs <- text_file %>%
  readr::read_tsv(col_names = c("english", "spanish"),
                  col_types = c("cc")) %>%
  within(spanish %<>% paste("[start]", ., "[end]"))


## ---- include = FALSE-----------------------------------------------------
set.seed(1)


## -------------------------------------------------------------------------
str(text_pairs[sample(nrow(text_pairs), 1), ])


## -------------------------------------------------------------------------
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


## -------------------------------------------------------------------------
punctuation_regex <- "[^[:^punct:][\\]]|[¡¿]"

library(tensorflow)
custom_standardization <- function(input_string) {
  input_string %>%
    tf$strings$lower() %>%
    tf$strings$regex_replace(punctuation_regex, "")
}

input_string <- as_tensor("[start] ¡corre! [end]")
custom_standardization(input_string)


## -------------------------------------------------------------------------
vocab_size <- 15000
sequence_length <- 20

source_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length
)

target_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length + 1,
  standardize = custom_standardization
)

adapt(source_vectorization, train_pairs$english)
adapt(target_vectorization, train_pairs$spanish)


## -------------------------------------------------------------------------
format_pair <- function(pair) {
  eng <- source_vectorization(pair$english)
  spa <- target_vectorization(pair$spanish)

  inputs <- list(english = eng,
                 spanish = spa[NA:-2])
  targets <- spa[2:NA]
  list(inputs, targets)
}


batch_size <- 64

library(tfdatasets, exclude = "shape")
make_dataset <- function(pairs) {
  tensor_slices_dataset(pairs) %>%
    dataset_map(format_pair, num_parallel_calls = 4) %>%
    dataset_cache() %>%
    dataset_shuffle(2048) %>%
    dataset_batch(batch_size) %>%
    dataset_prefetch(16)
}
train_ds <-  make_dataset(train_pairs)
val_ds <- make_dataset(val_pairs)


## -------------------------------------------------------------------------
c(inputs, targets) %<-% iter_next(as_iterator(train_ds))
str(inputs)
str(targets)


## -------------------------------------------------------------------------
inputs <- layer_input(shape = c(sequence_length), dtype = "int64")
outputs <- inputs %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128) %>%
  layer_lstm(32, return_sequences = TRUE) %>%
  layer_dense(vocab_size, activation = "softmax")
model <- keras_model(inputs, outputs)


## -------------------------------------------------------------------------
embed_dim <- 256
latent_dim <- 1024

source <- layer_input(c(NA), dtype = "int64", name = "english")
encoded_source <- source %>%
  layer_embedding(vocab_size, embed_dim, mask_zero = TRUE) %>%
  bidirectional(layer_gru(units = latent_dim), merge_mode = "sum")


## -------------------------------------------------------------------------
decoder_gru <- layer_gru(units = latent_dim, return_sequences = TRUE)

past_target <- layer_input(shape = c(NA), dtype = "int64", name = "spanish")
target_next_step <- past_target %>%
  layer_embedding(vocab_size, embed_dim, mask_zero = TRUE) %>%
  decoder_gru(initial_state = encoded_source) %>%
  layer_dropout(0.5) %>%
  layer_dense(vocab_size, activation = "softmax")
seq2seq_rnn <- keras_model(inputs = list(source, past_target),
                           outputs = target_next_step)


## -------------------------------------------------------------------------
seq2seq_rnn %>% compile(optimizer = "rmsprop",
                        loss = "sparse_categorical_crossentropy",
                        metrics = "accuracy")


## -------------------------------------------------------------------------
seq2seq_rnn %>% fit(train_ds, epochs = 15, validation_data = val_ds)


## -------------------------------------------------------------------------
spa_vocab <- get_vocabulary(target_vectorization)
max_decoded_sentence_length <- 20

decode_sequence <- function(input_sentence) {
  tokenized_input_sentence <-
    source_vectorization(array(input_sentence, dim = c(1, 1)))
  decoded_sentence <- "[start]"
  for (i in seq(max_decoded_sentence_length)) {
    tokenized_target_sentence <-
      target_vectorization(array(decoded_sentence, dim = c(1, 1)))
    next_token_predictions <- seq2seq_rnn %>%
      predict(list(tokenized_input_sentence,
                   tokenized_target_sentence))
    sampled_token_index <- which.max(next_token_predictions[1, i, ])
    sampled_token <- spa_vocab[sampled_token_index]
    decoded_sentence <- paste(decoded_sentence, sampled_token)
    if (sampled_token == "[end]")
      break
  }
  decoded_sentence
}


## -------------------------------------------------------------------------
for (i in seq(20)) {
    input_sentence <- sample(test_pairs$english, 1)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
}


## -------------------------------------------------------------------------
tf_decode_sequence <- tf_function(function(input_sentence) {

  withr::local_options(tensorflow.extract.style = "python")

  tokenized_input_sentence <- input_sentence %>%
    as_tensor(shape = c(1, 1)) %>%
    source_vectorization()

  spa_vocab <- as_tensor(spa_vocab)

  decoded_sentence <- as_tensor("[start]", shape = c(1, 1))

  for (i in tf$range(as.integer(max_decoded_sentence_length))) {

    tokenized_target_sentence <- decoded_sentence %>%
      target_vectorization()

    next_token_predictions <-
      seq2seq_rnn(list(tokenized_input_sentence,
                       tokenized_target_sentence))

    sampled_token_index <- tf$argmax(next_token_predictions[0, i, ])
    sampled_token <- spa_vocab[sampled_token_index]
    decoded_sentence <-
      tf$strings$join(c(decoded_sentence, sampled_token),
                      separator = " ")

    if (sampled_token == "[end]")
      break
  }

  decoded_sentence

})


## -------------------------------------------------------------------------
for (i in seq(20)) {
    input_sentence <- sample(test_pairs$english, 1)
    cat("-\n")
    cat(input_sentence, "\n")
    cat(input_sentence %>% as_tensor() %>%
          tf_decode_sequence() %>% as.character(), "\n")
}


## -------------------------------------------------------------------------
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


## -------------------------------------------------------------------------
embed_dim <- 256
dense_dim <- 2048
num_heads <- 8

encoder_inputs <- layer_input(shape(NA), dtype = "int64", name = "english")
encoder_outputs <- encoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  layer_transformer_encoder(embed_dim, dense_dim, num_heads)

transformer_decoder <-
  layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)

decoder_inputs <-  layer_input(shape(NA), dtype = "int64", name = "spanish")
decoder_outputs <- decoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  transformer_decoder(., encoder_outputs) %>%
  layer_dropout(0.5) %>%
  layer_dense(vocab_size, activation="softmax")

transformer <- keras_model(list(encoder_inputs, decoder_inputs),
                           decoder_outputs)


## -------------------------------------------------------------------------
transformer %>%
  compile(optimizer = "rmsprop",
          loss = "sparse_categorical_crossentropy",
          metrics = "accuracy")

transformer %>%
  fit(train_ds, epochs = 30, validation_data = val_ds)




## ---- eval=FALSE, include=FALSE-------------------------------------------
## transformer <- load_model(
##   "end_to_end_transformer.keras",
##   custom_objects = list(
##     layer_positional_embedding,
##     layer_transformer_decoder,
##     layer_transformer_encoder
##   )
## )


## -------------------------------------------------------------------------
tf_decode_sequence <- tf_function(function(input_sentence) {
  withr::local_options(tensorflow.extract.style = "python")

  tokenized_input_sentence <- input_sentence %>%
    as_tensor(shape = c(1, 1)) %>%
    source_vectorization()
  spa_vocab <- as_tensor(spa_vocab)
  decoded_sentence <- as_tensor("[start]", shape = c(1, 1))

  for (i in tf$range(as.integer(max_decoded_sentence_length))) {

    tokenized_target_sentence <-
      target_vectorization(decoded_sentence)[,NA:-1]

    next_token_predictions <-
      transformer(list(tokenized_input_sentence,
                       tokenized_target_sentence))

    sampled_token_index <- tf$argmax(next_token_predictions[0, i, ])
    sampled_token <- spa_vocab[sampled_token_index]
    decoded_sentence <-
      tf$strings$join(c(decoded_sentence, sampled_token),
                      separator = " ")

    if (sampled_token == "[end]")
      break
  }

  decoded_sentence

})

for (i in seq(20)) {

    c(input_sentence, correct_translation) %<-%
      test_pairs[sample.int(nrow(test_pairs), 1), ]
    cat("-\n")
    cat(input_sentence, "\n")
    cat(correct_translation, "\n")
    cat(input_sentence %>% as_tensor() %>%
          tf_decode_sequence() %>% as.character(), "\n")
}
