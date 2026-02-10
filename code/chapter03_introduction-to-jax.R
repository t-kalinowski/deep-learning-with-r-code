# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "lobstr")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
use_backend("jax")

jax <- import("jax")


# ----------------------------------------------------------------------
jnp <- import("jax.numpy")


# ----------------------------------------------------------------------
#| lst-cap: All-ones or all-zeros tensors
jnp$ones(shape = shape(2, 2))
jnp$zeros(shape = shape(2, 2))
jnp$array(c(1, 2, 3), dtype = "float32")


# ----------------------------------------------------------------------
#| lst-cap: Random tensors
runif(3)
runif(3)


# ----------------------------------------------------------------------
apply_noise <- function(x, seed) {
  set.seed(seed)
  x + array(runif(length(x), -1, 1))
}

x <- numeric(3)
seed <- 1337
identical(apply_noise(x, seed),
          apply_noise(x, seed))

seed <- seed + 1
identical(apply_noise(x, seed),
          apply_noise(x, seed))


# ----------------------------------------------------------------------
seed_key <- jax$random$key(1337L)


# ----------------------------------------------------------------------
seed_key <- jax$random$key(0L)
jax$random$normal(seed_key, shape = shape(3))


# ----------------------------------------------------------------------
#| lst-cap: Using a random seed in JAX.
#| results: hold
seed_key <- jax$random$key(123L)
jax$random$normal(seed_key, shape = shape(3))
jax$random$normal(seed_key, shape = shape(3))


# ----------------------------------------------------------------------
seed_key <- jax$random$key(123L)
jax$random$normal(seed_key, shape = shape(3))


# ----------------------------------------------------------------------
#| eval: false
# new_seed_key <- jax$random$split(seed_key, num = 1L)@py[0]                      # <1>
# jax$random$normal(new_seed_key, shape = shape(3))


# ----------------------------------------------------------------------
new_seed_key <- jax$random$split(seed_key, num = 1L)@py[0]                      # <1>
jax$random$normal(new_seed_key, shape = shape(3))


# ----------------------------------------------------------------------
#| lst-cap: Modifying values in a JAX array.
x <- jnp$array(c(1, 2, 3), dtype = "float32")
new_x <- x$at[0]$set(10)
new_x


# ----------------------------------------------------------------------
x@r[1] <- 20
x


# ----------------------------------------------------------------------
x <- array(1, c(2, 2))
orig_x_addr <- lobstr::obj_addr(x)
x[1, 1] <- 2
orig_x_addr == lobstr::obj_addr(x)


# ----------------------------------------------------------------------
#| lst-cap: A few basic math operations in JAX
a <- jnp$ones(shape(2, 2))
b <- jnp$square(a)                                                              # <1>
c <- jnp$sqrt(a)                                                                # <2>
d <- b + c                                                                      # <3>
e <- jnp$matmul(a, b)                                                           # <4>
e <- e * d                                                                      # <5>


# ----------------------------------------------------------------------
dense <- function(inputs, W, b) {
  jax$nn$relu(jnp$matmul(inputs, W) + b)
}


# ----------------------------------------------------------------------
compute_loss <- function(input_var) {
  jnp$square(input_var)
}


# ----------------------------------------------------------------------
grad_fn <- jax$grad(compute_loss)


# ----------------------------------------------------------------------
input_var <- jnp$array(3)
grad_of_loss_wrt_input_var <- grad_fn(input_var)


# ----------------------------------------------------------------------
grad_fn <- jax$value_and_grad(compute_loss)
.[output, grad_of_loss_wrt_input_var] <- grad_fn(input_var)


# ----------------------------------------------------------------------
#| eval: false
# compute_loss <- function(state, x, y) {                                         # <1>
#   .....
#   loss
# }
# 
# grad_fn <- jax$value_and_grad(compute_loss)
# state <- list(a, b, c)
# .[loss, grads_of_loss_wrt_state] <- grad_fn(state, x, y)                        # <2>


# ----------------------------------------------------------------------
#| eval: false
# compute_loss <- function(state, x, y) {
#   .....
#   tuple(loss, output)                                                           # <1>
# }
# 
# grad_fn <- jax$value_and_grad(compute_loss, has_aux = TRUE)                     # <2>
# .[loss, .[grads_of_loss_wrt_state, output]] <- grad_fn(state, x, y)             # <3>


# ----------------------------------------------------------------------
dense <- jax$jit(\(inputs, W, b) {
  jax$nn$relu(jnp$matmul(inputs, W) + b)
})


# ----------------------------------------------------------------------
model <- function(inputs, W, b) {
  jnp$matmul(inputs, W) + b
}

mean_squared_error <- function(targets, predictions) {
  jnp$mean(jnp$square(targets - predictions))
}


# ----------------------------------------------------------------------
learning_rate <- 0.1

compute_loss <- function(state, inputs, targets) {
  .[W, b] <- state
  predictions <- model(inputs, W, b)
  mean_squared_error(targets, predictions)
}


# ----------------------------------------------------------------------
grad_fn <- jax$value_and_grad(compute_loss)


# ----------------------------------------------------------------------
learning_rate <- 0.1

training_step <- jax$jit(\(inputs, targets, W, b) {                             # <1>
  .[loss, grads] <- grad_fn(list(W, b), inputs, targets)                        # <2>
  .[grad_wrt_W, grad_wrt_b] <- grads
  W <- W - grad_wrt_W * learning_rate                                           # <3>
  b <- b - grad_wrt_b * learning_rate                                           # <3>
  tuple(loss, W, b)                                                             # <4>
})


# ----------------------------------------------------------------------
num_samples_per_class <- 1000
Sigma <- rbind(c(1, 0.5),
               c(0.5, 1))
negative_samples <- MASS::mvrnorm(
  n = num_samples_per_class,
  mu = c(0, 3),
  Sigma = Sigma
)
positive_samples <- MASS::mvrnorm(
  n = num_samples_per_class,
  mu = c(3, 0),
  Sigma = Sigma
)
inputs <- rbind(negative_samples, positive_samples)
targets <- rbind(
  array(0, dim = c(num_samples_per_class, 1)),
  array(1, dim = c(num_samples_per_class, 1))
)


# ----------------------------------------------------------------------
input_dim <- 2
output_dim <- 1

W <- jnp$array(array(runif(input_dim * output_dim),
                     dim = c(input_dim, output_dim)))
b <- jnp$array(array(0, dim = c(output_dim)))
for (step in seq(40)) {
  .[loss, W, b] <- training_step(inputs, targets, W, b)
  if (!step %% 10)
    cat(sprintf("Loss at step %d: %.4f\n", step, as.array(loss)))
}


