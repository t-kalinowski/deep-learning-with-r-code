# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
#| message: false
library(tensorflow)
library(keras3)
use_backend("tensorflow")


# ----------------------------------------------------------------------
#| lst-cap: All-ones or all-zeros tensors
tf$ones(shape = shape(2, 2))
tf$zeros(shape = shape(2, 2))
tf$constant(c(1, 2, 3), dtype = "float32")


# ----------------------------------------------------------------------
#| eval: false
#| lst-cap: Random tensors
# tf$random$normal(shape(3, 1), mean = 0, stddev = 1)                             # <1>


# ----------------------------------------------------------------------
tf$random$normal(shape(3, 1), mean = 0, stddev = 1)                             # <1>


# ----------------------------------------------------------------------
#| eval: false
# tf$random$uniform(shape(3, 1), minval = 0, maxval = 1)                          # <1>


# ----------------------------------------------------------------------
tf$random$uniform(shape(3, 1), minval = 0, maxval = 1)                          # <1>


# ----------------------------------------------------------------------
#| lst-cap: R arrays are assignable
x <- array(1, dim = c(2, 2))
x[1, 1] <- 0


# ----------------------------------------------------------------------
#| error: true
#| eval: false
#| lst-cap: TensorFlow tensors are not assignable
# x <- tf$ones(shape(2, 2))
# x@r[1, 1] <- 0.                                                                 # <1>


# ----------------------------------------------------------------------
#| error: true
x <- tf$ones(shape(2, 2))
x@r[1, 1] <- 0.                                                                 # <1>


# ----------------------------------------------------------------------
#| lst-cap: "Creating a `tf.Variable`"
v <- tf$Variable(initial_value = tf$zeros(shape = shape(3, 1)))
v


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Assigning a value to a Variable
v$assign(tf$ones(shape(3, 1)))
v


# ----------------------------------------------------------------------
v


# ----------------------------------------------------------------------
#| lst-cap: Assigning a value to a Variable
v@r[] <- tf$zeros(shape(3, 1))
v


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Assigning a value to a subset of a Variable
v@r[1, 1]$assign(3)
v@r[2, 1] <- 4
v


# ----------------------------------------------------------------------
v


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Using assign_add
v$assign_add(tf$ones(shape(3, 1)))
v


# ----------------------------------------------------------------------
v


# ----------------------------------------------------------------------
#| lst-cap: A few basic math operations in TensorFlow
a <- tf$ones(shape(2, 2))
b <- tf$square(a)                                                               # <1>
c <- tf$sqrt(a)                                                                 # <2>
d <- b + c                                                                      # <3>
e <- tf$matmul(a, b)                                                            # <4>
f <- tf$concat(list(a, b), axis = 0L)                                           # <5>


# ----------------------------------------------------------------------
dense <- function(inputs, W, b) {
  tf$nn$relu(tf$matmul(inputs, W) + b)
}


# ----------------------------------------------------------------------
#| lst-cap: "Using the `GradientTape`"
input_var <- tf$Variable(initial_value = 3)
with(tf$GradientTape() %as% tape, {
  result <- tf$square(input_var)
})
gradient <- tape$gradient(result, input_var)


# ----------------------------------------------------------------------
#| lst-cap: "Using the `GradientTape` with constant tensor inputs"
input_const <- tf$constant(3)
with(tf$GradientTape() %as% tape, {
  tape$watch(input_const)
  result <- tf$square(input_const)
})
gradient <- tape$gradient(result, input_const)


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Using nested gradient tapes to compute second-order gradients
time <- tf$Variable(0)
with(tf$GradientTape() %as% outer_tape, {
  with(tf$GradientTape() %as% inner_tape, {
    position <- 4.9 * time^2
  })
  speed <- inner_tape$gradient(position, time)
})
acceleration <- outer_tape$gradient(speed, time)                                # <1>
acceleration


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: Using nested gradient tapes to compute second-order gradients
time <- tf$Variable(0)
with(tf$GradientTape() %as% outer_tape, {
  with(tf$GradientTape() %as% inner_tape, {
    position <- 4.9 * time^2
  })
  speed <- inner_tape$gradient(position, time)
})
acceleration <- outer_tape$gradient(speed, time)                                # <1>
acceleration


# ----------------------------------------------------------------------
dense <- tf_function(\(inputs, W, b) {
  tf$nn$relu(tf$matmul(inputs, W) + b)
})


# ----------------------------------------------------------------------
dense <- tf_function(jit_compile = TRUE, \(inputs, W, b) {
  tf$nn$relu(tf$matmul(inputs, W) + b)
})


# ----------------------------------------------------------------------
#| lst-cap: Generating two classes of random points in a 2D plane
num_samples_per_class <- 1000
Sigma <- rbind(c(1, 0.5),
               c(0.5, 1))
negative_samples <- MASS::mvrnorm(                                              # <1>
  n = num_samples_per_class,                                                    # <1>
  mu = c(0, 3),                                                                 # <1>
  Sigma = Sigma                                                                 # <1>
)                                                                               # <1>
positive_samples <- MASS::mvrnorm(                                              # <2>
  n = num_samples_per_class,                                                    # <2>
  mu = c(3, 0),                                                                 # <2>
  Sigma = Sigma                                                                 # <2>
)


# ----------------------------------------------------------------------
#| lst-cap: "Stacking the two classes into an array with shape (2000, 2)"
inputs <- rbind(negative_samples, positive_samples)


# ----------------------------------------------------------------------
#| lst-cap: Generating the corresponding targets (0 and 1)
targets <- rbind(array(0, dim = c(num_samples_per_class, 1)),
                 array(1, dim = c(num_samples_per_class, 1)))


# ----------------------------------------------------------------------
#| lst-cap: Plotting the two point classes
#| fig-cap: "Our synthetic data: two classes of random points in the 2D plane"
plot(x = inputs[, 1], y = inputs[, 2],
     col = ifelse(targets[, 1] == 0, "purple", "green"))


# ----------------------------------------------------------------------
#| lst-cap: Creating the linear classifier variables
input_dim <- 2                                                                  # <1>
output_dim <- 1                                                                 # <2>
W <- tf$Variable(
  initial_value = tf$random$uniform(shape(input_dim, output_dim))
)
b <- tf$Variable(
  initial_value = tf$zeros(shape(output_dim))
)


# ----------------------------------------------------------------------
#| lst-cap: The forward pass function
model <- function(inputs, W, b) {
  tf$matmul(inputs, W) + b
}


# ----------------------------------------------------------------------
#| lst-cap: The mean squared error loss function
mean_squared_error <- function(targets, predictions) {
  per_sample_losses <- tf$square(targets - predictions)                         # <1>
  tf$reduce_mean(per_sample_losses)                                             # <2>
}


# ----------------------------------------------------------------------
#| lst-cap: The training step function
learning_rate <- 0.1

training_step <- tf_function(                                                   # <1>
  jit_compile = TRUE,
  \(inputs, targets, W, b) {
    with(tf$GradientTape() %as% tape, {
      predictions <- model(inputs, W, b)                                        # <2>
      loss <- mean_squared_error(targets, predictions)                          # <2>
    })
    grad_loss_wrt <- tape$gradient(loss, list(W = W, b = b))                    # <3>
    W$assign_sub(grad_loss_wrt$W * learning_rate)                               # <4>
    b$assign_sub(grad_loss_wrt$b * learning_rate)                               # <4>
    loss
  }
)


# ----------------------------------------------------------------------
#| lst-cap: The batch training loop
inputs <- np_array(inputs, dtype = "float32")
targets <- np_array(targets, dtype = "float32")

for (step in 1:40) {
  loss <- training_step(inputs, targets, W, b)
  if (step < 5 || !step %% 5)
    cat(sprintf("Loss at step %d: %.4f\n", step, loss))
}


# ----------------------------------------------------------------------
#| fig-show: hide
#| fig-format: png
predictions <- model(inputs, W, b)

predictions <- as.array(predictions)                                            # <1>
inputs <- as.array(inputs)                                                      # <1>
targets <- as.array(targets)                                                    # <1>

plot(x = inputs[, 1], y = inputs[, 2],
     col = ifelse(predictions[, 1] > 0.5, "green", "purple"))


# ----------------------------------------------------------------------
#| fig-show: hide
#| fig-format: png
plot(x = inputs[, 1], y = inputs[, 2],
     col = ifelse(predictions[, 1] <= 0.5, "purple", "green"))

slope <- -W[1, ] / W[2, ]
intercept <- (0.5 - b) / W[2, ]
abline(as.array(intercept), as.array(slope), col = "red")


