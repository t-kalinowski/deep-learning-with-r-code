# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
library(keras3)
use_backend("torch")

torch <- import("torch")


# ----------------------------------------------------------------------
#| eval: false
#| lst-cap: All-ones or all-zeros tensors
# torch$ones(size = shape(2, 2))                                                  # <1>


# ----------------------------------------------------------------------
torch$ones(size = shape(2, 2))


# ----------------------------------------------------------------------
torch$zeros(size = shape(2, 2))


# ----------------------------------------------------------------------
#| eval: false
# torch$tensor(c(1, 2, 3), dtype = torch$float32)                                 # <1>


# ----------------------------------------------------------------------
torch$tensor(c(1, 2, 3), dtype = torch$float32)


# ----------------------------------------------------------------------
#| eval: false
#| lst-cap: Random tensors
# torch$normal(                                                                   # <1>
#   mean = torch$zeros(size = shape(1, 3)),                                       # <1>
#   std = torch$ones(size = shape(1, 3))                                          # <1>
# )


# ----------------------------------------------------------------------
torch$normal(                                                                   # <1>
  mean = torch$zeros(size = shape(1, 3)),                                       # <1>
  std = torch$ones(size = shape(1, 3))                                          # <1>
)


# ----------------------------------------------------------------------
#| eval: false
# torch$rand(1L, 3L)                                                              # <1>


# ----------------------------------------------------------------------
torch$rand(1L, 3L)                                                              # <1>


# ----------------------------------------------------------------------
torch$rand(!!!shape(1, 3))


# ----------------------------------------------------------------------
x <- torch$zeros(size = shape(2, 2))
x@py[0, 0] <- 1
x


# ----------------------------------------------------------------------
#| lst-cap: Creating a PyTorch parameter.
x <- torch$zeros(shape(2, 1))
p <- torch$nn$parameter$Parameter(data = x)                                     # <1>


# ----------------------------------------------------------------------
#| lst-cap: A few basic math operations in PyTorch
a <- torch$ones(shape(2, 2))
b <- torch$square(a)                                                            # <1>
c <- torch$sqrt(a)                                                              # <2>
d <- b + c                                                                      # <3>
e <- torch$matmul(a, b)                                                         # <4>
f <- torch$cat(list(a, b), dim = 0L)                                            # <5>


# ----------------------------------------------------------------------
dense <- function(inputs, W, b) {
  torch$nn$relu(torch$matmul(inputs, W) + b)
}


# ----------------------------------------------------------------------
#| results: hide
#| lst-cap: "Computing a gradient with `.backward()`"
input_var <- torch$tensor(3.0, requires_grad = TRUE)                            # <1>
result <- torch$square(input_var)
result$backward()                                                               # <2>
gradient <- input_var$grad                                                      # <2>
gradient


# ----------------------------------------------------------------------
gradient


# ----------------------------------------------------------------------
#| results: hide
result <- torch$square(input_var)
result$backward()
input_var$grad                                                                  # <1>


# ----------------------------------------------------------------------
input_var$grad


# ----------------------------------------------------------------------
input_var$grad <- NULL


# ----------------------------------------------------------------------
input_dim <- 2L
output_dim <- 1L

W <- torch$rand(input_dim, output_dim, requires_grad = TRUE)
b <- torch$zeros(output_dim, requires_grad = TRUE)


# ----------------------------------------------------------------------
model <- function(inputs, W, b) {
  torch$matmul(inputs, W) + b
}


# ----------------------------------------------------------------------
mean_squared_error <- function(targets, predictions) {
  per_sample_losses <- torch$square(targets - predictions)
  torch$mean(per_sample_losses)
}


# ----------------------------------------------------------------------
learning_rate <- 0.1

training_step <- function(inputs, targets, W, b) {
  predictions <- model(inputs, W, b)                                            # <1>
  loss <- mean_squared_error(targets, predictions)                              # <1>
  loss$backward()                                                               # <2>
  grad_loss_wrt_W <- W$grad                                                     # <3>
  grad_loss_wrt_b <- b$grad                                                     # <3>
  with(torch$no_grad(), {
    W$sub_(grad_loss_wrt_W * learning_rate)                                     # <4>
    b$sub_(grad_loss_wrt_b * learning_rate)                                     # <4>
  })
  W$grad <- b$grad <- NULL                                                      # <5>
  loss
}


# ----------------------------------------------------------------------
#| lst-cap: "Defining a `torch$nn$Module`"
LinearModel(torch$nn$Module) %py_class% {
  `__init__` <- function(self) {
    super()$`__init__`()
    self$W <- torch$nn$Parameter(torch$rand(input_dim, output_dim))
    self$b <- torch$nn$Parameter(torch$zeros(output_dim))
  }
  forward <- function(self, inputs) {
    torch$matmul(inputs, self$W) + self$b
  }
}


# ----------------------------------------------------------------------
model <- LinearModel()


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
torch_inputs <- torch$tensor(inputs, dtype = torch$float32)
output <- model(torch_inputs)


# ----------------------------------------------------------------------
optimizer <- torch$optim$SGD(model$parameters(), lr = learning_rate)


# ----------------------------------------------------------------------
training_step <- function(inputs, targets) {
  predictions <- model(inputs)
  loss <- mean_squared_error(targets, predictions)
  loss$backward()
  optimizer$step()
  model$zero_grad()
  loss
}


# ----------------------------------------------------------------------
#| eval: false
# with(torch$no_grad(), {
#   W$sub_(grad_loss_wrt_W * learning_rate)                                       # <1>
#   b$sub_(grad_loss_wrt_b * learning_rate)                                       # <1>
# })


# ----------------------------------------------------------------------
compiled_model <- torch$compile(model)


# ----------------------------------------------------------------------
dense <- torch$compile(function(inputs, W, b) {
  torch$nn$relu(torch$matmul(inputs, W) + b)
})


