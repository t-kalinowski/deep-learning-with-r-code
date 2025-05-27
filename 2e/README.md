## Deep Learning with R, 2nd Edition (Code Only)

The folder contains just the code from the book "Deep Learning with R, 2nd Edition".

You can install all the dependencies by cloning this repo and sourcing the `"install-r-tensorflow.R"` script,
either at the R console or the terminal:

```bash
Rscript install-r-tensorflow.R
```

The script creates a "r-keras" virtual environment that will automatically be
discovered by the `keras` R package.

Note: the install script assumes that R and CUDA drivers are already installed.

Modern reticulate no longer requires setting up a manual virtual environment.
Instead of running the `install-r-tensorflow.R` script, you can instead
request that reticulate use an ephemeral python installation by running this
at the start of the R session:

```r
reticulate::py_require(c(

  "tensorflow",
  # If you are on Linux and want to use a GPU, you can instead install:
  # "tensorflow[and-cuda]", or to force cpu only, "tensorflow-cpu"

  # install legacy (v2) keras
  "tf-keras",

  # additional packages used in the book
  "keras-tuner", "ipython", "kaggle"
))
```
