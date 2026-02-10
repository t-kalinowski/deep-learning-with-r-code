# ----------------------------------------------------------------------
# Install required R packages (if needed)
pkgs <- c("keras3", "rsconnect")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install)) install.packages(to_install)


# ----------------------------------------------------------------------
#| eval: false
# x <- scale(x)                                                                   # <1>


# ----------------------------------------------------------------------
#| eval: false
# library(keras3)
# model |>
#   export_savedmodel("path/to/location", format = "tf_saved_model")              # <1>
# 
# reloaded_artifact <- tensorflow::tf$saved_model$load("path/to/location")        # <2>
# predictions <- reloaded_artifact$serve(input_data)                              # <2>


# ----------------------------------------------------------------------
#| eval: false
# model |> export_savedmodel("path/to/location", format = "onnx")                 # <1>


# ----------------------------------------------------------------------
#| eval: false
# reticulate::py_require("onnxruntime")
# 
# onnxruntime <- reticulate::import("onnxruntime")
# ort_session <- onnxruntime$InferenceSession("path/to/location")
# predictions <- ort_session$run(NULL, input_data)


# ----------------------------------------------------------------------
#| eval: false
# export_savedmodel(model, "model-for-serving")
# rsconnect::deployTFModel("model-for-serving")


