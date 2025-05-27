#!/usr/bin/env Rscript

if(!requireNamespace("remotes")) install.packages("remotes")

remotes::update_packages()
remotes::install_cran(c("readr", "tibble", "zip", "fs", "listarrays", "keras"))

envname <- "r-keras"

if("--fresh" %in% commandArgs(TRUE)) {
  reticulate::miniconda_uninstall()
  unlink("~/.pyenv", recursive = TRUE)
  unlink(paste0("~/.virtualenvs/", envname), recursive = TRUE)
}


python <- reticulate::install_python("3.9:latest")
reticulate::virtualenv_create(envname, python = python)

keras::install_keras(
  envname = envname,
  extra_packages = c("keras-tuner", "ipython", "kaggle")
)
