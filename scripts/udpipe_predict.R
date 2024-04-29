# Load necessary libraries
library(udpipe)
library(optparse)

setwd("./conllu")
# # Define command line options
option_list <- list(
  make_option(c("-s", "--source"), type = "character", default = NULL, help = "Input 3 letter iso code for source language", metavar = "CONLL_FILE"),
  make_option(c("-t", "--transfer"), type = "character", default = NULL, help = "Input 3 letter iso code for train_language", metavar = "MODEL_FILE")
)

# Parse command line arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
# Check if the CoNLL-U file name and the model file name are provided
if (is.null(opt$source) || is.null(opt$transfer)) {
  cat("Error: Please provide the iso for source and transfer languages\n")
  quit(status = 1)
}
inverse = paste("./models", paste(paste(opt$transfer, opt$source,sep = "-"), ".udpipe", sep = ""))
if (!(file.exists(inverse))) {
  model_file = paste("./models/", paste(paste(opt$source, opt$transfer, sep = "-"), ".udpipe", sep = ""))
}else{ 
    model_file = inverse
}
m <- udpipe_load_model(model_file)

dev = paste(opt$source, "dev.conllu", sep = "_")

## Evaluate the accuracy
goodness_of_fit <- udpipe_accuracy(m, dev, tokenizer = "default", tagger = "default")
accuracy <- goodness_of_fit$accuracy

u_pattern <- "(upostag):\\s*([\\d\\.]+)%"
x_pattern <- "(xpostag):\\s*([\\d\\.]+)%"

upos <- regmatches(accuracy, regexpr(u_pattern, accuracy, perl = TRUE))
xpos <-regmatches(accuracy, regexpr(x_pattern, accuracy, perl = TRUE))
upos <- gsub("[^0-9.%]", "", upos)

# Create a data frame
data <- data.frame(
  source = opt$source,
  transfer = opt$transfer,
  upos = upos,
  xpos = xpos
)
existing_data <- read.csv("./models/accuracy.csv", stringsAsFactors = FALSE)

# Append new data to the existing data
combined_data <- rbind(existing_data, data)

# Save the combined data to the CSV file
write.csv(combined_data, "./models/accuracy.csv", row.names = FALSE)

