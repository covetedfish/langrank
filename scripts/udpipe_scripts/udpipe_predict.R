# Load necessary libraries
library(udpipe)
library(optparse)

setwd("./conllu")
# # Define command line options
option_list <- list(
  make_option(c("-t", "--target"), type = "character", default = NULL, help = "Input 3 letter iso code for target language", metavar = "CONLL_FILE"),
  make_option(c("-r", "--transfer"), type = "character", default = NULL, help = "Input 3 letter iso code for transfer language", metavar = "MODEL_FILE")
)

# Parse command line arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
# Check if the CoNLL-U file name and the model file name are provided
if (is.null(opt$target) || is.null(opt$transfer)) {
  cat("Error: Please provide the iso for target and transfer languages\n")
  quit(status = 1)
}
print(opt$target)
print(opt$transfer)

model_file = paste("./models/2000-500/", paste(paste(opt$target, opt$transfer, sep = "-"), ".udpipe", sep = ""), sep = "")
m <- udpipe_load_model(model_file)

test = paste("./dev-test/", paste(opt$target, "dev.conllu", sep = "_"), sep="")
print(test)
# Evaluate the accuracy
goodness_of_fit <- udpipe_accuracy(m, test, tokenizer = "default", tagger = "default")
accuracy <- goodness_of_fit$accuracy

u_pattern <- "(upostag):\\s*([\\d\\.]+)%"
x_pattern <- "(xpostag):\\s*([\\d\\.]+)%"
a_pattern <- "(alltags):\\s*([\\d\\.]+)%"

upos <- regmatches(accuracy, regexpr(u_pattern, accuracy, perl = TRUE))
xpos <-regmatches(accuracy, regexpr(x_pattern, accuracy, perl = TRUE))
apos <-regmatches(accuracy, regexpr(a_pattern, accuracy, perl = TRUE))

upos <- gsub("[^0-9.%]", "", upos)
xpos <- gsub("[^0-9.%]", "", xpos)
apos <- gsub("[^0-9.%]", "", apos)

# Create a data frame
data <- data.frame(
  target = opt$target,
  transfer = opt$transfer,
  upos = upos,
  xpos = xpos,
  apos = apos
)
existing_data <- read.csv("../udpipe_fra_dev.csv", stringsAsFactors = FALSE)

# Append new data to the existing data
combined_data <- rbind(existing_data, data)

# Save the combined data to the CSV file
write.csv(combined_data, "../udpipe_fra_dev.csv", row.names = FALSE)

