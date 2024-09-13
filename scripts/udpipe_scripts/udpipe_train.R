# Load necessary libraries
library(udpipe)
library(optparse)

setwd("./conllu")
# # Define command line options
option_list <- list(
  make_option(c("-t", "--target"), type = "character", default = NULL, help = "Input 3 letter iso code for target language", metavar = "target_FILE"),
  make_option(c("-r", "--transfer"), type = "character", default = NULL, help = "Input 3 letter iso code for train_language", metavar = "TRAIN_FILE")
)

# Parse command line arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
# Check if the CoNLL-U file name and the model file name are provided
if (is.null(opt$target) || is.null(opt$transfer)) {
  cat("Error: Please provide the iso for target and transfer languages\n")
  quit(status = 1)
}

# Function to train a POS tagging model with udpipe
train_pos_model <- function(train, dev, model_file) {
  # Load CoNLL-U formatted data
  if (!(file.exists(dev))) {
    cat("Error: dev file doesn't exist\n")
    quit(status = 1)
  }
  if (!(file.exists(train))) {
    cat("Error: train file doesn't exist\n")
    quit(status = 1)
  }
  if (!(file.exists(model_file))) {
    print("model file does not yet exist")
  }else{
    print("model file already exists")
  }
  #2 models (lemmatizer in one tagger in the other) shown in udpipe 1.2 to be better )
  #iterations set to 20 by default. Perhaps we just need to train longer?
  #or, the tokenizer is the problem?
  m <- udpipe_train(file = model_file, 
                    files_conllu_training = train, 
                    files_conllu_holdout  = dev,
                    annotation_tokenizer = list(dimension = 64, epochs = 20, segment_size=200, initialization_range = 0.1, 
                                                batch_size = 50, learning_rate = 0.002, learning_rate_final=0, dropout = 0.1, early_stopping = 1),
                    annotation_tagger = list(models = 2, 
                                             templates_1 = "lemmatizer", guesser_suffix_rules_1 = 8, guesser_enrich_dictionary_1 = 4, guesser_prefixes_max_1 = 4, 
                                             use_lemma_1 = 1,provide_lemma_1 = 1, use_xpostag_1 = 0, provide_xpostag_1 = 0, 
                                             use_feats_1 = 0, provide_feats_1 = 0, prune_features_1 = 1, iterations= 100, early_stopping = 1,
                                             templates_2 = "tagger", guesser_suffix_rules_2 = 8, guesser_enrich_dictionary_2 = 4, guesser_prefixes_max_2 = 0, 
                                             use_lemma_2 = 1, provide_lemma_2 = 0, use_xpostag_2 = 1, provide_xpostag_2 = 1, 
                                             use_feats_2 = 1, provide_feats_2 = 1, prune_features_2 = 1, iterations = 100, early_stopping = 1))
}

make_train_file <- function(target, transfer) {
  print(target)
  print(transfer)
  target_file = paste("500/", paste(target, "train.conllu", sep = "_"), sep = "")
  transfer_file = paste("2000/", paste(transfer, "train.conllu", sep = "_"), sep = "")
  
  # Load contents from two files
  file1 <- readLines(target_file)
  file2 <- readLines(transfer_file)

  # Concatenate their contents
  data <- c(file1, "", file2)  # Adding an empty line between the files
  save_file = paste("2000-500/", paste(paste(target, transfer, sep = "-"), "train.conllu", sep = "_"), sep = "")
  # Save to a new file
  writeLines(data, con = save_file)
  return(save_file)
}


a = Sys.time()
# file_conllu <- system.file(package = "udpipe", "dummydata", "traindata.conllu")
# Train POS model
dev = paste("dev-test/", paste(opt$target, "dev.conllu", sep = "_"), sep = "")
test= paste("dev-test/", paste(opt$target, "test.conllu",sep =  "_"), sep = "")
print(dev)
print(test)


if (!(file.exists(dev))) {
  cat("Error: dev file doesn't exist\n")
  quit(status = 1)
}
if (!(file.exists(test))) {
  cat("Error: test file doesn't exist\n")
  quit(status = 1)
}

# inverse = paste("./models/5000-500/", paste(paste(opt$transfer, opt$target,sep = "-"), ".udpipe", sep = ""), sep = "")
# if (!(file.exists(inverse))) {
#   print("NOT INVERSE")
model_file = paste("./models/2000-500-exp/", paste(paste(opt$target, opt$transfer, sep = "-"), ".udpipe", sep = ""), sep = "")
if (file.exists(model_file)) {
  print("model file already exists")
}
train <- make_train_file(opt$target, opt$transfer)
if (file.exists(train)) {
  print("train file successfully generated")
  print(train)
}
print(model_file)
train_pos_model(train, dev, model_file)
m <- udpipe_load_model(model_file)


# Evaluate the accuracy
goodness_of_fit <- udpipe_accuracy(m, dev, tokenizer = "default", tagger = "default")
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
existing_data <- read.csv("./models/2000-500-exp/udpipe_f1_dev.csv", stringsAsFactors = FALSE)

# Append new data to the existing data
combined_data <- rbind(existing_data, data)

# Save the combined data to the CSV file
write.csv(combined_data, "./models/2000-500/udpipe_f1_dev.csv", row.names = FALSE)

