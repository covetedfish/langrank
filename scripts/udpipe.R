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

# Function to train a POS tagging model with udpipe
train_pos_model <- function(train, dev, model_file) {
  # Load CoNLL-U formatted data

  m <- udpipe_train(file = model_file, 
                    files_conllu_training = train, 
                    files_conllu_holdout  = dev,
                    annotation_tokenizer = list(dimension = 64, epochs = 20, segment_size=200, initialization_range = 0.1, 
                                                batch_size = 50, learning_rate = 0.002, learning_rate_final=0, dropout = 0.1, early_stopping = 1),
                    annotation_tagger = list(models = 2, 
                                             templates_1 = "lemmatizer", guesser_suffix_rules_1 = 8, guesser_enrich_dictionary_1 = 4, guesser_prefixes_max_1 = 4, 
                                             use_lemma_1 = 1,provide_lemma_1 = 1, use_xpostag_1 = 0, provide_xpostag_1 = 0, 
                                             use_feats_1 = 0, provide_feats_1 = 0, prune_features_1 = 1, 
                                             templates_2 = "tagger", guesser_suffix_rules_2 = 8, guesser_enrich_dictionary_2 = 4, guesser_prefixes_max_2 = 0, 
                                             use_lemma_2 = 1, provide_lemma_2 = 0, use_xpostag_2 = 1, provide_xpostag_2 = 1, 
                                             use_feats_2 = 1, provide_feats_2 = 1, prune_features_2 = 1))
  
}

make_train_file <- function(source, transfer) {
  print(source)
  print(transfer)
  source_file = paste(source, "train.conllu", sep = "_")
  transfer_file = paste(transfer, "train.conllu", sep = "_")
  
  # Load contents from two files
  file1 <- udpipe_read_conllu(source_file)
  file2 <- udpipe_read_conllu(transfer_file)

  # Concatenate their contents
  data <- rbind(file1, file2)
  shuffled_data= data[sample(1:nrow(data)), ] 
  save_file = paste(paste(source, transfer, sep = "-"), "train.conllu", sep = "_")
  # Save to a new file
  cat(as_conllu(data), file = file(save_file, encoding = "UTF-8"))
  return(save_file)
}

a = Sys.time()
# file_conllu <- system.file(package = "udpipe", "dummydata", "traindata.conllu")
# Train POS model
dev = paste(opt$source, "dev.conllu", sep = "_")
test= paste(opt$source, "test.conllu",sep =  "_")

inverse = paste("./models", paste(paste(opt$transfer, opt$source,sep = "-"), ".udpipe", sep = ""))
if (!(file.exists(inverse))) {
  model_file = paste("./models/", paste(paste(opt$source, opt$transfer, sep = "-"), ".udpipe", sep = ""))
  print(opt$source)
  print(opt$transfer)
  train <- make_train_file(opt$source, opt$transfer)

  train_pos_model(train, dev, model_file)
  # train_pos_model(file_conllu, "toymodel.udpipe")
  file.remove(train)
  m <- udpipe_load_model(model_file)

} else {
  m <- udpipe_load_model(inverse)
}
## Evaluate the accuracy
goodness_of_fit <- udpipe_accuracy(m, dev, tokenizer = "default", tagger = "default")
accuracy <- goodness_of_fit$accuracy

b = Sys.time()
runtime = b - a

u_pattern <- "(upostag):\\s*([\\d\\.]+)%"
x_pattern <- "(xpostag):\\s*([\\d\\.]+)%"

upos <- regmatches(accuracy, regexpr(u_pattern, accuracy, perl = TRUE))
xpos <-regmatches(accuracy, regexpr(x_pattern, accuracy, perl = TRUE))

# Create a data frame
data <- data.frame(
  source = opt$source,
  transfer = opt$transfer,
  upos = upos,
  xpos = xpos,
  Runtime = runtime
)

# Save to CSV
write.csv(data, "./models/accuracy.csv", row.names = FALSE, append = TRUE)

