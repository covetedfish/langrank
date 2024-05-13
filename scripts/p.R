library(udpipe)
library(optparse)

setwd("./conllu")
model = ("./models/ test.udpipe")
print(file.exists(model))
train = "/projects/enri8153/langrank/conllu/fra_train.conllu"
dev = "/projects/enri8153/langrank/conllu/fra_dev.conllu"
m <- udpipe_train(file = model, 
                    files_conllu_training = train, 
                    files_conllu_holdout  = dev,
                    annotation_tokenizer = "default",
                    annotation_tagger = "default")