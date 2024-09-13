
library(udpipe)
library(optparse)

setwd("./conllu")

count_samples <- function(conllu_file) {
    conllu_data <- udpipe_read_conllu(conllu_file)
    
    # Count number of sentences
    num_sentences <- length(unique(conllu_data$sentence_id))
    
    return(num_sentences)
}

verify_language <- function(lang) {
    big_file =  paste("2000/", paste(lang, "train.conllu", sep = "_"), sep = "")
    small_file = paste("500/", paste(lang, "train.conllu", sep = "_"), sep = "")
    big_count = count_samples(big_file)
    small_count = count_samples(small_file)
    if (!(small_count==500 && big_count==2000)) {
        print(paste(lang, "data has issues", sep= " "))
    }
    else{
        print(paste(lang, "successful", sep = " "))
    }
}

language_file = "../resources/transfer_langs.txt"
languages <- readLines(language_file)
for (language in languages) {
    verify_language(language)
}