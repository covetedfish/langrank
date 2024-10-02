FILE_EXTENSION = "{source}/{task}/{arch}/{key}/{dataset}"
TRAIN_FILE = "./training-data/{arch}_{task}_ranked.pkl"
GOLD_FILE = "./training-data/{arch}_{task}_golds.pkl"
ABLATIONS_PATH = "./resources/fine_categorization.csv"
RESULTS_PATH = "./results.tsv"
RESULTS_DIR = "./results"       
OLD_RAW = "./resources/LangRank Transfer Language Raw Data - POS Results.csv"
FINAL_STANZA = "./golds/stanza_scores_2000_all.csv"
FINAL_XLMR =  "./golds/xlmr_scores_2000_all.csv"
STANZA_DEV = "../stanza/results_dev.csv"
XPOS_DEV = "../xpos/results.csv"
# xpos and stanza golds aligned to one another
ALIGNED_STANZA = "./golds/aligned_stanza.csv"
ALIGNED_XPOS = "./golds/aligned_xlmr.csv"

#old bilingual lstm data aligned to new bilingual lstm data (I fear these will be empty), yeah just hungarian and telegu :(
ALIGNED_ORIG = "./golds/aligned_orig.csv"
ALIGNED_STANZA_ORIG = "./golds/aligned_stanza_orig.csv"

GRAM_RAW = "./golds/orig_grambank.csv" #filtered original rankings from langrank to exclude languages not present in grambank
