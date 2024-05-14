#!/bin/bash

# Check if the file exists
src_language_file="./resources/missing_src.txt"
trnsfr_language_file="./resources/missing_trnsfr.txt"

if [ ! -f $src_language_file ]; then
    echo "Error: src file not found"
    exit 1
fi


if [ ! -f $trnsfr_language_file ]; then
    echo "Error: transfer file not found"
    exit 1
fi

# Read languages from the file into an array
mapfile -t src_languages < $src_language_file
mapfile -t trnsfr_languages < $trnsfr_language_file


# Iterate through the array to process each pair of languages exactly once
for ((i=0; i<${#src_languages[@]}; i++)); do
    # for ((j=0; j<${#src_languages[@]}; j++)); do
    # Call the other script and pass the languages as arguments
    language_pair="${src_languages[i]]} ${trnsfr_languages[i]}"
    FILE="./conllu/models/ ${src_languages[i]]}-${trnsfr_languages[i]}.udpipe"
    INVERSE="./conllu/models/ ${trnsfr_languages[i]]}-${src_languages[i]}.udpipe"
    if [ -f "$FILE" ] || [ -f "$INVERSE" ]; then
        echo "Predicting ${src_languages[i]} and ${trnsfr_languages[i]}"
        sbatch --export=LANGUAGES="${language_pair}" ./scripts/predict_udpipe.sh 
    else
        echo "Training ${src_languages[i]} and ${trnsfr_languages[i]}"
        sbatch --export=LANGUAGES="${language_pair}" ./scripts/train_udpipe.sh 
    fi
done

 echo "done"