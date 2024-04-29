#!/bin/bash

# Check if the file exists
language_file="./resources/pos_languages.txt"
if [ ! -f $language_file ]; then
    echo "Error: language file not found"
    exit 1
fi

# Read languages from the file into an array
mapfile -t languages < $language_file

# Iterate through the array to process each pair of languages exactly once
for ((i=0; i<${#languages[@]}; i++)); do
    for ((j=0; j<${#languages[@]}; j++)); do
        echo "Processing ${languages[i]} and ${languages[j]}"
        # Call the other script and pass the languages as arguments
        language_pair="${languages[i]} ${languages[j]}"
        sbatch --export=LANGUAGES="${language_pair}" ./scripts/predict_udpipe.sh 
    done
done

 echo "done"