#!/bin/bash

# Check if the file exists
language_file="../resources/test_langs.txt"
if [ ! -f $language_file ]; then
    echo "Error: language file not found"
    exit 1
fi

hile IFS= read -r language1; do
    while IFS= read -r language2; do
        if [ "$language1" != "$language2" ]; then
            echo "Processing $language1 and $language2"
            # Call the other script and pass the languages as arguments
            sbatch ./train_udpipe.sh "$language1" "$language2"
        fi
    done < $language_file
done < $language_file