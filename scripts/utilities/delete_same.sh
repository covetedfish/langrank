#!/bin/bash

# Define the directory path
directory="/projects/enri8153/langrank/conllu/models"

# Navigate to the directory
cd "$directory" || exit

# Loop through each file in the directory
for file in *; do
    # Check if the file is a regular file
    if [ -f "$file" ]; then
        # Get the first three characters of the filename
        x="${file:1:3}"
    # Check if the filename starts with "x"
        # Delete the file if it matches the pattern "x-x.udpipe"
        if [[ "$file" =~ ^\ $x-$x\.udpipe$ ]]; then
            rm "$file"
            echo "Deleted $file"
        fi
    fi
done

#what i want to do is iterate through missing pairs, 
#check if the model already exists and delete it if it does
