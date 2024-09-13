#!/bin/bash
#trains models for each combination of target languages and transfer languages

predict=false
missing=false
while getopts "pm" flag; do
    case "${flag}" in
        p) predict=true;;
        m) missing=true;;
    esac
done
echo $missing

if $missing; then
        tgt_language_file="./resources/missing_tgt.txt"
        trnsfr_language_file="./resources/missing_trnsfr.txt"
    else
        tgt_language_file="./resources/missing_tgt.txt"
        trnsfr_language_file="./resources/transfer_langs.txt"
fi

if [ ! -f $tgt_language_file ]; then
    echo "Error: tgt file not found"
    exit 1
fi


if [ ! -f $trnsfr_language_file ]; then
    echo "Error: transfer file not found"
    exit 1
fi

# Read languages from the file into an array
mapfile -t tgt_languages < $tgt_language_file
mapfile -t trnsfr_languages < $trnsfr_language_file
echo $tgt_languages

 if $missing; then
        for ((i=0; i<${#tgt_languages[@]}; i++)); do
            if $predict; then
                    language_pair="${tgt_languages[i]} ${trnsfr_languages[i]}"
                    echo "Predicting ${tgt_languages[i]} and ${trnsfr_languages[i]}"
                    sbatch --requeue --export=LANGUAGES="${language_pair}" ./scripts/udpipe_scripts/predict_udpipe.sh
            else
                language_pair="${tgt_languages[i]} ${trnsfr_languages[i]}"
                echo "Training ${tgt_languages[i]} and ${trnsfr_languages[i]}"
                sbatch --requeue --export=LANGUAGES="${language_pair}" ./scripts/udpipe_scripts/train_udpipe.sh
            fi
        done
    else
        for ((i=0; i<${#tgt_languages[@]}; i++)); do
            for ((j=0; j<${#trnsfr_languages[@]}; j++)); do
                if $predict; then
                        language_pair="${tgt_languages[i]} ${trnsfr_languages[j]}"
                        echo $language_pair
                        echo "Predicting ${tgt_languages[i]} and ${trnsfr_languages[j]}"
                        sbatch --requeue --export=LANGUAGES="${language_pair}" ./scripts/udpipe_scripts/predict_udpipe.sh
                    else
                        language_pair="${tgt_languages[i]} ${trnsfr_languages[j]}"
                        echo "Training ${tgt_languages[i]} and ${trnsfr_languages[j]}"
                        sbatch --requeue --export=LANGUAGES="${language_pair}" ./scripts/udpipe_scripts/train_udpipe.sh
                fi
            done
        done
 fi
    

 echo "done"