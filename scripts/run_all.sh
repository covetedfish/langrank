#!/bin/bash
for ablation in "None"; do
    for source in "syntax_knn"; do
        for task in "POS"; do
            for arch in "mtt" "xpos"; do
                combo=$(printf '{"task": "%s", "ablation": "%s" , "distance": "False" , "source":"%s" , "arch":"%s"}' "$task" "$ablation" "$source" "$arch")
                combos+=("$combo")
            done
        done
    done
done
counter=400 # Initialize a counter
for c in "${combos[@]}"; do
   ((counter++))  
    filename="/projects/enri8153/langrank/configs/combo_${counter}.json"
    echo "$c" > "$filename"
    sbatch --export=COMBOS_FILE="${filename}" /projects/enri8153/langrank/scripts/run_one.sh
done

