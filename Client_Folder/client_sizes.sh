#!/bin/bash
sizes=()
for i in {1..20}; do
  sizes+=($((i * 10)))
done

last_index=$(( ${#sizes[@]} - 1 ))
echo $last_index
max_value=${sizes[$last_index]}

for i in $(seq 1 "$max_value"); do
    touch "output$i.txt"
done

output_file="sizes_execution_times.log"
> "$output_file"

# Boucle à travers les tailles
for size in "${sizes[@]}"; do

    global_duration=0.0

    echo "Exécution pour la taille $size..."

    ./exec_clients "$size"


    for i in $(seq 1 "$size"); do
        while IFS= read -r line; do
            global_duration=$(echo "$global_duration + $line" | bc)
        done < "output$i.txt"
    done

    # Si la durée commence par un point, on ajoute un zéro avant
    if [[ "$global_duration" == .* ]]; then
        global_duration="0$global_duration"
    fi
    

    echo "$size,$global_duration" >> "$output_file"
    
    echo "Temps pour taille $size : $global_duration secondes"

    wait
done

for i in $(seq 1 "$max_value"); do
    rm -f "output$i.txt"
done

echo "Exécution complète. Résultats enregistrés dans $output_file."

# Générer le graphique avec gnuplot

gnuplot_script="plot_script.gp"
> "$gnuplot_script"

# Créer le fichier de script pour gnuplot
cat <<EOL > "$gnuplot_script"
set terminal pngcairo enhanced size 800,600
set output 'execution_times_graph.png'
set title "Temps d'exécution par nombre de clients"
set xlabel "Taille (size)"
set ylabel "Temps global (en secondes)"
# Utiliser tail pour ignorer la première ligne (en-tête)
set datafile separator ","
plot "$output_file" using 1:2 with linespoints title 'Temps Global'
EOL

gnuplot "$gnuplot_script"

echo "Graphique généré sous le nom 'execution_times_graph.png'."
