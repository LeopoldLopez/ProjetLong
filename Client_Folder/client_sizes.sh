#!/bin/bash

sizes=(2 4 8 16 32 64 128)

output_file="sizes_execution_times.log"
> "$output_file"

# Boucle à travers les tailles
for size in "${sizes[@]}"; do
    echo "Exécution pour la taille $size..."

    start_time=$(date +%s.%N)
    ./exec_clients.sh "$size"
    end_time=$(date +%s.%N)

    global_duration=$(echo "$end_time - $start_time" | bc)

    # Si la durée commence par un point, on ajoute un zéro avant
    if [[ "$global_duration" == .* ]]; then
        global_duration="0$global_duration"
    fi

    echo "$size,$global_duration" >> "$output_file"
    
    echo "Temps pour taille $size : $global_duration secondes"
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
