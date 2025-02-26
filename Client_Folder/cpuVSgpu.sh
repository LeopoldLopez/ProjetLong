#!/bin/bash

sizes=(1 10 100 1000 10000 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)

output_file="cpu_vs_gpu_time.log"
> "$output_file"

for size in "${sizes[@]}"; do
    param="$size "

    for i in $(seq 1 "$size"); do
        param+="9 "
    done
    
    start_time_cpu=$(date +%s.%N)
    ./client sumCPU $param 
    end_time_cpu=$(date +%s.%N)

    cpu_duration=$(echo "$end_time_cpu - $start_time_cpu" | bc)

    start_time_gpu=$(date +%s.%N)
    ./client sum $param 
    end_time_gpu=$(date +%s.%N)

    gpu_duration=$(echo "$end_time_gpu - $start_time_gpu" | bc)

    echo "$size,$cpu_duration,$gpu_duration" >> "$output_file" 
    wait
done

# Générer le graphique avec gnuplot

gnuplot_script="cpu_gpu_plot_script.gp"
> "$gnuplot_script"

# Créer le fichier de script pour gnuplot
cat <<EOL > "$gnuplot_script"
set terminal pngcairo enhanced size 800,600
set output 'execution_times_cpu_gpu.png'
set title "Temps d'exécution CPU vs GPU"
set xlabel "Taille du vecteur de paramètres"
set ylabel "Temps (en secondes)"
set datafile separator ","

# Tracer les deux courbes
plot "$output_file" using 1:2 with linespoints title 'CPU', \
     "$output_file" using 1:3 with linespoints title 'GPU'
EOL


gnuplot "$gnuplot_script"

echo "Graphique généré sous le nom 'execution_times_cpu_gpu.png'."
