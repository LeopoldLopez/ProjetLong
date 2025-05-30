#!/bin/bash

sizes=()
for i in {1..100}; do
  sizes+=($((i * 50)))
done
    

output_file="cpu_vs_gpu_time.log"
> "$output_file"

for size in "${sizes[@]}"; do
    
    cpu_duration=$(./"client_test" sumCPU $size)

    gpu_duration=$(./"client_test" sum $size)

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
set title "Execution time of the CPU and GPU
set xlabel "Size of the parameter vector"
set ylabel "Time (in seconds)"
set datafile separator ","

# Tracer les deux courbes
plot "$output_file" using 1:2 with linespoints title 'CPU', \
     "$output_file" using 1:3 with linespoints title 'GPU'
EOL


gnuplot "$gnuplot_script"

echo "Graphique généré sous le nom 'execution_times_cpu_gpu.png'."
