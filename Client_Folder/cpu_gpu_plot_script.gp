set terminal pngcairo enhanced size 800,600
set output 'execution_times_cpu_gpu.png'
set title "Temps d'exécution CPU vs GPU"
set xlabel "Taille du vecteur de paramètres"
set ylabel "Temps (en secondes)"
set datafile separator ","

# Tracer les deux courbes
plot "cpu_vs_gpu_time.log" using 1:2 with linespoints title 'CPU',      "cpu_vs_gpu_time.log" using 1:3 with linespoints title 'GPU'
