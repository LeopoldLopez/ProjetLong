set terminal pngcairo enhanced size 800,600
set output 'execution_times_graph.png'
set title "Temps d'exécution par nombre de clients"
set xlabel "Taille (size)"
set ylabel "Temps global (en secondes)"
# Utiliser tail pour ignorer la première ligne (en-tête)
set datafile separator ","
plot "sizes_execution_times.log" using 1:2 with linespoints title 'Temps Global'
