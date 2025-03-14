set terminal pngcairo enhanced size 800,600
set output 'execution_times_graph.png'
set title "Global execution time by number of clients"
set xlabel "Number of clients"
set ylabel "Total time (in seconds)"
# Utiliser tail pour ignorer la première ligne (en-tête)
set datafile separator ","
plot "sizes_execution_times.log" using 1:2 with linespoints title 'Temps Global'
