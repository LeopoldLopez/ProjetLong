set terminal pngcairo size 800,600 enhanced font 'Arial,10'
set output 'Output/plot_Add_Array_CPU.png'

# Set titles and labels
set title "Mean Times vs Size"
set xlabel "Size"
set ylabel "Time (seconds)"
set logscale x 2 
set grid

# Define line styles
set style line 1 lc rgb '#0072bd' lt 1 lw 2 pt 7 ps 1.5  # Blue
set style line 2 lc rgb '#d95319' lt 1 lw 2 pt 5 ps 1.5  # Orange

# Plot the data
plot "Output/results.dat" using 1:2 with linespoints ls 1 title "Mean Init Time (s)", \
     "Output/results.dat" using 1:3 with linespoints ls 2 title "Mean Add Time (s)" 

