set datafile separator ","
set terminal pngcairo size 800,600

# Memory Usage Plot
set output 'memory_usage.png'
set title "Memory Usage vs Grid Size"
set xlabel "Grid Size"
set ylabel "Memory Usage (B)"
plot "nvprof_results.csv" using 2:4 with linespoints title "Memory Usage" lt rgb "blue"

# Kernel Time Plot
set output 'kernel_time.png'
set title "Kernel Time vs Grid Size"
set xlabel "Grid Size"
set ylabel "Kernel Time (us)"
plot "nvprof_results.csv" using 2:5 with linespoints title "Kernel Time" lt rgb "red"

# GPU Time Plot
set output 'gpu_time.png'
set title "GPU Time vs Grid Size"
set xlabel "Grid Size"
set ylabel "GPU Time (us)"
plot "nvprof_results.csv" using 2:6 with linespoints title "GPU Time" lt rgb "green"

