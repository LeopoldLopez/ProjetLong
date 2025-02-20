#!/bin/bash


maxNumBlocksPerGrid=58
maxNumThreadsPerBlock=1024

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <executable> <arg1> <arg2> <arg3> <arg4> <arg5> ..."
    exit 1
fi


size=$(($# - 1))
args=("${@:2}")
rep=10

# Output CSV file
tmp_file="tmp_nvprof.log"
output_file="nvprof_results.csv"
if [ -f "$output_file" ]; then
    rm "$output_file"
    echo "$output_file removed."
else
    echo "$output_file does not exist."
fi
echo "Size,GridSize,BlockSize,MemoryUsage,KernelTime,GPUTime" > "$output_file"

for ((grid_size=1; grid_size<=size; grid_size++)); do 
    block_size=$(((size+grid_size)/grid_size)) 
    sumTotal=0
    kernel_timeTotal=0
    gpu_timeTotal=0
    
    for ((it=0; it<rep; it++)); do 
        output=$(nvprof --print-gpu-trace --csv --log-file "$tmp_file" ./"$1" "$size" "$grid_size" "$block_size" "${args[@]}")
        sumTotal=$(echo "$sumTotal + $(awk -F',' 'NR>2 {sum+=$12} END {print sum}' "$tmp_file")" | bc)


        


        output=$(nvprof --csv --log-file "$tmp_file" ./"$1" "$size" "$grid_size" "$block_size" "${args[@]}")
        kernel_timeTotal=$(echo "$kernel_timeTotal + $(awk -F',' '/"GPU activities"/ {print $3; exit}' "$tmp_file")" | bc)
        gpu_timeTotal=$(echo "$gpu_timeTotal + $(awk -F',' '/"GPU activities"/ {if (first) total+=$3; else first=1} END {print total}' "$tmp_file")" | bc)
        
    done
    
    sum=$(echo "scale=6; $sumTotal / $rep" | bc)
    kernel_time=$(echo "scale=6; $kernel_timeTotal / $rep" | bc)
    gpu_time=$(echo "scale=9; $gpu_timeTotal / $rep" | bc)
    echo "$size,$grid_size,$block_size,$sum,$kernel_time,$gpu_time" >> "$output_file"

done

echo "Profiling complete. Results saved in $output_file."

