#!/bin/bash


if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <executable> <arg1> <arg2> <arg3> <arg4> <arg5> ..."
    exit 1
fi

args=("${@:2}")

size=$(($# - 1))
grid_size=1
block_size=$(((size + grid_size - 1) / grid_size)) 

tmp_file="tmp_nvprof.log"
output_file="nvprof_results.csv"

output=$(nvprof --print-gpu-trace --csv --log-file "$tmp_file" ./"$1" "$size" "$grid_size" "$block_size" "${args[@]}")
sum=$(awk -F',' 'NR>2 {sum+=$12} END {print sum}' "$tmp_file")
echo "$sum" 


output=$(nvprof --trace gpu --csv --log-file "$tmp_file" ./"$1" "$size" "$grid_size" "$block_size" "${args[@]}")
avg_time=$(awk -F',' '/"GPU activities"/ {print $6; exit}' "$tmp_file")


echo "$avg_time" 

echo "$output"
echo "$sum"

