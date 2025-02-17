#!/bin/bash


if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <executable> <arg1> <arg2> <arg3> <arg4> <arg5> ..."
    exit 1
fi

args=("${@:2}")

size=$(($# - 1))
grid_size=1
block_size=$(((size + grid_size - 1) / grid_size)) 

echo 'nvprof ./"$1" "$size" "$grid_size" "$block_size" "${args[@]}"'
output=$(nvprof ./"$1" "$size" "$grid_size" "$block_size" "${args[@]}")


echo "$output"

