#!/bin/bash

sizes=(16 32 64 128 256 512 1024 2048 4096 8192 16284)
maxNumBlocksPerGrid=58
maxNumThreadsPerBlock=1024

rep=10

for size in "${sizes[@]}"; do 
    for ((grid_size=1; grid_size<=size; grid_size++)); do 
            block_size=$(((size+grid_size)/grid_size)) 
        
            sum_init_clock=0.0
            sum_add_clock=0.0
            
            for ((it=0; it<rep; it++)); do 
                output=$("./$1" "$size" "$grid_size" "$block_size") 
                init_time=$(echo "$output" | grep "Initialization Time (clock)" | awk '{print $4}')
                addition_time=$(echo "$output" | grep "Addition Time (clock)" | awk '{print $4}')
                
                # Accumulate times
                sum_add_clock=$(echo "$sum_add_clock + $addition_time" | bc -l)
                sum_init_clock=$(echo "$sum_init_clock + $init_time" | bc -l)
            done 
            
            # Compute means
            mean_init_time=$(echo "$sum_init_clock/$rep" | bc -l)
            mean_add_time=$(echo "$sum_add_clock/$rep" | bc -l)
            
            # Output results
            echo "$size $grid_size $block_size $mean_init_time $mean_add_time"
        
    done 
done
