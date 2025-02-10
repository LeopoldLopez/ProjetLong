#!/bin/bash

# Compile the C program
gcc grdLib_builder.c -o grdLib_builder

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."
    ./grdLib_builder
    #gcc -shared -fPIC -o grdLib.so grdLib.c -ldl
    #export LD_PRELOAD=./grdLib.so
else
    echo "Compilation failed. Please check for errors."
fi
