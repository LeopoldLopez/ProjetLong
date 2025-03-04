#!/bin/bash

# Vérifie si un argument est fourni
if [ -z "$1" ]; then
    echo "Usage: $0 <nombre_de_requetes>"
    exit 1
fi

num_requests=$1

for i in $(seq 1 "$num_requests"); do
    # Si on veut laisser les terminaux ouverts
    # gnome-terminal -- bash -c "./client sum $(( i * 1000 )); exec bash" &

    size=$(( i * 1000 )) 

    gnome-terminal -- bash -c "./client sum $size 2>&1 | tee -a output$i.txt" &
done


wait