#!/bin/bash

gcc ./server.c -pthread -o server
gcc ./sumCPU.c -o sumCPU
nvcc ./sum.cu -o sum
