#!/bin/bash


n_values=(1600 3200 6400 12800)


for n in "${n_values[@]}"
do
    echo "Running Scalene for n=$n"
    scalene --memory --json --cli --outfile scalene/scalene_profile_n${n}.json fft_convolution.py $n 
done