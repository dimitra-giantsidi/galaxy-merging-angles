#!/bin/bash

box="rogue"
num="5"
numpix="8000"
threshold=0.8

echo "Applying Segment"
sd_path="/Users/dimitragiantsidi/Desktop/MAP_all/MAP/substructure_detection/sd_outputs"
twod_exp_fits_path="$sd_path""/twod_exp_fit"
seg_path="$sd_path""/Segment"
mkdir -p "$seg_path"
galaxy_path="$seg_path""/$box""_""$num"
mkdir -p "$galaxy_path"
masks_path="$galaxy_path""/masks"
mkdir -p "$masks_path"
segmented_path="$galaxy_path""/segmented"
mkdir -p "$segmented_path"
for file in "$twod_exp_fits_path"/"$box"_"$num"/fits_files/*; do
    if [ -f "$file" ]; then
        echo $(basename "$file")
        numbers=()
        while IFS= read -r line; do
            numbers+=("$line")
        done < <(echo $(basename "$file") | grep -E -o '[0-9]+(\.[0-9]+)?')
        # readarray -t numbers < <(echo $(basename "$file") | grep -E -o '[0-9]+(\.[0-9]+)?')

        angles=("${numbers[@]:1:${#numbers[@]}-2}")
        dec=${angles[0]}
        azi=${angles[1]}
        
        astarithmetic "$twod_exp_fits_path"/"$box"_"$num"/fits_files/$(basename "$file") --hdu=0 $threshold  gt --output="$masks_path"/mask_"$box"_"$num"_d="$dec"_a="$azi"_pix="$numpix".fits

        astsegment "$twod_exp_fits_path"/"$box"_"$num"/fits_files/$(basename "$file") --hdu=0 --detection="$masks_path"/mask_"$box"_"$num"_d="$dec"_a="$azi"_pix="$numpix".fits --dhdu=1 --std=1 --clumpsnthresh=3 --output="$segmented_path"/segmented_"$box"_"$num"_d="$dec"_a="$azi"_pix="$numpix".fits
    fi
done

echo "$box""_$num"" Segment parameters: clumpsnthresh = $threshold" > "$galaxy_path""/parameters.txt"

echo "Segment finished (this doesn't guarantee it worked)."

    # echo "Applying stream fitting"
    # stream_fitting="/home/giantsid/MAP/stream_analysis/stream_fitting.ipynb"

    # papermill "$stream_fitting" "$stream_fitting"

    # if [ $? -eq 0 ]; then
    #     echo "Success on applying stream fitting!"
    # else
    #     echo "Failure on applying stream fitting."
    #     exit 1
    # fi

    # echo "Opening visualizer"
    # visualizer="/home/giantsid/MAP/code/visualizer.ipynb"

    # papermill "$visualizer" "$visualizer"

    # if [ $? -eq 0 ]; then
    #     echo "Success on opening visualizer!"
    # else
    #     echo "Failure on opening visualizer."
    #     exit 1
    # fi

    # jupyter notebook "$visualizer"

echo "All done!"
