#!/bin/bash

# Find directories with the specified pattern
find . -type d -name "*_ERA5_track-15x15" | while read -r dir; do
    # Extract the directory name
    dir_name=$(basename "$dir")
    
    mv $dir/$dir_name.csv ${dir}/10MostIntense-${dir_name}.csv     
    mv $dir/${dir_name}_track ${dir}/10MostIntense-${dir_name}_track
    
    mv $dir_name 10MostIntense-$dir_name

done

echo "File renaming complete."
