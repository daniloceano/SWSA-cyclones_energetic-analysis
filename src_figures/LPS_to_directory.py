# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    LPS_to_directory.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/06/28 12:25:38 by Danilo            #+#    #+#              #
#    Updated: 2023/06/28 12:50:38 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import shutil

source_directory = "../"
destination_directory = "../figures/LPS/"

# Filter the directories to include only those containing "LEC_results"
lec_directories = [dir_name for dir_name in os.listdir(source_directory) if "LEC_results" in dir_name]

# Iterate over the filtered directories
for top_dir in lec_directories:
    top_dir_path = os.path.join(source_directory, top_dir)
    if os.path.isdir(top_dir_path):
        
        # Iterate over the distinct directories inside the top-level directory
        for sub_dir in os.listdir(top_dir_path):
            sub_dir_path = os.path.join(top_dir_path, sub_dir)
            
            # Check if the sub directory matches the expected pattern
            if sub_dir.startswith("RG") and "-q" in sub_dir:
                
                # Extract the {quantile} and {RG} patterns from the sub directory name
                quantile, rest = sub_dir.split("-q")
                RG = quantile[2:]  # Remove "RG" prefix
                quantile, file_id = rest.split("_")[0].split('-')  # Extract quantile value
                
                # Construct the destination directory path
                destination_dir = os.path.join(destination_directory, f"q{quantile}")
                
                # Create the destination directory if it doesn't exist
                os.makedirs(destination_dir, exist_ok=True)
                
                # Create subdirectories for each file
                for file in ["LPS_1H.png", "LPS_1H_zoom.png", "LPS_periods.png", "LPS_periods_zoom.png"]:
                    dir_name = os.path.splitext(file)[0]  # Remove file extension
                    file_directory = os.path.join(destination_directory, f"q{quantile}", dir_name)
                    os.makedirs(file_directory, exist_ok=True)
                    
                    # Copy the file to the respective subdirectory
                    source_file = os.path.join(sub_dir_path, "Figures", "LPS", file)
                    destination_file = os.path.join(file_directory, f"RG{RG}-{file_id}.png")
                    shutil.copy(source_file, destination_file)

print("Copying files completed!")
