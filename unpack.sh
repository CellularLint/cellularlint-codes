#!/bin/bash

# Set the directory name here
directory="Pretrained Models"

cd "$directory" || { echo "Directory not found: $directory"; exit 1; }

for zip_file in *.zip; 
do 
  #Name without .zip extension
  base_name="${zip_file%.zip}"

  mkdir -p "$base_name"

  unzip -q "$zip_file" -d "$base_name"

  echo "Unzipped $zip_file into $base_name"
done

