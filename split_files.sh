#!/bin/bash
srcdir=/home/karyo/corpus/data/mc4_jsonl    # Source directory where the files are
destdir=/home/karyo/corpus/data/mc4_jsonl_split  # Destination base directory where the folders will be created

filecount=0
dircount=0
files_per_dir=8

# Create the first directory
mkdir -p "$destdir/mc4_jsonl_split$dircount"

for file in "$srcdir"/*; do
  if ((filecount == files_per_dir)); then
    ((dircount++))
    mkdir -p "$destdir/mc4_jsonl_split$dircount"
    filecount=0
  fi

  cp "$file" "$destdir/mc4_jsonl_split$dircount"
  ((filecount++))
done