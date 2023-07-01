#!/bin/bash

filename="Harrison"
input_file="../data/medicine_books/${filename}.pdf"
start_pages="[7842,8162]"
end_pages="[11500,11915]"
top_k=2
output_file_folder="../extracted_text/${filename}/"
output_file_name="${filename}_top_${top_k}.txt"

cmd="python data_extract.py --input_file ${input_file} --output_file ${output_file_folder}${output_file_name} --start_pages ${start_pages} --end_pages ${end_pages} --top_k ${top_k}"

$cmd

echo "${cmd}" >> ../extracted_text/Harrison/log.txt
# echo $cmd >> ${output_file_folder}log.txt

# log="input_file=${!input_file}
# output_file=${!output_file_folder}${!output_file_name}
# start_pages=${start_pages}
# end_pages=${end_pages}
# top_k=${top_k}
# "