#!/bin/bash

filename="kumar_and_clark"
input_file="../data/medicine_books/${filename}.pdf"
start_pages=489
end_pages=530
top_k=3
output_file_folder="../extracted_text/${filename}/"
output_file_name="${filename}_respiratory_top_${top_k}"

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