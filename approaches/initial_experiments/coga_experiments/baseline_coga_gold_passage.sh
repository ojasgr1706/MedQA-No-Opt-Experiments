model_name=google/flan-t5-xl
tokenizer=google/flan-t5-xl
data_path=../../extracted_text/chatgpt_qa/chatgpt_qa_resp.json
dataset="medicine"
# task="short_"
output_dir=./baseline_results/
model_path=/dccstor/yatinccc/projects/checkpoints/coga-5.6.2/global_step20000_hf/
num_gpus=1
max_new_tokens=200
# checkpoint=10

mkdir -p ${output_dir}
cmd="python baseline_coga_gold_passage.py --model_name ${model_name} --tk_name ${tokenizer} --model_path ${model_path} --data ${data_path} --output_dir ${output_dir} --max_new_tokens ${max_new_tokens}"

$cmd

# jbsub -q x86_24h -mem 96G -cores 1x8+${num_gpus} -require a100_80gb ${cmd}

echo $cmd >> ${output_dir}log.txt