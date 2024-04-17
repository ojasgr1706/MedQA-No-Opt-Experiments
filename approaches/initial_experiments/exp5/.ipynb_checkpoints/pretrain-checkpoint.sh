tokenizer=gpt2-xl
model_name=gpt2-xl
output_dir=../../model/exp5/pretrained/
train_data=../../extracted_text/chatgpt_qa/chatgpt_qa_resp.json
test_data=""
num_gpus=2
checkpoint=80


mkdir -p "${output_dir}"
cmd="accelerate launch --num_processes=${num_gpus} pretrain.py --model_name ${model_name} --tk_name ${tokenizer} --train_data ${train_data} --output_dir ${output_dir}"

# jbsub -q x86_24h -mem 96G -cores 1x8+${num_gpus} -require a100_80gb ${cmd}
$cmd

echo $cmd >> ${output_dir}log.txt