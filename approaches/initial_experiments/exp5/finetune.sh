tokenizer=gpt2-xl
model_name=gpt2-xl
output_dir=../../model/exp5/finetuned_corrected/
data=../../extracted_text/chatgpt_qa/chatgpt_qa_resp.json
# test_data=""
num_gpus=1
pretrained=50
pretrained_path=../../model/exp5/pretrained_attention/
checkpoint=50

mkdir -p "${output_dir}"

cmd="accelerate launch --num_processes=${num_gpus} finetune.py --model_name ${model_name} --tk_name ${tokenizer} --data ${data} --output_dir ${output_dir}"
# --pretrained_checkpoint ${pretrained} --pretrained_path ${pretrained_path}"

# jbsub -q x86_24h -mem 96G -cores 1x8+${num_gpus} -require a100_80gb ${cmd}
$cmd

echo $cmd >> ${output_dir}log.txt