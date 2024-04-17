model_name=gpt2-xl
tokenizer=gpt2-xl
data_path=../../extracted_text/chatgpt_qa/chatgpt_qa_resp.json
dataset="medicine"
# task="short_"
output_dir=../../model/exp5/finetuned_corrected/
num_gpus=1
# checkpoint=10

cmd="python eval_finetuned.py --model_name ${model_name} --tk_name ${tokenizer} --data ${data_path} --output_dir ${output_dir}"

$cmd

# jbsub -q x86_24h -mem 96G -cores 1x8+${num_gpus} -require a100_80gb ${cmd}

echo $cmd >> ${output_dir}log.txt