model_name=gpt2-xl
tokenizer=gpt2-xl
data_path=../extracted_text/short_cases_medicine/short_cases_medicine_annotated.json
dataset="medicine"
# task="short_"
output_dir=../model/qa_models/${model_name}-medicine-exp4/
num_gpus=1
checkpoint=20

cmd="python eval_finetune_multi.py --model_name ${model_name} --tk_name ${tokenizer} --data ${data_path} --output_dir ${output_dir} --load_checkpoint_epoch ${checkpoint}"

$cmd

# jbsub -q x86_24h -mem 96G -cores 1x8+${num_gpus} -require a100_80gb ${cmd}

echo $cmd >> ${output_dir}log.txt