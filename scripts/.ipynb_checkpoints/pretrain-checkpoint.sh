tokenizer=gpt2-xl
model_name=gpt2-xl
output_dir=../model/pretrained_models/${model_name}-medicine/resp/
train_data=["../extracted_text/kumar_and_clark/kumar_and_clark_respiratory_top_3.txt","../extracted_text/Harrison/Harrison_respiratory_top_3.txt"]
test_data=""
num_gpus=1
checkpoint=80


mkdir -p "${output_dir}"
cmd="accelerate launch --num_processes=${num_gpus} pretrain.py --model_name ${model_name} --tk_name ${tokenizer} --train_data ${train_data} --output_dir ${output_dir} --load_checkpoint_epoch ${checkpoint}"

# jbsub -q x86_24h -mem 96G -cores 1x8+${num_gpus} -require a100_80gb ${cmd}
$cmd

echo $cmd >> ${output_dir}log.txt