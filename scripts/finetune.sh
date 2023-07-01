tokenizer=gpt2-xl
model_name=gpt2-xl
output_dir=../model/qa_models/${model_name}-medicine-exp3/
data=../extracted_text/short_cases_medicine/short_cases_medicine_annotated.json
# test_data=""
num_gpus=1
pretrained=70
pretrained_path=../model/pretrained_models/${model_name}-medicine/resp_endo/
checkpoint=50

mkdir -p "${output_dir}"

cmd="accelerate launch --num_processes=${num_gpus} finetune.py --model_name ${model_name} --tk_name ${tokenizer} --data ${data} --output_dir ${output_dir}"

# jbsub -q x86_24h -mem 96G -cores 1x8+${num_gpus} -require a100_80gb ${cmd}
$cmd

echo $cmd >> ${output_dir}log.txt