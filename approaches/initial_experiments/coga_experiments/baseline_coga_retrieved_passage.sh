model_name=google/flan-t5-xl
tokenizer=google/flan-t5-xl
data_path=../../extracted_text/chatgpt_qa/chatgpt_qa_resp.json
dataset="medicine"
# task="short_"
retrieval_method="similarity"
output_dir=./baseline_results/${retrieval_method}/
model_path=/dccstor/yatinccc/projects/checkpoints/coga-5.6.2/global_step20000_hf/
num_gpus=1
max_new_tokens=200
vectorstore="chromadb"
persistent_path=./${vectorstore}/
emb_model=sentence-transformers/all-MiniLM-L6-v2
retriever_level=2
# checkpoint=10

rm -r ${persistent_path}

mkdir -p ${output_dir}
mkdir -p ${persistent_path}
cmd="python baseline_coga_retrieved_passage.py --model_name ${model_name} --tk_name ${tokenizer} --model_path ${model_path} --emb_model ${emb_model} --data ${data_path} --output_dir ${output_dir} --vectorstore ${vectorstore} --persistent_path ${persistent_path} --retrieval_method ${retrieval_method} --max_new_tokens ${max_new_tokens} --retriever_level ${retriever_level}"

$cmd
# jbsub -q x86_24h -mem 96G -cores 1x8+${num_gpus} -require a100_80gb ${cmd}

echo $cmd >> ${output_dir}log.txt