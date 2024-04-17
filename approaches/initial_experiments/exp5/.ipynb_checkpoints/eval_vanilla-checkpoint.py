from evaluate import load
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset, DatasetDict, Dataset
import torch
from transformers import AutoModelForCausalLM
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import os
import re
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pdb
import pandas as pd

def str_or_list(val):
    if re.search(r"^\[",val):
        sep_list = val.strip("[]").split(',')
        return sep_list
    return [val]

def tokenize(element):
    
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=args.max_new_tokens,
        return_overflowing_tokens=True,
        return_length=True,
        padding=True,
        # stride=args.stride,
        return_tensors="pt"
    )
    
    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {(outputs['length'])}")
    # print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
    # print(f"attention mask :\n {outputs['attention_mask']}")

    token_data = []
    for input_ids in outputs["input_ids"]:
        # print(input_ids)
        token_data.append(input_ids)

    # print(token_data)
    # print("example :")
    # print(token_data[0])

#     input_batch = [stride*[tokenizer.pad_token_id] + token_batch[0][:stride]]
#     input_batch += token_batch

#     output_batch = [token_batch[-1][-1*stride:]] + stride*[tokenizer.pad_token_id]]
    
    input_data = token_data[:-1]
    output_data = token_data[1:]
    
    # print("input_batch")
    # print(input_batch)
    # print("padded_batch")
    # print(padded_batch)
    # print(input_batch[0])
    return {"input" : input_data, "output" : output_data}


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Model to evaluate", required=True, type=str)
    parser.add_argument("--tk_name", help="tokenizer name", required=True, type=str)
    parser.add_argument("--data", help="Add testing data files (single file name or list fo files in the format : [a,b,c,...]. The files in the list will be concatenated before being used as training data)", required=True, type=str_or_list)
    parser.add_argument("--output_dir", help="output directory", required=False, type=str, default="./")
    parser.add_argument("--load_checkpoint_epoch", help="Continue training from a previous checkpoint", required=False, default = 0, type=int)
    parser.add_argument("--max_new_tokens", help="tokenier name", required=False, type=int, default = 256)
    parser.add_argument("--stride", help="tokenier name", required=False, type=int, default = 256)
    
    batch_size = 1
    context_length = 256
    
    global args
    args = parser.parse_args()
    
    # load_checkpoint = args.load_checkpoint_epoch
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print("loading modules")
    
    tokenizer_name = args.tk_name
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{tokenizer_name}")
    tokenizer.pad_token = tokenizer.eos_token
    # bertscore = load("bertscore")
       
    print("everything loaded")
   
    # accelerator = Accelerator(mixed_precision = mixed_precision)
    # accelerator.print("accelerator initialised")
    
    with open(args.data[0], 'r') as f:
        json_data = json.load(f)
    
    # resp = json_data['respiratory']
    data = []
    for elem in json_data:
        inp = f"Question : \n{elem['q']} \nAnswer : \n"
        out = elem['a']
        data_point = {"text" : inp, "labels" : out}
        data.append(data_point)
        
        # with open("./gold.txt","a") as f:
        #     f.write(f"question : + {inp}")
        #     f.write(f"answer :{out} + \n")
            
    _, data = train_test_split(data, test_size=0.15, random_state=42)
    # data, _ = train_test_split(data, test_size=0.15, random_state=42)
    # data = data[:30]
    # pdb.set_trace()
    
    ######## creating dataframes for saving the results
    
    iodf = pd.DataFrame()
    results_df = pd.DataFrame(index = ['RougeL', 'bertscore'])
    inp = []
    out = []
    for elem in data:
        inp.append(elem['text'])
        out.append(elem['labels'])
        
    iodf['inputs'] = inp
    iodf['gold_outputs'] = out
    ########

    data = Dataset.from_list(data)
    
    dataset = DatasetDict({"test" : data})  
    
    # print(dataset)
    # print("example :")
    # print(dataset['train'][0])

    tokenizer = AutoTokenizer.from_pretrained(args.tk_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(element):
        tokenizer.pad_token = tokenizer.eos_token
        seq = [text + label for text,label in zip(element['text'], element["labels"])]
        text = element['text']

        ctx_ids = tokenizer(
            text,
            return_length=True,
        )
        label_ids = tokenizer(
            seq,
            return_length=True,
            padding=True,
        )

        ctx_batch = []
        input_batch = []
        label_batch = []
        for ctx_id, label_id in zip(ctx_ids["input_ids"], label_ids["input_ids"]):

            ctx_batch.append(ctx_id)
            len_ctx = len(ctx_id)
            # pdb.set_trace()
            inp = label_id.copy()
            label_id[:len_ctx] = [-100] * len_ctx
            input_batch.append(inp)
            label_batch.append(label_id)
            
        # print(f"Number of Input chunks: {len(text_ids['input_ids'])}")
        # print(f"Input chunk lengths: {(text_ids['length'])}")

        return {"ctx_ids" : ctx_batch, "input_ids": input_batch, "label_ids" : label_batch}


    tokenized_datasets = dataset.map(
        tokenize, batched=True, remove_columns=dataset["test"].column_names
    )
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets)

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    
    dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size)
    
    # output_dataloader = DataLoader(output_dataset, batch_size=batch_size)
    # dataloader = accelerator.prepare(dataloader)
    
    print("dataset loaded")
    
    # print("example :")
    # print(input_dataset[0])
    
    print("load model")
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(f"{model_name}")
    
    print("starting evaluation")
    progress_bar = tqdm(range(len(tokenized_datasets['test'])//batch_size))
    
    # for checkpoint in range(10,71,10):

    model = model.to(device)

    model.eval()

    rouge = load('rouge')
    # bertscore = load('bertscore')

    bertscore_metric = load('bertscore')


    step_losses = []
    results_bert = []
    results_rouge = []

    predictions = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        inputs = batch['input_ids']
        labels = batch['label_ids']

        loss = model(inputs, labels = labels).loss
        loss = loss.item()

        step_losses.append(loss)
        ppl = np.exp(loss)
        # print(f"loss : {loss:.6f} , perplexity : + {ppl:.6f} \n")

        # pdb.set_trace()

        gen_in = batch['ctx_ids']
        ctx_len = len(gen_in[0])
        gen_expt_out = batch['label_ids']

        gen_out = model.generate(gen_in, max_new_tokens = 1024 - ctx_len)

        # print(gen_expt_out)
        # preds = tokenizer.batch_decode(gen_out, skip_special_tokens = True)
        # gold = tokenizer.batch_decode(gen_expt_out, skip_special_tokens = True)


        gen_expt_out = gen_expt_out[:,ctx_len:]
        gen_out = gen_out[:,ctx_len:]

        # ques = tokenizer.batch_decode(gen_in, skip_special_tokens = True)
        preds = tokenizer.batch_decode(gen_out, skip_special_tokens = True)
        gold = tokenizer.batch_decode(gen_expt_out, skip_special_tokens = True)

        preds = [' '.join(preds[0].split())]
        if preds == '':
            preds = ' '
        predictions.append(preds[0])

        print("predicted output :\n", preds)
        # print("gold output :\n", gold)

        # with open("./main_model_qa.txt", "a") as f:
        #     f.write(f"context : \n {ques[0]} \n")
        #     f.write(f"predicted output :\n {' '.join(preds[0].split())} \n")
        #     f.write(f"gold output :\n {gold[0]} \n")

        # print("input :", ''.join(inputs), "\n")
        # print("preds :", preds[0], "\n")
        # print("gold :", gold[0], "\n")

        results = rouge.compute(predictions=preds, references = gold)
        results_rouge.append(results['rougeL'])

        results = bertscore_metric.compute(predictions=preds, references=gold,
                        lang="en",rescale_with_baseline = True,
                        model_type = 'microsoft/deberta-xlarge-mnli')
        results_bert.append(results['f1'][0])
        # pdb.set_trace()
        print(f"rougeL : {results_rouge[-1]} , bertscore : {results_bert[-1]}")

        progress_bar.update(1)

    mean_rouge = sum(results_rouge)/len(results_rouge)
    mean_bert = sum(results_bert)/len(results_bert)

    print(f"mean_rougeL : {mean_rouge} , mean_bertscore : {mean_bert}")

    iodf[f'model_outputs'] = predictions
    results_df[f'model_scores'] = [mean_rouge,mean_bert]
        
    iodf.to_csv(f"{args.output_dir}exp5_vanilla_io.csv")
    results_df.to_csv(f"{args.output_dir}exp5_vanilla_scores.csv")

#         # results = bertscore.compute(predictions=preds, references = gold, lang = 'en')

#         print(f"loss : {loss:.6f} , perplexity : {ppl:.6f} , rougeL : {results_rouge[-1]:.6f} , bertscore : {results_bert[-1]} , epoch : {args.load_checkpoint_epoch} \n")
#         with open(f"{args.output_dir}resp_finetune_baseline_eval.txt","a") as f:
#             f.write(f"loss : {loss:.6f} , perplexity : {ppl:.6f} , rougeL : {results_rouge[-1]:.6f} , bertscore : {results_bert[-1]} , epoch : {args.load_checkpoint_epoch} \n")

        
#     mean_rougeL = sum(results_rouge)/len(results_rouge)
#     mean_bertscore = sum(results_bert)/len(results_bert)
#     mean_loss = sum(step_losses)/len(step_losses)
#     mean_ppl = np.exp(mean_loss)

#     with open(f"{args.output_dir}resp_finetune_baseline_eval.txt","a") as f:
#         f.write(f"mean loss : {mean_loss:.6f} , mean perplexity : {mean_ppl:.6f} , mean rougeL : {mean_rougeL:.6f} , mean bertscore : {mean_bertscore} , epoch : {args.load_checkpoint_epoch} \n")
    
    # print("bert f1 - avg : ", sum(results_bert)/len(results_bert))
    
    
    # {args.output_dir}/{model_name}_medicine_epoch{epoch+1}.pth
    # with open(args.output_dir + "eval_results.txt","w") as f:
    #     f.write("bert_score results f1 : \n")
    #     f.write(str(results_bert))
    #     f.write("Average : \n")
    #     f.write(str(sum(results_bert)/len(results_bert)))
        
    
# print("bert\n", results_acc)

if __name__ == "__main__":
    main()