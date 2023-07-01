# import evaluate
# import argparse
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from torch.utils.data import DataLoader
# import re
# from datasets import load_dataset, DatasetDict
# import torch
# from tqdm.auto import tqdm

# def str_or_list(val):
#     if re.search(r"^\[",val):
#         sep_list = val.strip("[]").split(',')
#         return sep_list
#     return [val]

# def tokenize(element):
    
#     outputs = tokenizer(
#         element["text"],
#         truncation=True,
#         max_length=args.max_new_tokens,
#         return_overflowing_tokens=True,
#         return_length=True,
#         padding=True,
#         # stride=args.stride,
#         return_tensors="pt"
#     )
    
#     print(f"Input IDs length: {len(outputs['input_ids'])}")
#     print(f"Input chunk lengths: {(outputs['length'])}")
#     # print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
#     # print(f"attention mask :\n {outputs['attention_mask']}")

#     token_data = []
#     for input_ids in outputs["input_ids"]:
#         # print(input_ids)
#         token_data.append(input_ids)

#     # print(token_data)
#     print("example :")
#     print(token_data[0])

# #     input_batch = [stride*[tokenizer.pad_token_id] + token_batch[0][:stride]]
# #     input_batch += token_batch

# #     output_batch = [token_batch[-1][-1*stride:]] + stride*[tokenizer.pad_token_id]]
    
#     input_data = token_data[:-1]
#     output_data = token_data[1:]
    
#     # print("input_batch")
#     # print(input_batch)
#     # print("padded_batch")
#     # print(padded_batch)
#     # print(input_batch[0])
#     return {"input" : input_data, "output" : output_data}


# def main():
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", help="Model to evaluate", required=True, type=str)
#     parser.add_argument("--tk_name", help="tokenizer name", required=True, type=str)
#     parser.add_argument("--data", help="Add testing data files (single file name or list fo files in the format : [a,b,c,...]. The files in the list will be concatenated before being used as training data)", required=True, type=str_or_list)
#     parser.add_argument("--output_dir", help="output directory", required=False, type=str, default="./")
#     parser.add_argument("--max_new_tokens", help="tokenier name", required=False, type=int, default = 256)
#     parser.add_argument("--stride", help="tokenier name", required=False, type=int, default = 256)
    
#     batch_size = 32
#     context_length = 256
    
#     global args
#     args = parser.parse_args()
    
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
#     print("loading modules")
    
#     tokenizer_name = args.tk_name
#     global tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(f"{tokenizer_name}")
#     tokenizer.pad_token = tokenizer.eos_token
#     rouge = evaluate.load('rouge')
       
#     print("everything loaded")
   
#     # accelerator = Accelerator(mixed_precision = mixed_precision)
#     # accelerator.print("accelerator initialised")
    
#     dataset = load_dataset('text',data_files={'test': args.data})
    
#     print("loading dataset")
#     tokenized_dataset = dataset.map(
#         tokenize, batched=True, remove_columns=dataset["test"].column_names
#     )
#     tokenized_dataset.set_format("torch")
    
#     print(tokenized_dataset)
#     input_dataset = tokenized_dataset['test']['input']
#     output_dataset = tokenized_dataset['test']['output']
    
#     input_dataloader = DataLoader(input_dataset, batch_size=batch_size)
#     output_dataloader = DataLoader(output_dataset, batch_size=batch_size)
#     # dataloader = accelerator.prepare(dataloader)
    
#     print("dataset loaded")
    
#     # print("example :")
#     # print(input_dataset[0])
    
#     print("load model")
#     model_name = args.model_name
#     model = AutoModelForCausalLM.from_pretrained(f"{model_name}")
#     model = model.to(device)
    
#     print("starting evaluation")
    
#     progress_bar = tqdm(range(len(input_dataset)//batch_size))

#     results_rouge = []
#     output_it = iter(output_dataloader)
#     for batch in input_dataloader:
#         # print(batch)
#         # print(batch.item())
#         batch = batch.to(device)
#         # batch = {k: v.to(device) for k, v in batch.item()}
#         out = model.generate(batch, max_new_tokens = args.max_new_tokens)
#         out = out[:,context_length:]

#         # targets = output_dataloader[i]
#         targets = next(output_it)
        
#         preds = tokenizer.batch_decode(out, skip_special_tokens = True)
#         gold = tokenizer.batch_decode(targets, skip_special_tokens = True)
        
#         # print("input :", ''.join(inputs), "\n")
#         # print("preds :", preds[0], "\n")
#         # print("gold :", gold[0], "\n")
        
#         results = rouge.compute(predictions=preds, references = gold)
#         results_rouge.append(results['rougeL'])
#         print(results_rouge[-1])
#         progress_bar.update(1)
    
#     print("rougeL - avg : ", sum(results_rouge)/len(results_rouge))
    
#     with open(args.output_dir + "eval_results.txt","w") as f:
#         f.write(str(results_rouge))
#         f.write(str(sum(results_rouge)/len(results_rouge)))
        
    
# # print("rouge\n", results_acc)

# if __name__ == "__main__":
#     main()



import evaluate
import argparse
from transformers import AutoTokenizer, GPT2Model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
# from accelerate import notebook_launcher
from datasets import load_dataset, DatasetDict
import torch
from transformers import AutoModelForCausalLM
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import os
import re
import numpy as np
import pdb
import pandas as pd


# Accelerate training loop

def evaluation(args, dataset, mixed_precision="fp16"):
    
    # model_name = "bloom-1b1"
    model_name = args.model_name
    
    accelerator = Accelerator(mixed_precision = mixed_precision)
    accelerator.print("accelerator initialised")
    
    set_seed(42)
    accelerator.print("seed set")
    model = AutoModelForCausalLM.from_pretrained(f"{model_name}")
    accelerator.print("model loaded")
        
    # HYPERPARAMETERS

    test_batch_size = 16
    checkpoint = True

    # train_dataloader = DataLoader(dataset['train'], shuffle=False, batch_size=train_batch_size)
    # train_dataloader, model, optimizer = accelerator.prepare(
    #     train_dataloader, model, optimizer
    # )
    
    # if(args.test_data):
    test_dataloader = DataLoader(dataset['test'], batch_size=test_batch_size)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
        
    accelerator.print("dataloader initialised")
    
    # Training conditions    
    
    if args.load_checkpoint_epoch:
        Path = f'{args.output_dir}{model_name}_resp_endo_epoch{args.load_checkpoint_epoch}.pth'
        ckpt_dict = torch.load(Path)
        model.load_state_dict(ckpt_dict['model_state_dict'])
        
    print("checkpoint_loaded")

    progress_bar = tqdm(range(1 + len(test_dataloader)//test_batch_size))
    epoch_losses = []
    
    def causallm_loss(inputs, logits):
        # stride=256
        
        # Shift so that tokens < n predict n
        shift_labels = inputs[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()

        preds = shift_logits.view(-1, shift_logits.size(-1))
        targets = shift_labels.view(-1)
        targets = targets.clone()
        targets[:stride-1] = -100

        # Calculate per-token loss
        loss_fct = CrossEntropyLoss()

        loss = loss_fct(preds, targets)
        # print(loss)
        return loss
    
    model.eval()
    accelerator.print("evaluation started")

    rouge = evaluate.load('rouge')
    # with open(f"{args.output_dir}train_logs.txt","a") as f:
    #         f.write("losses :" + '\n')
            
    step_losses = []
    results_rouge = []
    
    df_outputs = pd.DataFrame(columns = ['gold','predicted'])
    for step,batch in enumerate(test_dataloader, start = 1):
        # batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(batch['input_ids']).logits
            loss = causallm_loss(batch['input_ids'],logits)

        # pdb.set_trace()

        gen_in = batch['input_ids'][:,:stride]
        targets = batch['input_ids'][:,stride:]
        gen_out = model.generate(gen_in,max_new_tokens = stride)
        gen_out = gen_out[:,stride:]

        preds = tokenizer.batch_decode(gen_out, skip_special_tokens = True)
        gold = tokenizer.batch_decode(targets, skip_special_tokens = True)
        
        with open(f"{args.output_dir}generated_outputs.txt","a") as f:
            f.write(f"GOLD : \n{gold[0]}\n")
            f.write(f"PREDICTED : \n{preds[0]}\n\n")
            
        df_outputs.loc[len(df_outputs)] = [gold[0],preds[0]]
        
        # print("input :", ''.join(inputs), "\n")
        # print("preds :", preds[0], "\n")
        # print("gold :", gold[0], "\n")
        
        
        # print("input :", ''.join(inputs), "\n")
        # print("preds :", preds[0], "\n")
        # print("gold :", gold[0], "\n")
        
        results = rouge.compute(predictions=preds, references = gold)
        results_rouge.append(results['rougeL'])

        step_loss = loss.item()
        step_losses.append(step_loss)

        # optimizer.step()
        # lr_scheduler.step()
        # optimizer.zero_grad()
        # progress_bar.update(1)

        # epoch_loss = sum(step_losses)/len(step_losses)
        # epoch_losses.append(epoch_loss)
        ppl = np.exp(step_loss)
        with open(f"{args.output_dir}resp_endo_eval.txt","a") as f:
            f.write(f"loss : {step_loss:.6f} , perplexity : {ppl:.6f} , rougeL : {results_rouge[-1]} , epoch : {args.load_checkpoint_epoch} \n")
            
        progress_bar.update(1)
            
    df_outputs.to_csv(f"{args.output_dir}generated_outputs.csv")

    mean_loss = sum(step_losses)/len(step_losses)
    mean_ppl = np.exp(mean_loss)
    mean_rougeL = sum(results_rouge)/len(results_rouge)

    with open(f"{args.output_dir}resp_endo_eval.txt","a") as f:
        f.write(f"mean_loss : {mean_loss:.6f} , mean perplexity : {mean_ppl:.6f} , mean rougeL : {mean_rougeL:.6f} , epoch : {args.load_checkpoint_epoch} \n")
        
    print(f"mean_loss : {mean_loss:.6f} , mean perplexity : {mean_ppl:.6f} , mean rougeL : {mean_rougeL:.6f} , epoch : {args.load_checkpoint_epoch} \n")
            
                
    accelerator.print("evaluation ended")
        
#     if epoch_loss <= epoch_losses[best-1]:
#         best = epoch - args.load_checkpoint_epoch
#         ckpt_dict = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         }

#         Path = f'{args.output_dir}{model_name}_resp_last.pth'
#         torch.save(ckpt_dict,Path)

def main():
    
    def str_or_list(val):
        if re.search(r"^\[",val):
            sep_list = val.strip("[]").split(',')
            return sep_list
        return [val]

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Add input data files (single file name or list fo files in the format : [a,b,c,...]. The files in the list will be concatenated before being used as training data)", required=True, type=str_or_list)
    parser.add_argument("--test_data", help="Add testing data files (single file name or list fo files in the format : [a,b,c,...]. The files in the list will be concatenated before being used as training data)", required=False, type=str_or_list)
    parser.add_argument("--model_name", help="Model name", required=True, type=str)
    parser.add_argument("--tk_name", help="Tokenizer name", required=True, type=str)
    parser.add_argument("--output_dir", help="Directory to save the trained models and checkpoints", required=False, type=str, default="./")
    parser.add_argument("--load_checkpoint_epoch", help="Load checkpoint or not", required=False, type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset('text',data_files={'test': args.train_data})
            
    # print(dataset)
    # print("example :")
    # print(dataset['train'][0])

    global context_length
    global stride
    context_length = 512
    stride = 256

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tk_name)
    tokenizer.pad_token = tokenizer.eos_token

    # outputs = tokenizer(
    #     dataset["test"][:]["text"],
    #     truncation=True,
    #     # padding=True,
    #     max_length=context_length,
    #     return_overflowing_tokens=True,
    #     stride=stride,
    #     return_length=True,
    #     # padding=True,
    # )

    # print(f"Input IDs length: {len(outputs['input_ids'])}")
    # print(f"Input chunk lengths: {(outputs['length'])}")
    # print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
    # print(f"attention mask :\n {outputs['attention_mask']}")


    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
            padding=True,
            stride=stride
        )
        
        print(f"Input IDs length: {len(outputs['input_ids'])}")
        print(f"Input chunk lengths: {(outputs['length'])}")
        # print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
        # print(f"attention mask :\n {outputs['attention_mask']}")

        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        
        padded_batch = [stride*[tokenizer.pad_token_id] + input_batch[0][:stride]]
        padded_batch += input_batch
        
        return {"input_ids": padded_batch}


    tokenized_datasets = dataset.map(
        tokenize, batched=True, remove_columns=dataset["test"].column_names
    )
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets)

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    
    evaluation(args, tokenized_datasets)
    
if __name__ == "__main__":
    main()