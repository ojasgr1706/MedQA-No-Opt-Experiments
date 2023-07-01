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
# Accelerate training loop

def training_loop(args, dataset, mixed_precision="fp16"):
    
    # model_name = "bloom-1b1"
    model_name = args.model_name
    
    accelerator = Accelerator(mixed_precision = mixed_precision)
    accelerator.print("accelerator initialised")
    
    set_seed(42)
    accelerator.print("seed set")
    model = AutoModelForCausalLM.from_pretrained(f"{model_name}")
    accelerator.print("model loaded")
    
    # HYPERPARAMETERS
    
    num_epochs = 70
    warm_up_steps = 100
    lr = 5e-6
    train_batch_size = 2
    test_batch_size = 16
    checkpoint = True
    load_checkpoint = args.load_checkpoint_epoch
    evaluate = False

    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(dataset['test'],shuffle=False, batch_size=test_batch_size)
    train_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, test_dataloader, model, optimizer
    )
            
    accelerator.print("dataloaders initialised")

    training_steps = num_epochs * len(train_dataloader) - warm_up_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warm_up_steps, num_training_steps=training_steps
    )
    
    accelerator.print("scheduler initialised")
    
    
    # Training conditions

    if args.pretrained_checkpoint:
        Path = f'{args.pretrained_path}{model_name}_resp_endo_epoch{args.pretrained_checkpoint}.pth'
        ckpt_dict = torch.load(Path)
        model.load_state_dict(ckpt_dict['model_state_dict'])
    
    if load_checkpoint:
        Path = f'{args.output_dir}{model_name}_finetuned_exp4_epoch{load_checkpoint}.pth'
        ckpt_dict = torch.load(Path)
        model.load_state_dict(ckpt_dict['model_state_dict'])
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])

        print("checkpoint loaded")

    progress_bar = tqdm(range(training_steps))
    epoch_losses = []
    best = 1
    best_val_loss = float("inf")
    
    loss_fct = CrossEntropyLoss(reduction='mean')
    accelerator.print("training started")
    for epoch in range(args.load_checkpoint_epoch,args.load_checkpoint_epoch + num_epochs):
        model.train()
        step_losses = []
        for step,batch in enumerate(train_dataloader, start = 1):
            # batch = {k: v.to(device) for k, v in batch.items()}
            # logits = model(batch['input_ids']).logits
            # loss = causallm_loss(logits,batch)

            # preds = logits.view(-1, logits.size(-1))
            # targets = labels.view(-1)
            
            # loss = loss_fct(preds, targets)
            inputs = batch['input_ids']
            labels = batch['label_ids']

            loss = model(inputs, labels = labels).loss
            # loss.backward()
            accelerator.backward(loss)
            step_losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        epoch_loss = sum(step_losses)/len(step_losses)
        epoch_losses.append(epoch_loss)
        ppl = np.exp(epoch_loss)
        with open(f"{args.output_dir}exp4_finetune_logs.txt","a") as f:
            f.write(f"loss : {epoch_loss:.3f} , perplexity : {ppl:.3f} , epoch : {epoch} \n")
            
        print(f"loss : {epoch_loss:.3f} , perplexity : + {ppl:.3f} \n")
        
        if(checkpoint and (epoch+1)%10 == 0):
            print("testing")
            model.eval()
            test_losses = []
            for step,test_batch in enumerate(test_dataloader, start = 1):
                inputs = test_batch['input_ids']
                labels = test_batch['label_ids']
                
                with torch.no_grad():
                    test_loss = model(inputs, labels = labels).loss
                test_losses.append(test_loss.item())

            epoch_test_loss = sum(test_losses)/len(test_losses)
            test_ppl = np.exp(epoch_test_loss)
            
            if epoch_test_loss <= best_val_loss:
                best = epoch - args.load_checkpoint_epoch
                best_val_loss = epoch_test_loss
                
            with open(f"{args.output_dir}exp4_finetune_logs.txt","a") as f:
                f.write(f"test_loss : {epoch_test_loss:.3f} , test_perplexity : + {test_ppl:.3f}, learning_rate : {lr} \n")
                
            print(f"test_loss : {epoch_test_loss:.3f} , test_perplexity : + {test_ppl:.3f} \n")

            ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            }
            
            Path = f'{args.output_dir}/{model_name}_finetuned_exp4_epoch{epoch+1}.pth'
            
            torch.save(ckpt_dict,Path)
        
                
    accelerator.print("evaluation ended")
    # accelerator.print(epoch_losses)
    with open(f"{args.output_dir}exp4_finetune_logs.txt","a") as f:
        f.write(f"best = {best}\n")
    # torch.save(model.state_dict(),f'../model/trained_models/{model_name}_harrison_respiratory.pth')
    # accelerator.print("best saved")
    

    
# class Args:
#     def __init__(self):
#         self.train_data = ["../extracted_text/kumar_and_clark/kumar_and_clark_top_3.txt","../extracted_text/Harrison/Harrison_top_3.txt"]
#         self.test_data = ""
#         self.model_name = "gpt2-xl"
#         self.output_dir = f"../model/qa_models/{self.model_name}-medicine/"

def str_or_list(val):
    if re.search(r"^\[",val):
        sep_list = val.strip("[]").split(',')
        return sep_list
    return [val]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data file", required=True, type=str)
    parser.add_argument("--model_name", help="Model name", required=True, type=str)
    parser.add_argument("--tk_name", help="tokenizer name", required=True, type=str)
    parser.add_argument("--output_dir", help="Directory to save the trained models and checkpoints", required=False, type=str, default="./")
    parser.add_argument("--load_checkpoint_epoch", help="Continue training from a previous checkpoint", required=False, default = 0, type=int)
    parser.add_argument("--pretrained_checkpoint", help="load pretrained model checkpoint", required=False, default = 0, type=int)
    parser.add_argument("--pretrained_path", help="Path to pretrained model", required=False, default="./", type=str)
    args = parser.parse_args()
    
    with open(args.data, 'r') as f:
        json_data = json.load(f)
    
    resp = json_data['endocrine']
    data = []
    for elem in resp:
        inp = f"Context : \n{elem['med_hist']} \nQuestion : \n{elem['ques']} \nAnswer : \n"
        out = elem['ans']
        data_point = {"text" : inp, "labels" : out}
        data.append(data_point)

    
    # print(data[:5])
    
    train, test = train_test_split(data, test_size=0.15, random_state=42)

    train = Dataset.from_list(train)
    test = Dataset.from_list(test)
    
    dataset = DatasetDict({"train" : train, "test" : test})  
    
    print(dataset)
    # print("example :")
    # print(dataset['train'][0])

    tokenizer = AutoTokenizer.from_pretrained(args.tk_name)
    
    def tokenize(element):
        tokenizer.pad_token = tokenizer.eos_token
        seq = [text + label for text,label in zip(element['text'], element["labels"])]
        # print(seq)
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

        input_batch = []
        label_batch = []
        for ctx_id, label_id in zip(ctx_ids["input_ids"], label_ids["input_ids"]):

            len_ctx = len(ctx_id)
            # pdb.set_trace()
            inp = label_id.copy()
            label_id[:len_ctx] = [-100] * len_ctx
            input_batch.append(inp)
            label_batch.append(label_id)
            
        # print(f"Number of Input chunks: {len(text_ids['input_ids'])}")
        # print(f"Input chunk lengths: {(text_ids['length'])}")
        
        # pdb.set_trace()

        return {"input_ids": input_batch, "label_ids" : label_batch}


    tokenized_datasets = dataset.map(
        tokenize, batched=True, remove_columns=dataset["train"].column_names
    )
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets)

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    
    training_loop(args, tokenized_datasets)
    
if __name__ == "__main__":
    main()