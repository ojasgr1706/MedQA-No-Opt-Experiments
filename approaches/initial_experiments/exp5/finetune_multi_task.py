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

def training_loop(args, tok_len, dataset, mixed_precision="fp16"):
    
    # model_name = "bloom-1b1"
    model_name = args.model_name
    
    accelerator = Accelerator(mixed_precision = mixed_precision)
    accelerator.print("accelerator initialised")
    
    set_seed(42)
    accelerator.print("seed set")
    model = AutoModelForCausalLM.from_pretrained(f"{model_name}")
    model.resize_token_embeddings(tok_len)
    accelerator.print("model loaded")
    
    # HYPERPARAMETERS
    
    num_epochs = 100
    warm_up_steps = 100
    lr = 5e-6
    train_batch_size = 1
    test_batch_size = 8
    checkpoint = True
    evaluate = False

    optimizer = AdamW(model.parameters(), lr=lr)
    
    qa_dataset = dataset["qa"]
    text_dataset = dataset["text"]

    qa_train_dataloader = DataLoader(qa_dataset['train'], shuffle=True, batch_size=train_batch_size)
    qa_test_dataloader = DataLoader(qa_dataset['test'],shuffle=False, batch_size=test_batch_size)
    text_train_dataloader = DataLoader(text_dataset['train'], shuffle=False, batch_size=train_batch_size)
    qa_train_dataloader, qa_test_dataloader, text_train_dataloader, model, optimizer = accelerator.prepare(
        qa_train_dataloader, qa_test_dataloader, text_train_dataloader, model, optimizer
    )
            
    accelerator.print("dataloaders initialised")

    training_steps = num_epochs * (len(qa_train_dataloader) + len(text_train_dataloader)) - warm_up_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warm_up_steps, num_training_steps=training_steps
    )
    
    accelerator.print("scheduler initialised")
    
    
    # Training conditions

    if args.pretrained_checkpoint:
        Path = f'{args.pretrained_path}{model_name}_resp_epoch{args.pretrained_checkpoint}.pth'
        ckpt_dict = torch.load(Path)
        model.load_state_dict(ckpt_dict['model_state_dict'])
    
    if args.load_checkpoint_epoch:
        Path = f'{args.output_dir}{model_name}_finetuned_exp5_epoch{args.load_checkpoint_epoch}.pth'
        ckpt_dict = torch.load(Path)
        model.load_state_dict(ckpt_dict['model_state_dict'])
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])

        print("checkpoint loaded")
        
    def causallm_loss(inputs, logits):
        stride=256
        
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

    progress_bar = tqdm(range(training_steps))
    epoch_losses = []
    best = 1
    best_val_loss = float("inf")
    
    accelerator.print("training started")
    for epoch in range(args.load_checkpoint_epoch,args.load_checkpoint_epoch + num_epochs):
        model.train()

        text_step_losses = []
        for step,batch in enumerate(text_train_dataloader, start = 1):
            # batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch['input_ids']).logits
            loss = causallm_loss(batch['input_ids'],logits)
            # loss.backward()
            accelerator.backward(loss)
            text_step_losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if(step%10 == 0):
                progress_bar.update(10)

        epoch_loss = sum(text_step_losses)/len(text_step_losses)
        epoch_losses.append(epoch_loss)
        ppl = np.exp(epoch_loss)
        with open(f"{args.output_dir}exp5_multi_task_finetuning_logs.txt","a") as f:
            f.write(f"Text completion | loss : {epoch_loss:.3f} , perplexity : {ppl:.3f} , epoch : {epoch} \n")
        
        qa_step_losses = []
        for step,batch in enumerate(qa_train_dataloader, start = 1):
           
            inputs = batch['input_ids']
            labels = batch['label_ids']
            attention_mask = batch['attention_mask']

            loss = model(inputs, labels = labels, attention_mask = attention_mask).loss
            # loss.backward()
            accelerator.backward(loss)
            qa_step_losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        epoch_loss = sum(qa_step_losses)/len(qa_step_losses)
        epoch_losses.append(epoch_loss)
        ppl = np.exp(epoch_loss)

        with open(f"{args.output_dir}exp5_multi_task_finetuning_logs.txt","a") as f:
            f.write(f"QA | loss : {epoch_loss:.3f} , perplexity : {ppl:.3f} , epoch : {epoch} \n")
            
        print(f"loss : {epoch_loss:.3f} , perplexity : + {ppl:.3f} \n")
        
        if(checkpoint and (epoch+1)%10 == 0):
            print("testing")
            model.eval()
            test_losses = []
            for step,test_batch in enumerate(qa_test_dataloader, start = 1):
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
                
            with open(f"{args.output_dir}exp5_multi_task_finetuning_logs.txt","a") as f:
                f.write(f"QA | test_loss : {epoch_test_loss:.3f} , test_perplexity : + {test_ppl:.3f}, learning_rate : {lr} \n")
                
            print(f"QA | test_loss : {epoch_test_loss:.3f} , test_perplexity : + {test_ppl:.3f} \n")

            ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            }
            
            Path = f'{args.output_dir}/{model_name}_multi_task_finetuned_exp5_epoch{epoch+1}.pth'
            
            torch.save(ckpt_dict,Path)
        
                
    accelerator.print("evaluation ended")
    # accelerator.print(epoch_losses)
    with open(f"{args.output_dir}exp5_multi_task_finetuning_logs.txt","a") as f:
        f.write(f"best = {best}\n")
    # torch.save(model.state_dict(),f'../model/trained_models/{model_name}_harrison_respiratory.pth')
    # accelerator.print("best saved")

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
    
    text = ""
    last = ""
    
    data = []
    for elem in json_data:
        inp = f"Question : \n{elem['q']} \nAnswer : \n"
        out = elem['a']
        data_point = {"text" : inp, "labels" : out}
        data.append(data_point)
        
        if last != elem['psg']:
            text += elem['psg'] + '\n '
            last = elem['psg']
            
    # with open("./text_data_test.txt", 'w') as f:
    #     f.write(text[:2500])

    
    # print(data[:5])
    
    qa_train, qa_test = train_test_split(data, test_size=0.15, random_state=42)

    qa_train = Dataset.from_list(qa_train)
    qa_test = Dataset.from_list(qa_test)
    
    qa_dataset = DatasetDict({"train" : qa_train, "test" : qa_test})
    
    text_dict = [{'text' : text}]
    text_data = Dataset.from_list(text_dict)
    text_dataset = DatasetDict({"train" : text_data})
    
    print(qa_dataset)
    print(text_dataset)
    # print("example :")
    # print(dataset['train'][0])

    tokenizer = AutoTokenizer.from_pretrained(args.tk_name)
    
    text_token = '[T]'
    qa_token = '[Q]'
    
    tokenizer.add_tokens([text_token,qa_token], special_tokens = True)
    
    
    def qa_tokenize(element):
        
        tokenizer.pad_token = tokenizer.eos_token
        
        text = element['text']
        
        for i in range(len(text)):
            text[i] = qa_token + " " + text[i]
        
        seq = [txt + label for txt,label in zip(text, element["labels"])]
        
        # pdb.set_trace()
        # print(seq)

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
        attention_mask_batch = []
        for ctx_id, label_id, attention_mask in zip(ctx_ids["input_ids"], label_ids["input_ids"], label_ids['attention_mask']):

            len_ctx = len(ctx_id)
            # pdb.set_trace()
            inp = label_id.copy()
            label_id[:len_ctx] = [-100] * len_ctx
            input_batch.append(inp)
            label_batch.append(label_id)
            attention_mask_batch.append(attention_mask)
            
        # print(f"Number of Input chunks: {len(text_ids['input_ids'])}")
        # print(f"Input chunk lengths: {(text_ids['length'])}")
        
        # pdb.set_trace()

        return {"input_ids": input_batch, "label_ids" : label_batch, 'attention_mask' : attention_mask_batch}
    
    def text_tokenize(element):
        context_length = 512
        stride = 256
            
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
        # print("input_batch")
        # print(input_batch)
        # print("padded_batch")
        # print(padded_batch)
        # print(input_batch[0])
        return {"input_ids": padded_batch}


    qa_tokenized_datasets = qa_dataset.map(
        qa_tokenize, batched=True, remove_columns=qa_dataset["train"].column_names
    )
    qa_tokenized_datasets.set_format("torch")
    print("qa data")
    print(qa_tokenized_datasets)

    text_tokenized_datasets = text_dataset.map(
        text_tokenize, batched=True, remove_columns=text_dataset["train"].column_names
    )
    text_tokenized_datasets.set_format("torch")
    print("text data")
    print(text_tokenized_datasets)

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    
    training_loop(args, len(tokenizer), {"qa":qa_tokenized_datasets, "text":text_tokenized_datasets})
    
if __name__ == "__main__":
    main()