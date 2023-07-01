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
import pandas as pd


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
    
    num_epochs = 30
    warm_up_steps = 100
    lr = 1e-3
    train_batch_size = 4
    test_batch_size = 8
    checkpoint = True
    evaluate = False

    optimizer = AdamW(model.parameters(), lr=lr)
    
    train_dataloader = DataLoader(dataset['train'], shuffle=False, batch_size=train_batch_size)
    val_dataloader = DataLoader(dataset['train'], batch_size=test_batch_size)

    train_dataloader, val_dataloader , model, optimizer = accelerator.prepare(
        train_dataloader, val_dataloader, model, optimizer
    )
        
    accelerator.print("dataloaders initialised")

    training_steps = num_epochs * len(train_dataloader) - warm_up_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warm_up_steps, num_training_steps=training_steps
    )
    
    accelerator.print("scheduler initialised")
    
    # Training conditions    
    
    if args.load_checkpoint_epoch:
        Path = f'{args.output_dir}{model_name}_resp_endo_epoch{args.load_checkpoint_epoch}.pth'
        ckpt_dict = torch.load(Path)
        model.load_state_dict(ckpt_dict['model_state_dict'])
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
        
    print("checkpoint_loaded")

    progress_bar = tqdm(range(training_steps))
    epoch_losses = []
    best = 1
    
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
    
    model.train()
    accelerator.print("training started")
    best_val_loss = float('inf')
            
    for epoch in range(args.load_checkpoint_epoch,args.load_checkpoint_epoch + num_epochs):
        step_losses = []
        for step,batch in enumerate(train_dataloader, start = 1):
            # batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch['input_ids']).logits
            loss = causallm_loss(batch['input_ids'],logits)
            # loss.backward()
            accelerator.backward(loss)
            step_losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if(step%10 == 0):
                progress_bar.update(10)

        epoch_loss = sum(step_losses)/len(step_losses)
        epoch_losses.append(epoch_loss)
        ppl = np.exp(epoch_loss)
        with open(f"{args.output_dir}resp_endo_train_logs.txt","a") as f:
            f.write(f"loss : {epoch_loss:.3f} , perplexity : {ppl:.3f} , epoch : {epoch} \n")

        if(checkpoint and (epoch+1)%10 == 0):

            print("testing")
            model.eval()
            val_losses = []
            for step, val_batch in enumerate(val_dataloader, start = 1):
                
                with torch.no_grad():
                    logits = model(val_batch['input_ids']).logits
                    val_loss = causallm_loss(val_batch['input_ids'],logits)

                val_losses.append(val_loss.item())

            epoch_val_loss = sum(val_losses)/len(val_losses)
            val_ppl = np.exp(epoch_val_loss)
            
            if epoch_val_loss <= best_val_loss:
                best = epoch - args.load_checkpoint_epoch
                best_val_loss = epoch_val_loss

            with open(f"{args.output_dir}resp_endo_train_logs.txt","a") as f:
                f.write(f"loss : {epoch_val_loss:.3f} , perplexity : {val_ppl:.3f} , epoch : {epoch} \n")

            print(f"loss : {epoch_val_loss:.3f} , perplexity : {val_ppl:.3f} , epoch : {epoch} \n")
            
            ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
            
            Path = f'{args.output_dir}{model_name}_resp_endo_epoch{epoch+1}.pth'
            
            torch.save(ckpt_dict,Path)
#             if epoch_loss <= epoch_losses[best-1]:
#                 best = epoch - args.load_checkpoint_epoch
            
                
    accelerator.print("training ended")
    # accelerator.print(epoch_losses)
    with open(f"{args.output_dir}resp_endo_train_logs.txt","a") as f:
        f.write(f"\nbest = {best + args.load_checkpoint_epoch}\n")
    # torch.save(model.state_dict(),f'{args.output_dir}{model_name}_resp_endo_last.pth')
    accelerator.print("best saved")

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

    if(args.test_data):
        dataset = load_dataset('text',data_files={'train': args.train_data, 'test': args.test_data})
    else:
        dataset = load_dataset('text',data_files={'train': args.train_data})
            
    print(dataset)
    # print("example :")
    # print(dataset['train'][0])

    context_length = 512
    stride = 256

    tokenizer = AutoTokenizer.from_pretrained(args.tk_name)
    tokenizer.pad_token = tokenizer.eos_token

    outputs = tokenizer(
        dataset["train"][:]["text"],
        truncation=True,
        # padding=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        stride=stride,
        return_length=True,
        # padding=True,
    )

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
        
        # print(f"Input IDs length: {len(outputs['input_ids'])}")
        # print(f"Input chunk lengths: {(outputs['length'])}")
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
        tokenize, batched=True, remove_columns=dataset["train"].column_names
    )
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets)

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    
    training_loop(args, tokenized_datasets)
    
if __name__ == "__main__":
    main()