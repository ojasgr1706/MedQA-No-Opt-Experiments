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



# def str_or_list(val):
#     if re.search(r"^\[",val):
#         sep_list = val.strip("[]").split(',')
#         return sep_list
#     return [val]

# parser = argparse.ArgumentParser()
# # parser.add_argument("--train_data", help="Add input data files (single file name or list fo files in the format : [a,b,c,...]. The files in the list will be concatenated before being used as training data)", required=True, type=str_or_list)
# parser.add_argument("--boolean",action="store_true")
# # parser.add_argument("--test_data", help="Add testing data files (single file name or list fo files in the format : [a,b,c,...]. The files in the list will be concatenated before being used as training data)", required=True, type=str_or_list)
# args = parser.parse_args()

# # print(args.train_data)

# # dataset = load_dataset('text',data_files={'train': args.train_data, 'test': "../extracted_text/kumar_and_clark/kumar_and_clark_top_3.txt"})
# # print(dataset)
# # print("example :")
# # print(dataset['train'][0])

# if(args.boolean):
#     print("this")
# else:
#     print("not this")

class Args:
    def __init__(self):
        self.train_data = ["../extracted_text/kumar_and_clark/kumar_and_clark_respiratory_top_3.txt","../extracted_text/Harrison/Harrison_respiratory_top_3.txt"]
        self.test_data = ""
        self.model_name = "gpt2-xl"
        self.output_dir = f"../model/qa_models/{self.model_name}-medicine/"
        self.tk_name = "gpt2-xl"
        
        
def main():
    args = Args()
#     print(args.model_name)
#     print(args.output_dir)
    
#     bertscore = load("bertscore")
#     predictions = ["I have a good dog now"]
#     references = ["Now there is a dog that I have, which is good"]
#     results = bertscore.compute(predictions=predictions, references=references, lang="en")

    dataset = load_dataset('text',data_files={'train': args.train_data})
    print(dataset)
    
    
    context_length = 512
    stride = 256
    
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

    tokenizer = AutoTokenizer.from_pretrained(args.tk_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_datasets = dataset.map(
        tokenize, batched=True, remove_columns=dataset["train"].column_names
    )
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets)
    
    num_epochs = 100
    warm_up_steps = 100
    train_batch_size = 4
    
    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=False, batch_size=train_batch_size)
    training_steps = num_epochs * len(train_dataloader) - warm_up_steps
    print(training_steps)

    
    
if __name__ == "__main__":
    main()