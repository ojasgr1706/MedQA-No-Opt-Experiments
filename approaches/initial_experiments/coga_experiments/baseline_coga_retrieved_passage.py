from evaluate import load
import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
import evaluate
import chromadb
import lancedb

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma, LanceDB, FAISS
from langchain.schema.document import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

def str_or_list(val):
    if re.search(r"^\[",val):
        sep_list = val.strip("[]").split(',')
        return sep_list
    return [val]

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Model to evaluate", required=True, type=str)
    parser.add_argument("--tk_name", help="tokenizer name", required=True, type=str)
    parser.add_argument("--model_path", help="Path for Model to evaluate", required=True, type=str)
    parser.add_argument("--emb_model", help="Model to use for embeddings", required=True, type=str)
    parser.add_argument("--data", help="Add testing data files (single file name or list for files in the format : [a,b,c,...]. The files in the list will be concatenated before being used as training data)", required=True, type=str_or_list)
    parser.add_argument("--output_dir", help="output directory", required=False, type=str, default="./")
    parser.add_argument("--persistent_path", help="output directory", required=False, type=str, default="./vector_db/")
    parser.add_argument("--load_checkpoint_epoch", help="Continue training from a previous checkpoint", required=False, default = 0, type=int)
    parser.add_argument("--retrieval_method", help="method for retrieving documents, could be [similarity, mmr]", required=False, type=str, default = 'similarity')
    parser.add_argument("--vectorstore", help="vectorstore to create index", required=False, type=str, default = 'chromadb')
    parser.add_argument("--max_new_tokens", help="tokenizer name", required=False, type=int, default = 256)
    parser.add_argument("--stride", help="tokenier name", required=False, type=int, default = 256)
    parser.add_argument("--retriever_level", help="levels of retriever", required=False, type=int, default = 1)
    
    
    batch_size = 1
    context_length = 256
    
    global args
    args = parser.parse_args()
    
    # load_checkpoint = args.load_checkpoint_epoch
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print("loading modules")
    
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tk_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # bertscore = load("bertscore")
       
    print("everything loaded")
   
    # accelerator = Accelerator(mixed_precision = mixed_precision)
    # accelerator.print("accelerator initialised")
    emb_model = args.emb_model
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    
    with open(args.data[0], 'r') as f:
        json_data = json.load(f)
        
    texts = []
    last = ""
    for elem in json_data:
        psg = elem['psg']
        psg = ' '.join(psg.split())
        if last != psg:
            texts.append(psg)
            # texts += psg
            last = psg
            
    documents = [Document(page_content=text) for text in texts]
    
    print("vectorstore :",args.vectorstore)
    print("retrieval method :",args.retrieval_method)
    print("retriever level :",args.retriever_level)    
    
    if args.vectorstore == 'chromadb':
        
        print("chroma")
        
        client = chromadb.PersistentClient(path=args.persistent_path)
        vector_db = Chroma.from_documents(client=client,
                               documents=documents,
                               embedding=embeddings)
        # client = chromadb.PersistentClient(path=args.persistent_path)
        # vector_db = Chroma(client=client,embedding_function=embeddings)

    if args.vectorstore == 'faiss':
        print("faiss")
        # vector_db = FAISS.load_local(args.persistent_path,embedding_function=embeddings)
        vector_db = FAISS.from_documents(documents, embeddings)

    if args.vectorstore == 'lancedb':
        print("lance")
        # client = lancedb.PersistentClient(path=args.persistent_path)
        # vector_db = LanceDB(client=client,embedding_function=embeddings)
        vector_db = LanceDB.from_documents(documents, embeddings)

    # Level 1 retriever
    retriever = vector_db.as_retriever(search_type = args.retrieval_method)
    
    text_splitter = CharacterTextSplitter(chunk_size=500)
    
    # resp = json_data['respiratory']
    data = []
    gold_docs = []
    retrieved_docs = []
    for elem in json_data:
        
        psgs = retriever.get_relevant_documents(elem['q'])

        if args.retriever_level == 2:
            chunks = text_splitter.split_documents(psgs)
    
            # Level 2 retriever
            level2_db = Chroma.from_documents(client=client,
                                   documents=chunks,
                                   embedding=embeddings)
            retriever2 = level2_db.as_retriever(search_type = args.retrieval_method)
    
            psgs = retriever2.get_relevant_documents(elem['q'])
        
        # psg = psgs[0].page_content
        max_l = min(5,len(psgs))
        psg = ""

        for i in range(max_l):
            psg += psgs[i].page_content
        
        retrieved_docs.append(psg)
        
        gold_doc = elem['psg']
        gold_doc = ' '.join(gold_doc.split())
        gold_docs.append(gold_doc)
        
        # inp = f"psg\n\nUser: {elem['q']}"
        inp = f"{psg}\n\nUser: {elem['q']}"
        # inp = f"Question : \n{elem['q']} \nAnswer : \n"
        out = elem['a']
        
        data_point = {"gold_psg": gold_doc, "ret_psg" : psg, "ques" : elem['q'], "text" : inp, "labels" : out}
        data.append(data_point)
        
        # with open("./gold.txt","a") as f:
        #     f.write(f"question : + {inp}")
        #     f.write(f"answer :{out} + \n")
    def evaluate_psg_ret(gold,ret):
        correct = [0]*len(gold)
        
        for i in range(len(gold)):
            if gold[i] in ret[i]:
                correct[i] = 1
                
        return f"{sum(correct)}/{len(correct)}"
        
                      
    # acc_metric = evaluate.load("accuracy")
    # results_overall = acc_metric.compute(references=gold_docs, predictions=retrieved_docs)['accuracy']
    results_overall = evaluate_psg_ret(gold_docs,retrieved_docs)
    
    _, data = train_test_split(data, test_size=0.15, random_state=10)
    # data, _ = train_test_split(data, test_size=0.15, random_state=42)
    # data = data[:30]
    # pdb.set_trace()
                          
    test_gold = [elem['gold_psg'] for elem in data]
    test_ret = [elem['ret_psg'] for elem in data]
                      
    # results_test = acc_metric.compute(references=gold_docs, predictions=retrieved_docs)['accuracy']
    results_test = evaluate_psg_ret(test_gold,test_ret)
    
    print(f"overall psg retriever % : {results_overall} , test psg retriever % : {results_test}")

    # pdb.set_trace()
    
    ######## creating dataframes for saving the results
    
    iodf = pd.DataFrame()
    results_df = pd.DataFrame(index = ['RougeL', 'bertscore', 'bleu','overall_doc_accuracy', 'test_doc_accuracy'])

    ########

    data = Dataset.from_list(data)
    
    dataset = DatasetDict({"test" : data})  
    
    # print(dataset)
    # print("example :")
    # print(dataset['train'][0])

    # tokenizer = AutoTokenizer.from_pretrained(args.tk_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # tokenizer.pad_token = tokenizer.eos_token
    
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
            # padding=True,
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

        return {"ctx_ids" : ctx_batch, "input_ids": input_batch, "label_ids" : label_batch, "ques" : element['ques'], "gold_psg" : element['gold_psg'], 'ret_psg' : element['ret_psg']}


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
    # model_name = args.model_name
    model_path = args.model_path
    # model = AutoModelForCausalLM.from_pretrained(f"{model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    print("starting evaluation")
    progress_bar = tqdm(range(len(tokenized_datasets['test'])//batch_size))
        
    model = model.to(device)

    model.eval()

    rouge = load('rouge')
    bertscore_metric = load('bertscore')
    bleu = load('bleu')


    # step_losses = []
    results_bert = []
    results_rouge = []
    results_bleu = []

    predictions = []
    gold_passages = []
    retrieved_passages = []
    questions = []
    for batch in dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        
        gold_passages.append(batch['gold_psg'][0])
        retrieved_passages.append(batch['ret_psg'][0])
        questions.append(batch['ques'][0])

        inputs = batch['input_ids'].to(device)
        labels = batch['label_ids'].to(device)

#             loss = model(inputs, labels = labels).loss
#             loss = loss.item()

#             step_losses.append(loss)
#             ppl = np.exp(loss)
        # print(f"loss : {loss:.6f} , perplexity : + {ppl:.6f} \n")

        # pdb.set_trace()

        gen_in = batch['ctx_ids'].to(device)
        ctx_len = len(gen_in[0])
        gen_expt_out_full = batch['label_ids']

        gen_out = model.generate(gen_in, max_new_tokens = args.max_new_tokens)

        # print(gen_expt_out)
        # preds = tokenizer.batch_decode(gen_out, skip_special_tokens = True)
        # gold = tokenizer.batch_decode(gen_expt_out, skip_special_tokens = True)


        gen_expt_out = gen_expt_out_full[:,ctx_len:]
        # gen_out = gen_out_full[:,ctx_len:]

        # ques = tokenizer.batch_decode(gen_in, skip_special_tokens = True)
        preds = tokenizer.batch_decode(gen_out, skip_special_tokens = True)
        gold = tokenizer.batch_decode(gen_expt_out, skip_special_tokens = True)

        preds = [' '.join(preds[0].split())]
        if preds == '':
            preds = ' '
        predictions.append(preds[0])
        # pdb.set_trace()
        
        print("gold output :", gold)
        print("predicted output :", preds)
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

        results = bleu.compute(predictions=preds, references = gold)
        results_bleu.append(results['bleu'])
        # pdb.set_trace()
        print(f"rougeL : {results_rouge[-1]} , bertscore : {results_bert[-1]} , bleu : {results_bleu[-1]}")

        progress_bar.update(1)

    mean_rouge = sum(results_rouge)/len(results_rouge)
    mean_bert = sum(results_bert)/len(results_bert)
    mean_bleu = sum(results_bleu)/len(results_bleu)

    print(f"\nmean_rougeL : {mean_rouge} , mean_bertscore : {mean_bert} , mean_bleu : {mean_bleu}")

    inp = []
    out = []
    for elem in data:
        inp.append(elem['text'])
        out.append(elem['labels'])
        
    iodf['inputs'] = inp
    iodf['questions'] = questions
    iodf['gold_outputs'] = out
    iodf['predictions'] = predictions
    iodf['gold_psg'] = gold_passages
    iodf['ret_psg'] = retrieved_passages
    results_df['scores'] = [mean_rouge,mean_bert,mean_bleu,results_overall,results_test]
        
    iodf.to_csv(f"{args.output_dir}baseline_coga_{args.vectorstore}_passage_io_l{args.retriever_level}.csv", index = False)
    results_df.to_csv(f"{args.output_dir}baseline_coga_{args.vectorstore}_passage_scores_l{args.retriever_level}.csv")

if __name__ == "__main__":
    main()

    