from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import pickle
from datasets import Dataset
from transformers import default_data_collator
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import torch
from huggingface_hub import login
import xml.etree.ElementTree as ET
import re
import sys
import time

login(token = "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")

def append_to_file(filename, data):
    """Append the given data to the specified file."""
    with open(filename, 'a') as file:  # 'a' opens the file in append mode
        file.write(data + '\n')  # Append data with a newline to separate entries
        
def load_data_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['texts'],data['A'],data['Q']

def custom_data_collator(features):
    batch = default_data_collator(features)
    input_ids = [f['input_ids'] for f in features]
    input_ids = tokenizer(input_ids, padding=True, return_tensors="pt")
    batch['input_ids'] = input_ids['input_ids']
    batch['labels'] =input_ids['input_ids'] # labels same as input_ids for LM
    #batch['labels'][:,:20] = -100
    return batch

def prepare_dataset(texts,A,Q):
    train_size = int(0.8 * len(texts))
    train_set = Dataset.from_dict({
        'input_ids': texts[:train_size],
        'A': A[:train_size],
        'Q': Q[:train_size]
    })
    test_set = Dataset.from_dict({
        'input_ids': texts[train_size:],
        'A': A[train_size:],
        'Q': Q[train_size:]
    })
    return train_set, test_set

def prsanswer(text):
        # Using regex to find the content inside the <ans> tags
    match = re.search(r'<ans>\s*(.*?)\s*</ans>', text)
    if match:
        return match.group(1).strip()

    return None  # In case there is no match

def calculate_accuracy(dataset, model, n=200, ml = 128):
    corr = 0
    tot = 0
    genlen = 0
    for i in tqdm(range(0, min(len(dataset),n))):
        tot+=1
        input_ids = [dataset[i]['Q']+' <ans>']
        input_ids = tokenizer(input_ids, return_tensors="pt")
        predictions = model.generate(input_ids=input_ids['input_ids'].to('cuda:0'), attention_mask=input_ids['attention_mask'].to('cuda:0'), max_length=ml)
        output = tokenizer.decode(predictions[0].tolist(), skip_special_tokens=True)
        #print(output)
        genlen += len(predictions[0].tolist())
        output = prsanswer(output)
        
        corr += (output == dataset[i]['A'])
        #print(output,dataset[i]['A'],output == dataset[i]['A'])
    return corr/tot, genlen/tot

if __name__ == "__main__":
    TMA = [["x_proj", "embeddings", "in_proj", "out_proj"], ["q_proj", "up_pro_proj", "k_proj", "down_proj", "v_proj"],
      ["q_proj", "o_proj", "k_proj","v_proj"]]
    model_name = sys.argv[1]
    DS = sys.argv[2]
    hh = sys.argv[3]
    output_dir = sys.argv[4]
    lr = float(sys.argv[5])
    epoch = int(sys.argv[6])
    TM = int(sys.argv[7])
    max_len = int(sys.argv[8])
    bs = int(sys.argv[9])
    append_to_file("results2.txt", f'{model_name}, {DS}, {hh}, {output_dir}, {lr}, {epoch}, {TMA[TM]}, {max_len} {bs}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.torch_dtype = "bfloat16"
    model.torch_dtype = "bfloat16"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    #model.generation_config.pad_token_id = model.generation_config.eos_token_id
    texts,A,Q = load_data_from_pickle(f'{DS}.pkl')
    train_dataset, test_dataset = prepare_dataset(texts,A,Q)
    texts,A,Q = load_data_from_pickle(f'{DS}_OOD.pkl')
    ood_test_dataset, _ = prepare_dataset(texts,A,Q)
    model.to('cuda:0')
    start_time = time.time()
    n_eval = 200
    train_acc = calculate_accuracy(train_dataset, model, n_eval,max_len)
    append_to_file("results2.txt",f'{model_name} {DS} "train ACC: " {train_acc}')
    #test_acc = calculate_accuracy(test_dataset, model, n_eval,max_len)
    #append_to_file("results2.txt",f'{model_name} {DS} "test ACC: " {test_acc}')
    #ood_acc = calculate_accuracy(ood_test_dataset, model, n_eval,max_len)
    #append_to_file("results2.txt",f'{model_name} {DS} "ood ACC: " {ood_acc}')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    append_to_file("results2.txt",f'{model_name} {DS} "inference time: " {elapsed_time}')
