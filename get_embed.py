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
import numpy as np
import pickle
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
import xml.etree.ElementTree as ET
import re
import sys
import time
login(token = "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")






if __name__ == '__main__':
    model_name = sys.argv[1]
    DS = sys.argv[2]
    hh = sys.argv[3]
    output_dir = sys.argv[4]
    lr = float(sys.argv[5])
    epoch = int(sys.argv[6])
    TM = int(sys.argv[7])
    max_len = int(sys.argv[8])
    bs = int(sys.argv[9])
    with open('dataset/fairness.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data[0][:10],data[1][:10],data[2])
    tokenizer = AutoTokenizer.from_pretrained(model_name,torch_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(model_name,torch_dtype=torch.bfloat16)
    model.to('cuda:0')
        # Initialize a 2D array to store the similarities
    sims = np.zeros((len(data[0]), len(data[1])))
    
    # Precompute hidden states for data[1]
    precomputed_hidden_states = []
    with torch.no_grad():
        for j in tqdm(range(len(data[1]))):
            inputs = tokenizer(data[1][j], return_tensors="pt").to(model.device)
            outputs = model(inputs['input_ids'], output_hidden_states=True)
            p1 = outputs["hidden_states"][-1][0, -1, :].to('cpu')
            precomputed_hidden_states.append(p1)
    
    with torch.no_grad():
        for i in tqdm(range(len(data[0]))):
            inputs = tokenizer(data[0][i], return_tensors="pt").to(model.device)
            outputs = model(inputs['input_ids'], output_hidden_states=True)
            s1 = outputs["hidden_states"][-1][0, -1, :].to('cpu')
            for j in range(len(data[1])):
                p1 = precomputed_hidden_states[j]
                sims[i, j] = F.cosine_similarity(s1, p1, dim=0).item()
    
    # Print or use the sims array as needed
    print(sims)
    with open(f'{hh}_fair.pkl', 'wb') as f:
        pickle.dump(sims, f)