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

login(token = "#")

from typing import Any, Dict, Optional, Tuple, Union
from transformers import MambaConfig
class MambaCache:
    """
    Arguments:
        config: MambaConfig
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
        conv_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, conv_kernel_size]
        ssm_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, ssm_state_size]
    """

    def __init__(
        self, config: MambaConfig, batch_size: int, dtype: torch.dtype = torch.bfloat16, device: Optional[str] = None
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


def append_to_file(filename, data):
    """Append the given data to the specified file."""
    with open(filename, 'a') as file:  # 'a' opens the file in append mode
        file.write(data + '\n')  # Append data with a newline to separate entries
        
def calculate_gen_time(model, cache = 0, n=200, ml = 128):
    genlen = 0
    if cache == 0:
        cache_params =MambaCache(model.config, 1, device = model.device)
    for i in tqdm(range(n)):     
        inputs = tokenizer("You need to write a story about a beautiful girl, Make the story as long as possible. Start Story: ", return_tensors="pt").to(model.device)
        if cache ==0: 
            outputs = model.generate(inputs['input_ids'], max_length=ml,cache_params =cache_params,use_cache = True)
        else:
            outputs = model.generate(inputs['input_ids'], max_length=ml,use_cache = True)       
        genlen += len(outputs[0].tolist())
        del inputs
        del outputs
    return genlen/n

if __name__ == "__main__":
    model_name = sys.argv[1]
    cache = int(sys.argv[2])
    
    append_to_file("results3.txt", f'{model_name} {cache}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.torch_dtype = "bfloat16"
    model.torch_dtype = "bfloat16"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    model.to('cuda:0')
    model.eval()
    start_time = time.time()
    n_eval = 10
    gl = calculate_gen_time( model,cache, n_eval,2000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_tok = elapsed_time/gl*100
    append_to_file("results3.txt",f'{model_name} "inference time: " {time_per_tok}')

    start_time = time.time()
    n_eval = 10
    gl = calculate_gen_time( model, cache, n_eval,4000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_tok = elapsed_time/gl*100
    append_to_file("results3.txt",f'{model_name} "inference time: " {time_per_tok}')

    start_time = time.time()
    n_eval = 10
    gl = calculate_gen_time( model, cache, n_eval,8000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_tok = elapsed_time/gl*100
    append_to_file("results3.txt",f'{model_name} "inference time: " {time_per_tok}')
