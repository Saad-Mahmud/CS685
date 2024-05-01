from transformers import JambaConfig, JambaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from datasets import Dataset
import torch
import h5py
import pickle
import numpy as np
from datagen.tokens import GLOBAL_TOKENS_MAP, GLOBAL_TOKENS_RMAP
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from transformers import default_data_collator
import torch
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

def get_config(HL = 11, ss = 24):
    # Define a custom model configuration with fewer parameters
    config = JambaConfig( vocab_size=len(GLOBAL_TOKENS_MAP),  # Based on the number of unique tokens
        hidden_size = 256,
        intermediate_size = 512,
        num_hidden_layers = 22,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        use_cache = True,
        max_position_embeddings = 128,
        num_experts_per_tok = 2,
        num_experts = 4,
        expert_layer_period = 2,
        expert_layer_offset = 1,
        attn_layer_period = 2,
        attn_layer_offset = 1,
        mamba_d_state = 16,
        mamba_d_conv = 4,
        mamba_expand = 2,
        pad_token_id = 31,
        bos_token_id = 30,
        eos_token_id = 29,
    )
    
    return config

def custom_data_collator(features):
    batch = default_data_collator(features)
    input_ids = [f['input_ids'] for f in features]

    batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
    batch['labels'] = torch.tensor(input_ids, dtype=torch.long)  # labels same as input_ids for LM
    #batch['labels'][:,:20] = -100
    return batch

def load_data_from_pickle(filepath='dataset/addition_dataset.pkl'):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['texts'],data['info']


def prepare_dataset(texts,info):
    train_size = int(0.8 * len(texts))
    train_set = Dataset.from_dict({
        'input_ids': texts[:train_size],
        'info': info[:train_size]
    })
    test_set = Dataset.from_dict({
        'input_ids': texts[train_size:],
        'info': info[train_size:]
    })
    return train_set, test_set

# Decoding function to transform tokens back to text using GLOBAL_TOKENS_MAP
def decode(tokens):
    return ' '.join(GLOBAL_TOKENS_MAP.get(token, "") for token in tokens if token in GLOBAL_TOKENS_MAP)


def calculate_accuracy(dataset,model,boundary = 20):
    #print(decode(dataset[0]['input_ids'][:boundary]))
    corr = 0
    batch_size = 64
    for i in range(0, len(dataset), batch_size):
        #print(i)
        bs = min(batch_size,len(dataset)-i)
        predictions = model.generate(torch.tensor([dataset[j]['input_ids'][:boundary] for j in range(i,i+bs)]).to('cuda:0'), max_length=128, )
        
        for k in range(bs):    
            pred = []
            add = False
            for j in predictions[k][boundary:].tolist():
                if j == 15:
                    break
                if add:
                    pred.append(j)
                if j == 14:
                    add = True
                
            if len(dataset[i+k]['info']) != len(pred):
                continue
            if dataset[i+k]['info'] == pred:
                corr+=1
    #print("Accuracy: ",corr/len(dataset))
    return corr/len(dataset)


def train_model(HL, rope, epoch):
    print("####################################################")
    texts,info = load_data_from_pickle()
    train_dataset, test_dataset = prepare_dataset(texts,info)
    config = get_config(HL,rope)
    model = JambaForCausalLM(config)
    print(f"Total parameters in the model: {model.num_parameters()}")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epoch,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy='epoch',
        learning_rate=1e-4
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=custom_data_collator
    )   
    trainer.train()
    accs = []
    ac = calculate_accuracy(test_dataset,model, 22*2+4)
    for i in range(5):
        accs.append(ac)
    for i in tqdm(range(6,21)):
        texts, info = load_data_from_pickle(f'dataset/addition_dataset{i}.pkl')
        train_dataset, test_dataset = prepare_dataset(texts, info)
        ac = calculate_accuracy(test_dataset,model, 22*2+4)
        accs.append(ac)
    print(accs)
    return np.array(accs)


def experiment1():
    HLs = [33,44,55,66]
    ropes = [20,84,128]
    trial = 3
    for HL in HLs:
        for rope in ropes:
            accs = None
            for t in range(trial):
                if accs is None:
                    accs = train_model(HL,rope, 100)
                else:
                    accs +=train_model(HL,rope, 100)
            accs/=trial
            res = f'HL: {HL}, rope: {rope}, performance: {accs}\n'
            print(res)
            with open("final_res.txt", "a") as file:
                file.write(res)

if __name__ == "__main__":
    experiment1()
