import numpy as np
import h5py
from tokens import GLOBAL_TOKENS_MAP as TM, GLOBAL_TOKENS_RMAP as RM
import pickle
import random

def make_addition(min_len = 1, max_len = 5, 
                  max_num = 22, max_seq_len = 84):
    # Generate two random numbers
    p = np.array([i+0.0 for i in range(min_len,max_len+1)])
    p/=np.sum(p)
    _len = np.random.choice([i for i in range(min_len, max_len+1)], p = p)
    num1 = random.randint(1, 10**int(_len))
    num2 = random.randint(1, 10**int(_len))
 
    # Convert numbers to padded strings
    num1_str = str(num1).zfill(max_num)
    num2_str = str(num2).zfill(max_num)
    
    num1_tokens = num1_str
    num2_tokens = num2_str
    
    # Generate the prompt with tokens
    prompt_tokens = '<prompt> ' + num1_tokens + ' + ' + num2_tokens+' </prompt>'  
    
    answer = num1 + num2
    answer = str(answer).zfill(max_num)
    info = str(answer)
    answer_tokens = ' <ans> ' + info + ' </ans>'
    texts = prompt_tokens + answer_tokens
    return {"texts": texts,"Q": prompt_tokens, 'A': info}

def make_addition_dataset(name, nData=500, min_len = 1, max_len = 5):
    dataset = {"texts": [],"Q": [], 'A': []}
    
    for _ in range(nData):
        data_point = make_addition(min_len,max_len)
        dataset['texts'].append(data_point['texts'])
        dataset['A'].append(data_point['A'])
        dataset['Q'].append(data_point['Q'])
        
    # Save dataset to a pickle file
    with open(f'addition_dataset_text{name}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    addition_example = make_addition(min_len=1, max_len=10)
    print(addition_example)
    print(len(addition_example['texts']))
    print(addition_example['texts'])
    make_addition_dataset('', 10000,1,5)  # Create a dataset with 500 examples
    for i in range(6,21):
        make_addition_dataset(f'{i}', 10000,i,i)  # Create a dataset with 500 examples
    
    
