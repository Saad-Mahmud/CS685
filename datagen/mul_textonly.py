import numpy as np
import h5py
from tokens import GLOBAL_TOKENS_MAP as TM, GLOBAL_TOKENS_RMAP as RM
import pickle
import random

def make_mul(min_len = 1, max_len = 5, max_num = 22):
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
    prompt_tokens = '<prompt> ' + num1_tokens + ' * ' + num2_tokens+' </prompt>'  
    
    answer = num1 * num2
    answer = str(answer).zfill(max_num)
    info = answer
    answer_tokens = ' <ans> ' + info + ' </ans>'
    texts = prompt_tokens + answer_tokens
    print(len(texts))
    return {"texts": texts,"Q": prompt_tokens, 'A': info}

def make_mul_dataset(name, nData=500, max_num = 22, min_len = 1, max_len = 5):
    dataset = {"texts": [],"Q": [], 'A': []}
    
    for _ in range(nData):
        data_point = make_mul(min_len, max_len, max_num)
        dataset['texts'].append(data_point['texts'])
        dataset['A'].append(data_point['A'])
        dataset['Q'].append(data_point['Q'])
        
    # Save dataset to a pickle file
    with open(f'mul_dataset_text{name}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    make_mul_dataset('_Large', 10000, 42, 1, 10)  # Create a dataset with 500 examples
    make_mul_dataset(f'_Large_OOD', 10000, 42, 11, 20)  # Create a dataset with 500 examples
    
    
