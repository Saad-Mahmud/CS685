import numpy as np
import h5py
from tokens import GLOBAL_TOKENS_MAP as TM, GLOBAL_TOKENS_RMAP as RM
import pickle
import random
def tokenize_number(n):
    """ Tokenize the number where each digit becomes a separate token represented as an integer. """
    return [RM[digit] for digit in str(n)]

def get_addition_reasoning(num1, num2):
    ret = []
    #print(num1,len(num1))
    c = 0
    for i in range(len(num1)-1,-1,-1):
        _add = RM[num1[i]] + RM[num2[i]] + c
        add = str(_add).zfill(2)
        ret = ret + [RM['<step>'], RM[num1[i]], RM['+'] , RM[num2[i]], RM['+'] , 
                     c, RM['='], RM[add[0]], RM[add[1]], RM['</step>\n']]
        c = _add//10
        ret = ret + [RM['<step>'], RM['C:'], RM['='] , c, RM['</step>\n']]
        ret = ret + [RM['<step>'], RM['A:'], RM['='] , RM[add[1]], RM['</step>\n']]
    return ret


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
    
    # Tokenize the numbers
    num1_tokens = tokenize_number(num1_str)
    num2_tokens = tokenize_number(num2_str)
    
    # Generate the prompt with tokens
    prompt_tokens = [RM['BOS']]+[RM['<prompt>']] + num1_tokens + [RM['+']] + num2_tokens+[RM['</prompt>\n']] 
    #while len(prompt_tokens)<max_prompt_len:
    #    prompt_tokens.append(RM['PAD'])
    #prompt_tokens[-1] = RM['</prompt>\n']
    
    #reasoning_token = get_addition_reasoning(num1_str, num2_str)
    reasoning_token = []
    # Calculate the answer and tokenize
    answer = num1 + num2
    answer = str(answer).zfill(max_num)
    info = tokenize_number(answer)
    answer_tokens = [RM['<ans>']] + tokenize_number(answer) + [RM['</ans>']]
    #while len(answer_tokens)+len(prompt_tokens)<max_seq_len:
    #    answer_tokens.append(RM['PAD'])
    answer_tokens += [RM['EOS']]
    #answer =  reasoning_token+answer_tokens
    texts = prompt_tokens + answer_tokens
    print(len(texts))
    return {"texts": texts, 'info': info}

def make_addition_dataset(name, nData=500, min_len = 1, max_len = 5):
    dataset = {'texts': [], 'info': []}
    
    for _ in range(nData):
        data_point = make_addition(min_len,max_len)
        dataset['texts'].append(data_point['texts'])
        dataset['info'].append(data_point['info'])
        
    # Save dataset to a pickle file
    with open(f'addition_dataset{name}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

def print_addition(texts):
    
    texts_string = ' '.join(TM[token] for token in texts)
    
    print("texts:", texts_string)
    
if __name__ == "__main__":
    addition_example = make_addition(min_len=1, max_len=10)
    print(addition_example)
    print(len(addition_example['texts']))
    print_addition(addition_example['texts'])
    make_addition_dataset('', 10000,1,5)  # Create a dataset with 500 examples
    for i in range(6,21):
        make_addition_dataset(f'{i}', 10000,i,i)  # Create a dataset with 500 examples
    
    
