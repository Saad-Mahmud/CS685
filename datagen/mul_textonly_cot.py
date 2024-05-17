import numpy as np
import h5py
from tokens import GLOBAL_TOKENS_MAP as TM, GLOBAL_TOKENS_RMAP as RM
import pickle
import random
def get_reasoning(n1, n1s, n2s, info, sizes):
    #print(n1, n1s, n2s, info, sizes)
    ret = '<Chain of Thought>\n'
    e = 0
    K = []
    for i in n2s[::-1]:
        ret = ret + n1s + ' * ' + i + ' = '
        l = str(n1*int(i)*10**e).zfill(sizes[2]) 
        ret +=l+'\n'
        K.append(l)
        n1s = "same"
        e+=1
    
    for i in range(len(K[0])-1,-1,-1):
        ret+= K[0][i]+"+"+K[1][i]+"+"+K[2][i]+"+"+K[3][i]+"+"+K[4][i]+" = "+info[i]+'\n'
    ret += '</Chain of Thought>\n'
    #print(len(ret))
    return ret


def make_mul(sizes):
    # Generate two random numbers
    p = np.array([i+0.0 for i in range(sizes[3],sizes[4]+1)])
    p/=np.sum(p)
    _len = np.random.choice([i for i in range(sizes[3], sizes[4]+1)], p = p)
    num1 = random.randint(1, 10**int(_len))
    p = np.array([i+0.0 for i in range(sizes[5],sizes[6]+1)])
    p/=np.sum(p)
    _len = np.random.choice([i for i in range(sizes[5], sizes[6]+1)], p = p)
    num2 = random.randint(1, 10**int(_len))
    
    
    # Convert numbers to padded strings
    num1_str = str(num1).zfill(sizes[0])
    num2_str = str(num2).zfill(sizes[1])
    
    num1_tokens = num1_str
    num2_tokens = num2_str
    
    # Generate the prompt with tokens
    prompt_tokens = '<prompt> ' + num1_tokens + ' * ' + num2_tokens+' </prompt>'  
    
    answer = num1 * num2
    answer = str(answer).zfill(sizes[2])
    info = answer
    answer_tokens = ' <ans> ' + info + ' </ans>'
    reasoning = get_reasoning(num1, num1_str, num2_str,info,sizes)
    return {"Q": prompt_tokens, 'A_': answer_tokens, 'A': info, 'CoT': reasoning}

def make_mul_dataset(name, nData=500, sizes = (10, 5, 15, 1, 5, 1, 5)):
    dataset = {"texts": [],"Q": [],  'A': []}
    
    for _ in range(nData):
        data_point = make_mul(sizes)
        print(dataset['texts'])
        dataset['texts'].append(data_point['Q']+data_point['CoT']+data_point['A_'])
        dataset['A'].append(data_point['A'])
        dataset['Q'].append(data_point['Q'])
        print(dataset['texts'][-1])
        
        dataset['texts'].append(data_point['Q']+data_point['A_'])
        dataset['A'].append(data_point['A'])
        dataset['Q'].append(data_point['Q'])
        print(dataset['texts'][-1])
        
        
        
        
    # Save dataset to a pickle file
    with open(f'mul_dataset_text{name}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    make_mul_dataset("dum",1)
    #make_mul_dataset('_woC OT', 10, (10, 5, 15, 1, 5, 1, 5))  # Create a dataset with 500 examples
    #make_mul_dataset(f'_woCOT_OOD', 10, (10, 5, 15, 1, 10, 1, 5))  # Create a dataset with 500 examples
    
    
