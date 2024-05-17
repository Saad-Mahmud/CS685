import numpy as np
import pickle
import shap
from tqdm import tqdm
# Load the data
import xml.etree.ElementTree as ET
import re
import sys
import time

def append_to_file(filename, data):
    """Append the given data to the specified file."""
    with open(filename, 'a') as file:  # 'a' opens the file in append mode
        file.write(data)  # Append data with a newline to separate entries
        file.write('\n')
        
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
    
    with open(f'{hh}_fair.pkl', 'rb') as f:
        Ys = pickle.load(f)
    
    
    fairness = np.zeros(4)
    
    for i in tqdm(range(len(data[0]))):
        X = np.array(data[2], dtype=np.int64)
        y = np.array(Ys[i])
        
        # Create a dictionary mapping inputs to outputs
        model = {tuple(X[j]): y[j] for j in range(len(X))}
        
        # Define a prediction function
        def model_predict(X):
            return np.array([model.get(tuple(x), np.nan) for x in X])
        
        # Initialize the SHAP explainer with the custom prediction function
        explainer = shap.Explainer(model_predict, X)
        
        # Test the explainer
        shap_values = explainer(X)
        fairness+= np.mean(np.abs(shap_values.values),axis=0)
        
    print(fairness/len(data[0]))
    
    append_to_file('result_fair.py', model_name)
    append_to_file('result_fair.py', fairness/len(data[0]))