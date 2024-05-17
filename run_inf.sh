#!/bin/bash

# Define the path to the Python script
PYTHON_SCRIPT=run_all_inf.py

# Call the Python script with hardcoded arguments

#Small
#python $PYTHON_SCRIPT "state-spaces/mamba-1.4b-hf" "dataset/addition_dataset_text_Small" "Mamba1.4B_add_small" "Mamba1.4B/add_small" 2e-4 8 0 128 8

#python $PYTHON_SCRIPT "microsoft/phi-1_5" "dataset/addition_dataset_text_Small" "Phi1.5B_add_small" "Phi1.5B/add_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "google/gemma-2b" "dataset/addition_dataset_text_Small" "Gemma2B_add_small" "Gemma2B/add_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "google/recurrentgemma-2b" "dataset/addition_dataset_text_Small" "RGemma2B_add_small" "RGemma2B/add_small" 2e-4 8 2 128 8

#python $PYTHON_SCRIPT "state-spaces/mamba-2.8b-hf" "dataset/addition_dataset_text_Small" "Mamba2.8B_add_small" "Mamba2.8B/add_small" 2e-4 8 0 128 8

#python $PYTHON_SCRIPT "stabilityai/stablelm-zephyr-3b" "dataset/addition_dataset_text_Small" "Zephyr3B_add_small" "Zephyr3B/add_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/addition_dataset_text_Small" "Mamba7B_add_small" "Mamba7B/add_small" 2e-4 8 0 128 8

#python $PYTHON_SCRIPT "meta-llama/Llama-2-7b-hf" "dataset/addition_dataset_text_Small" "Llama7B_add_small" "Llama7B/add_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/addition_dataset_text_Small" "Llama8B_add_small" "Llama8B/add_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "mistralai/Mistral-7B-v0.1" "dataset/addition_dataset_text_Small" "Mistral7B_add_small" "Mistral7B/add_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "google/gemma-7b" "dataset/addition_dataset_text_Small" "Gemma7B_add_small" "Gemma7B/add_small" 2e-4 8 1 128 8

#Long

#python $PYTHON_SCRIPT "state-spaces/mamba-1.4b-hf" "dataset/addition_dataset_text_Large" "Mamba1.4B_add_large" "Mamba1.4B/add_large" 2e-4 8 0 175 8

#python $PYTHON_SCRIPT "microsoft/phi-1_5" "dataset/addition_dataset_text_Large" "Phi1.5B_add_large" "Phi1.5B/add_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "google/gemma-2b" "dataset/addition_dataset_text_Large" "Gemma2B_add_large" "Gemma2B/add_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "google/recurrentgemma-2b" "dataset/addition_dataset_text_Large" "RGemma2B_add_large" "RGemma2B/add_large" 2e-4 8 2 175 8

#python $PYTHON_SCRIPT "state-spaces/mamba-2.8b-hf" "dataset/addition_dataset_text_Large" "Mamba2.8B_add_large" "Mamba2.8B/add_large" 2e-4 8 0 175 8

#python $PYTHON_SCRIPT "stabilityai/stablelm-zephyr-3b" "dataset/addition_dataset_text_Large" "Zephyr3B_add_large" "Zephyr3B/add_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/addition_dataset_text_Large" "Mamba7B_add_large" "Mamba7B/add_large" 2e-4 8 0 175 8

#python $PYTHON_SCRIPT "meta-llama/Llama-2-7b-hf" "dataset/addition_dataset_text_Large" "Llama7B_add_large" "Llama7B/add_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/addition_dataset_text_Large" "Llama8B_add_large" "Llama8B/add_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "mistralai/Mistral-7B-v0.1" "dataset/addition_dataset_text_Large" "Mistral7B_add_large" "Mistral7B/add_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "google/gemma-7b" "dataset/addition_dataset_text_Large" "Gemma7B_add_large" "Gemma7B/add_large" 2e-4 8 1 175 8


#mul SMALL

#python $PYTHON_SCRIPT "state-spaces/mamba-1.4b-hf" "dataset/mul_dataset_text_Small" "Mamba1.4B_mul_small" "Mamba1.4B/mul_small" 2e-4 8 0 128 8

#python $PYTHON_SCRIPT "microsoft/phi-1_5" "dataset/mul_dataset_text_Small" "Phi1.5B_mul_small" "Phi1.5B/mul_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "google/gemma-2b" "dataset/mul_dataset_text_Small" "Gemma2B_mul_small" "Gemma2B/mul_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "google/recurrentgemma-2b" "dataset/mul_dataset_text_Small" "RGemma2B_mul_small" "RGemma2B/mul_small" 2e-4 8 2 128 8

#python $PYTHON_SCRIPT "state-spaces/mamba-2.8b-hf" "dataset/mul_dataset_text_Small" "Mamba2.8B_mul_small" "Mamba2.8B/mul_small" 2e-4 8 0 128 8

#python $PYTHON_SCRIPT "stabilityai/stablelm-zephyr-3b" "dataset/mul_dataset_text_Small" "Zephyr3B_mul_small" "Zephyr3B/mul_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_Small" "Mamba7B_mul_small" "Mamba7B/mul_small" 2e-4 8 0 128 8

#python $PYTHON_SCRIPT "meta-llama/Llama-2-7b-hf" "dataset/mul_dataset_text_Small" "Llama7B_mul_small" "Llama7B/mul_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_Small" "Llama8B_mul_small" "Llama8B/mul_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "mistralai/Mistral-7B-v0.1" "dataset/mul_dataset_text_Small" "Mistral7B_mul_small" "Mistral7B/mul_small" 2e-4 8 1 128 8

#python $PYTHON_SCRIPT "google/gemma-7b" "dataset/mul_dataset_text_Small" "Gemma7B_mul_small" "Gemma7B/mul_small" 2e-4 8 1 128 8


#mul LARGE

#python $PYTHON_SCRIPT "state-spaces/mamba-1.4b-hf" "dataset/mul_dataset_text_Large" "Mamba1.4B_mul_large" "Mamba1.4B/mul_large" 2e-4 8 0 175 8

#python $PYTHON_SCRIPT "microsoft/phi-1_5" "dataset/mul_dataset_text_Large" "Phi1.5B_mul_large" "Phi1.5B/mul_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "google/gemma-2b" "dataset/mul_dataset_text_Large" "Gemma2B_mul_large" "Gemma2B/mul_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "google/recurrentgemma-2b" "dataset/mul_dataset_text_Large" "RGemma2B_mul_large" "RGemma2B/mul_large" 2e-4 8 2 175 8

#python $PYTHON_SCRIPT "state-spaces/mamba-2.8b-hf" "dataset/mul_dataset_text_Large" "Mamba2.8B_mul_large" "Mamba2.8B/mul_large" 2e-4 8 0 175 8

#python $PYTHON_SCRIPT "stabilityai/stablelm-zephyr-3b" "dataset/mul_dataset_text_Large" "Zephyr3B_mul_large" "Zephyr3B/mul_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_Large" "Mamba7B_mul_large" "Mamba7B/mul_large" 2e-4 8 0 175 8

#python $PYTHON_SCRIPT "meta-llama/Llama-2-7b-hf" "dataset/mul_dataset_text_Large" "Llama7B_mul_large" "Llama7B/mul_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_Large" "Llama8B_mul_large" "Llama8B/mul_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "mistralai/Mistral-7B-v0.1" "dataset/mul_dataset_text_Large" "Mistral7B_mul_large" "Mistral7B/mul_large" 2e-4 8 1 175 8

#python $PYTHON_SCRIPT "google/gemma-7b" "dataset/mul_dataset_text_Large" "Gemma7B_mul_large" "Gemma7B/mul_large" 2e-4 8 1 175 8



#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_wCOT" "Llama8B_wCOT" "Llama8B/wCOT" 2e-4 8 1 450 2

#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_wCOT" "Mamba7B_wCOT" "Mamba7B/wCOT" 2e-4 8 1 450 2

#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_woCOT" "Llama8B_woCOT" "Llama8B/woCOT" 2e-4 8 1 128 2

#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_woCOT" "Mamba7B_woCOT" "Mamba7B/woCOT" 2e-4 8 1 128 2



PYTHON_SCRIPT=arithmatic_add2.py

#python $PYTHON_SCRIPT "mistralai/Mistral-7B-v0.1" "dataset/mul_dataset_text_Large" "Mistral7B_mul_large" "Mistral7B/mul_large" 2e-4 8 1 175 2

#python $PYTHON_SCRIPT "google/gemma-7b" "dataset/mul_dataset_text_Large" "Gemma7B_mul_large" "Gemma7B/mul_large" 2e-4 8 1 175 2



#python $PYTHON_SCRIPT "state-spaces/mamba-2.8b-hf" "dataset/mul_dataset_text_woCOT" "Mamba2.8B_woCOT" "Mamba2.8B/woCOT" 2e-4 8 0 128 2
#python $PYTHON_SCRIPT "state-spaces/mamba-2.8b-hf" "dataset/mul_dataset_text_wCOT" "Mamba2.8B_wCOT" "Mamba2.8B/wCOT" 2e-4 8 0 512 2
#python $PYTHON_SCRIPT "stabilityai/stablelm-zephyr-3b" "dataset/mul_dataset_text_woCOT" "Zephyr3B_woCOT" "Zephyr3B/woCOT" 2e-4 8 1 128 2
#python $PYTHON_SCRIPT "stabilityai/stablelm-zephyr-3b" "dataset/mul_dataset_text_wCOT" "Zephyr3B_wCOT" "Zephyr3B/wCOT" 2e-4 8 1 512 2


#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_wCOT" "Mamba7B_wCOT" "Mamba7B/wCOT" 2e-4 8 0 512 2
#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_woCOT" "Llama8B_woCOT" "Llama8B/woCOT" 2e-4 8 1 128 2
#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_wCOT" "Llama8B_wCOT" "Llama8B/wCOT" 2e-4 8 1 512 2
#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_woCOT" "Mamba7B_woCOT" "Mamba7B/woCOT" 2e-4 8 0 128 2



#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_woCOT" "Llama8B_woCOT" "Llama8B/woCOT" 2e-4 8 1 128 2
#PYTHON_SCRIPT=get_embed.py
PYTHON_SCRIPT=SHAP.py
python $PYTHON_SCRIPT "state-spaces/mamba-1.4b-hf" "dataset/addition_dataset_text_Small" "Mamba1.4B" "Mamba1.4B/add_small" 2e-4 8 0 128 8

python $PYTHON_SCRIPT "microsoft/phi-1_5" "dataset/addition_dataset_text_Small" "Phi1.5B" "Phi1.5B/add_small" 2e-4 8 1 128 8

python $PYTHON_SCRIPT "google/gemma-2b" "dataset/addition_dataset_text_Small" "Gemma2B" "Gemma2B/add_small" 2e-4 8 1 128 8

python $PYTHON_SCRIPT "google/recurrentgemma-2b" "dataset/addition_dataset_text_Small" "RGemma2B" "RGemma2B/add_small" 2e-4 8 2 128 8

python $PYTHON_SCRIPT "state-spaces/mamba-2.8b-hf" "dataset/addition_dataset_text_Small" "Mamba2.8B" "Mamba2.8B/add_small" 2e-4 8 0 128 8

python $PYTHON_SCRIPT "stabilityai/stablelm-zephyr-3b" "dataset/addition_dataset_text_Small" "Zephyr3B" "Zephyr3B/add_small" 2e-4 8 1 128 8

python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/addition_dataset_text_Small" "Mamba7B" "Mamba7B/add_small" 2e-4 8 0 128 8

python $PYTHON_SCRIPT "meta-llama/Llama-2-7b-hf" "dataset/addition_dataset_text_Small" "Llama7B" "Llama7B/add_small" 2e-4 8 1 128 8

python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/addition_dataset_text_Small" "Llama8B" "Llama8B/add_small" 2e-4 8 1 128 8

python $PYTHON_SCRIPT "mistralai/Mistral-7B-v0.1" "dataset/addition_dataset_text_Small" "Mistral7B" "Mistral7B/add_small" 2e-4 8 1 128 8

python $PYTHON_SCRIPT "google/gemma-7b" "dataset/addition_dataset_text_Small" "Gemma7B" "Gemma7B/add_small" 2e-4 8 1 128 8





#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_Small" "Mamba7B_mul_small" "Mamba7B/mul_small" 1e-4 20 0 128 8
#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/addition_dataset_text_Large" "Llama8B_add_large" "Llama8B/add_large" 1e-4 20 1 175 8
#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_Large" "Mamba7B_mul_large" "Mamba7B/mul_large" 1e-4 20 0 175 8
#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_Large" "Llama8B_mul_large" "Llama8B/mul_large" 1e-4 20 1 175 8

#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_Small" "Mamba7B_mul_small" "Mamba7B/mul_small" 5e-5 40 0 128 8
#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/addition_dataset_text_Large" "Llama8B_add_large" "Llama8B/add_large" 5e-5 40 1 175 8
#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_Large" "Mamba7B_mul_large" "Mamba7B/mul_large" 5e-5 40 0 175 8
#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_Large" "Llama8B_mul_large" "Llama8B/mul_large" 5e-5 40 1 175 8

#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_Small" "Mamba7B_mul_small" "Mamba7B/mul_small" 2e-4 8 0 128 8
#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/addition_dataset_text_Large" "Llama8B_add_large" "Llama8B/add_large" 2e-4 8 1 175 8
#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" "dataset/mul_dataset_text_Large" "Mamba7B_mul_large" "Mamba7B/mul_large" 2e-4 8 0 175 8
#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/mul_dataset_text_Large" "Llama8B_mul_large" "Llama8B/mul_large" 2e-4 8 1 175 8

PYTHON_SCRIPT=run_all_inf.py
#python $PYTHON_SCRIPT "meta-llama/Llama-2-7b-hf" "dataset/addition_dataset_text_Large" "Llama7B_add_large" "Llama7B/add_large" 2e-4 8 1 175 2

#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" "dataset/addition_dataset_text_Large" "Llama8B_add_large" "Llama8B/add_large" 2e-4 8 1 175 2

#python $PYTHON_SCRIPT "mistralai/Mistral-7B-v0.1" "dataset/addition_dataset_text_Large" "Mistral7B_add_large" "Mistral7B/add_large" 2e-4 8 1 175 2

#python $PYTHON_SCRIPT "google/gemma-7b" "dataset/addition_dataset_text_Large" "Gemma7B_add_large" "Gemma7B/add_large" 2e-4 8 1 175 2


PYTHON_SCRIPT=run_all_inf2.py
#python $PYTHON_SCRIPT "state-spaces/mamba-1.4b-hf" 1
#python $PYTHON_SCRIPT "microsoft/phi-1_5" 0
#python $PYTHON_SCRIPT "google/gemma-2b" 0
#python $PYTHON_SCRIPT "google/recurrentgemma-2b" 0
#python $PYTHON_SCRIPT "state-spaces/mamba-2.8b-hf"  1
#python $PYTHON_SCRIPT "stabilityai/stablelm-zephyr-3b" 0
#python $PYTHON_SCRIPT "tri-ml/mamba-7b-rw" 1
#python $PYTHON_SCRIPT "meta-llama/Llama-2-7b-hf" 0
#python $PYTHON_SCRIPT "meta-llama/Meta-Llama-3-8B" 0
#python $PYTHON_SCRIPT "mistralai/Mistral-7B-v0.1" 0
#python $PYTHON_SCRIPT "google/gemma-7b" 0