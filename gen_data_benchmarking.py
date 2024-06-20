

from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from BidirectionalUtils.BidirectionalUtils import BidirectionalSwitch

# from decoderOnly.transponder import Transponder
from utils.SamplingStrategies import GreedySamplingStrategy, TopKSamplingStrategy
from utils.PonderingCriteria import EntropyPonderingCriteria
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Dataset
import zstandard as zstd
import requests
import pandas as pd
import torch
from torch import nn
import logging
from pathlib import Path
import random
import copy
import sys
import os
import gc

class LlamaBidirectionalSwitch(BidirectionalSwitch):
    @property
    def __backbone(self):
        return self.llm.model

    def __enter__(self):
        for transformer_decoder_layer in self.__backbone.layers:
            transformer_decoder_layer.self_attn.force_bidirectional = True

    def __exit__(self, type, value, traceback):
        for transformer_decoder_layer in self.__backbone.layers:
            transformer_decoder_layer.self_attn.force_bidirectional = False

model_checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

class GenData:
    def __init__(self):
        self.logger = logging.getLogger("transponder_logger")
        self.LLM_VOCAB_SIZE = 32000
        self.LLM_HIDDEN_SIZE = 2048
        self.sampling_percent = 0.2
        # self.batch_size = 16

        # mlm_head is not used for generating training data
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def generate_data(self, input_df: pd.DataFrame, output_data_path: str, current_idx: int, use_gpu_:bool=False):
        return self.__generate_data(input_df, output_data_path, current_idx, use_gpu_)
    
    def get_hidden_layer(self, prompt: list):
        return self.__get_hidden_layer(prompt)
    
    def __generate_data(self, input_df: pd.DataFrame, output_data_path: str, current_idx: int, use_gpu_:bool=False):
        # input_df = pd.read_csv(input_data_path)
        print(input_df.columns)
        result_df = input_df.text
        input_prompts = list(set(result_df.tolist()))
        input_prompts = input_prompts[:500]
        # print(len(input_prompts))
        print(f"Total prompts: {len(input_prompts)}")
        # for dummy 
        # input_prompts = ["I am Iron man. Am I made of iron? Wait, I think I am.", "List the top 10 tallest mountains in the world and list their locations."]

        groundtruth_token_total_list = []
        # prompt_tensor = []
        for idx, prompt in enumerate(input_prompts):
            print(f"Current prompt number: {idx}")
            # print(f"The prompt: {prompt}")
            try:
                last_tokens_last_hidden_state_tensor, last_tokens_last_hidden_state_tensor_nobi, masked_token_index_tensor, masked_token_list = self.__generate_row_data(prompt, use_gpu_)
                try:
                    hidden_state_tensor = torch.cat((hidden_state_tensor, last_tokens_last_hidden_state_tensor), dim=0)
                    hidden_state_tensor_nobi = torch.cat((hidden_state_tensor_nobi, last_tokens_last_hidden_state_tensor_nobi), dim=0)
                    groundtruth_tensor = torch.cat((groundtruth_tensor, masked_token_index_tensor), dim=0)
                except:
                    hidden_state_tensor = last_tokens_last_hidden_state_tensor
                    hidden_state_tensor_nobi = last_tokens_last_hidden_state_tensor_nobi
                    groundtruth_tensor = masked_token_index_tensor
                groundtruth_token_total_list += masked_token_list
                # prompt_tensor += updated_prompt_tensor
                # print(f"Total samples generated from this prompt: {len(masked_token_list)}\n")
            except Exception as e:
                print(f"Prompt skipped!")
                print(f"Error: {e}")

        # output_df = pd.DataFrame({'prompt': prompt_tensor, 'groundtruth_token': groundtruth_token_total_list})
        # output_df.to_csv(f"{output_data_path}.csv", index=False)
        os.makedirs(f"{output_data_path}/hidden_state", exist_ok=True)
        os.makedirs(f"{output_data_path}/hidden_state_nobi", exist_ok=True)
        os.makedirs(f"{output_data_path}/groundtruth", exist_ok=True)
        torch.save(hidden_state_tensor, f"{output_data_path}/hidden_state/hidden_state_{current_idx}.pt")
        torch.save(hidden_state_tensor_nobi, f"{output_data_path}/hidden_state_nobi/hidden_state_nobi_{current_idx}.pt")
        torch.save(groundtruth_tensor, f"{output_data_path}/groundtruth/groundtruth_index_{current_idx}.pt")
        # print(f"Combined hidden layer shape: {hidden_state_tensor.shape}")
        # print(f"Combined hidden layer shape nobi: {hidden_state_tensor_nobi.shape}")
        # print(f"Combined groundtruth one hot tensor shape: {groundtruth_tensor.shape}")
        print(f"Total generated samples: {len(groundtruth_tensor)}")
        return len(groundtruth_tensor)

    def __generate_row_data(self, prompt: str, use_gpu_:bool=False):
        masked_token_list = []
        backbone_inputs = self.tokenizer(prompt, return_tensors="pt")
        total_tokens_in_prompt = int(backbone_inputs['input_ids'].shape[1])
        random_numbers = [random.randint(1, total_tokens_in_prompt-1) for _ in range(int(self.sampling_percent*total_tokens_in_prompt))]
        token_indexes = []
        masked_token_list = []

        # Mask the tokens in given positions
        for pos in random_numbers:
            token_indexes.append(torch.tensor([backbone_inputs['input_ids'][0][pos]]))
            masked_token_list.append(self.tokenizer.batch_decode([backbone_inputs['input_ids'][0][pos]])[0])
            (backbone_inputs['input_ids'])[0][pos] = 0
            (backbone_inputs['attention_mask'])[0][pos] = 0

        last_hidden_state, last_hidden_state_nobi = self.__get_hidden_layer(
            prompt = [prompt], 
            original_backbone_inputs = backbone_inputs,
            use_gpu_ = use_gpu_
        )
        # print(f"Total Hidden layer size:{last_hidden_state.shape}")
        # print(f"Total Hidden layer size nobi:{last_hidden_state_nobi.shape}")

        # new_prompt = self.tokenizer.batch_decode([backbone_inputs['input_ids'][0]])[0]
        for pos in random_numbers:
            last_token_last_hidden_state = last_hidden_state[:,pos,:]
            last_token_last_hidden_state_nobi = last_hidden_state_nobi[:,pos-1,:]

            # print(f"Each pos Hidden layer size:{last_token_last_hidden_state.shape}")
            # print(f"Each pos Hidden layer size nobi:{last_token_last_hidden_state_nobi.shape}")
            try:
                last_tokens_last_hidden_state_tensor = torch.cat((last_tokens_last_hidden_state_tensor, last_token_last_hidden_state), dim=0)
                last_tokens_last_hidden_state_tensor_nobi = torch.cat((last_tokens_last_hidden_state_tensor_nobi, last_token_last_hidden_state_nobi), dim=0)
            except:
                last_tokens_last_hidden_state_tensor = last_token_last_hidden_state
                last_tokens_last_hidden_state_tensor_nobi = last_token_last_hidden_state_nobi
        masked_token_index_tensor = torch.tensor(token_indexes)

        # print(f"All pos in row layer size:{last_tokens_last_hidden_state_tensor.shape}")
        # print(f"All pos in row layer size nobi:{last_tokens_last_hidden_state_tensor_nobi.shape}")

        return last_tokens_last_hidden_state_tensor, last_tokens_last_hidden_state_tensor_nobi, masked_token_index_tensor, masked_token_list

    def __get_hidden_layer(self, prompt: list, original_backbone_inputs = None, use_gpu_:bool=False):
        backbone_inputs = copy.deepcopy(original_backbone_inputs)
        if backbone_inputs is None:
            backbone_inputs = self.tokenizer(prompt, return_tensors="pt")

        if use_gpu_:
            backbone_inputs['attention_mask'] = backbone_inputs['attention_mask'].to('cuda')
            backbone_inputs['input_ids'] = backbone_inputs['input_ids'].to('cuda')

            model_cuda = model.to('cuda')
            with torch.no_grad():
                with LlamaBidirectionalSwitch(model_cuda):
                    model_out = model_cuda(**backbone_inputs, output_hidden_states=True)
                    # last_token_last_hidden_state_gpu = model_out.hidden_states[-1][:,current_token_pos,:]
                    last_token_last_hidden_state_gpu = model_out.hidden_states[-1]
                    last_token_last_hidden_state = last_token_last_hidden_state_gpu.to('cpu')

                    del model_out
                    del last_token_last_hidden_state_gpu
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # without bidirectional switch
                model_out_nobi = model_cuda(**backbone_inputs, output_hidden_states=True)
                # last_token_last_hidden_state_gpu = model_out.hidden_states[-1][:,current_token_pos,:]
                last_token_last_hidden_state_gpu_nobi = model_out_nobi.hidden_states[-1]
                last_token_last_hidden_state_nobi = last_token_last_hidden_state_gpu_nobi.to('cpu')

                del model_out_nobi
                del last_token_last_hidden_state_gpu_nobi
                gc.collect()
                torch.cuda.empty_cache()

        else:
            with LlamaBidirectionalSwitch(model):
                model_out = model(**backbone_inputs, output_hidden_states=True)
                last_token_last_hidden_state = model_out.hidden_states[-1]
                # last_token_last_hidden_state = model_out.hidden_states[-1][:,current_token_pos,:]
            
            # without bidirectional switch
            model_out_nobi = model(**backbone_inputs, output_hidden_states=True)
            last_token_last_hidden_state_nobi = model_out_nobi.hidden_states[-1]

        return last_token_last_hidden_state, last_token_last_hidden_state_nobi
    

# Additional functions
def get_hugging_face_dataset(path : str, url : str):
    zstd_file_path = 'temp_dataset.zst'
    response = requests.get(url)
    with open(zstd_file_path, 'wb') as f:
        f.write(response.content)
    decompressed_file_path = 'temp_dataset.json'
    with open(zstd_file_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with open(decompressed_file_path, 'wb') as decompressed_file:
            dctx.copy_stream(compressed_file, decompressed_file)
    dataset = load_dataset('json', data_files=decompressed_file_path)
    # print(f"Number of examples: {len(dataset['train'])}")
    # print(f"Features: {dataset['train'].features}")

    # dataset = load_dataset("json", path, data_files=data_files, split="train")
    dataset_df = dataset['train'].to_pandas()
    os.remove(zstd_file_path)
    os.remove(decompressed_file_path)
    return dataset_df

def random_gen_bool(prob_true=0.2):
    return random.random() < prob_true

if __name__=="__main__":
    input_data = "data_benchmarking/input_data.csv"
    log_file = "data_benchmarking/log.txt"
    os.makedirs("data_benchmarking", exist_ok=True)
    output_data_path = "data_benchmarking"
    gen_obj = GenData()

    for idx in range(1):
        print("Getting the input data.")
        url = f"https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk1/example_train_{idx}.jsonl.zst"
        with open(log_file, 'a+') as file:
            file.write(url)

        try:
            dataset_df = get_hugging_face_dataset(path="cerebras/SlimPajama-627B", url=url)
            total_samples_generated = gen_obj.generate_data(
                input_df=dataset_df,
                output_data_path=output_data_path,
                current_idx = idx,
                use_gpu_=True
            )
            log_message = f"\nTotal samples generated from this file: {total_samples_generated}\n\n"
        except Exception as e:
            log_message = f"\n{str(e)}\n\n"

        with open(log_file, 'a+') as file:
            file.write(log_message)
    