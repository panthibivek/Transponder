

from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from BidirectionalUtils.BidirectionalUtils import BidirectionalSwitch

# from decoderOnly.transponder import Transponder
from utils.SamplingStrategies import GreedySamplingStrategy, TopKSamplingStrategy
from utils.PonderingCriteria import EntropyPonderingCriteria
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import pandas as pd
import torch
from torch import nn
import logging
from pathlib import Path
import random
import copy
import sys
import os

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
        self.PONDER_CONTEXT_LENGTH = 5

        # mlm_head is not used for generating training data
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def generate_data(self, input_data_path: str, output_data_path: str, use_gpu_:bool=False):
        return self.__generate_data(input_data_path, output_data_path, use_gpu_)
    
    def get_hidden_layer(self, prompt: list):
        return self.__get_hidden_layer(prompt)
    
    def __generate_data(self, input_data_path: str, output_data_path: str, use_gpu_:bool=False):
        input_df = pd.read_csv(input_data_path)
        result_df = input_df.loc[input_df['correctness'] > 3, 'prompt']
        input_prompts = list(set(result_df.tolist()))
        input_prompts = input_prompts[:200]
        print(f"Total prompts: {len(input_prompts)}")
        # for dummy 
        input_prompts = ["I am Iron man. Am I made of iron? Wait, I think I am.", "List the top 10 tallest mountains in the world and list their locations. List the top 10 tallest mountains in the world and list their locations. List the top 10 tallest mountains in the world and list their locations."]

        groundtruth_token_total_list = []
        prompt_tensor = []
        for idx, prompt in enumerate(input_prompts):
            print(f"Current prompt number: {idx}")
            print(f"The prompt: {prompt}")
            last_tokens_last_hidden_state_tensor, masked_token_index_tensor, masked_token_list, updated_prompt_tensor = self.__generate_row_data(prompt, use_gpu_)
            try:
                try:
                    hidden_state_tensor = torch.cat((hidden_state_tensor, last_tokens_last_hidden_state_tensor), dim=0)
                    groundtruth_tensor = torch.cat((groundtruth_tensor, masked_token_index_tensor), dim=0)
                except:
                    hidden_state_tensor = last_tokens_last_hidden_state_tensor
                    groundtruth_tensor = masked_token_index_tensor
                groundtruth_token_total_list += masked_token_list
                prompt_tensor += updated_prompt_tensor
                print(f"Total samples generated: {len(updated_prompt_tensor)}\n")
            except:
                print(f"Prompt skipped!")

        output_df = pd.DataFrame({'prompt': prompt_tensor, 'groundtruth_token': groundtruth_token_total_list})
        output_df.to_csv(f"{output_data_path}.csv", index=False)
        torch.save(hidden_state_tensor, f"{output_data_path}_hidden_state.pt")
        torch.save(groundtruth_tensor, f"{output_data_path}_groundtruth_index.pt")
        print(f"Combined hidden layer shape: {hidden_state_tensor.shape}")
        print(f"Combined groundtruth one hot tensor shape: {groundtruth_tensor.shape}")
        print(f"Total generated samples: {len(prompt_tensor)}")

    def __generate_row_data(self, prompt: str, use_gpu_:bool=False):
        sampling_skip = 2
        masked_token_list = []
        updated_prompt_list = []
        # loop through all tokens
        ########
        backbone_inputs = self.tokenizer(prompt, return_tensors="pt")
        samples_from_each_prompt = (int(backbone_inputs['input_ids'].shape[1])-self.PONDER_CONTEXT_LENGTH-1)//sampling_skip
        for idx in range(0, samples_from_each_prompt, 1):
            # if random_gen_bool(0.1):
            if random_gen_bool(0.3):
                if int(backbone_inputs['input_ids'].shape[1]) <= 5:
                    break
                last_token_last_hidden_state, token_index, masked_token = self.__get_hidden_layer(
                    prompt = [prompt], 
                    original_backbone_inputs = backbone_inputs,
                    use_gpu_ = use_gpu_
                )
                try:
                    last_tokens_last_hidden_state_tensor = torch.cat((last_tokens_last_hidden_state_tensor, last_token_last_hidden_state), dim=0)
                    masked_token_index_tensor = torch.cat((masked_token_index_tensor, token_index), dim=0)
                except:
                    last_tokens_last_hidden_state_tensor = last_token_last_hidden_state
                    masked_token_index_tensor = token_index
                masked_token_list.append(masked_token)
                updated_prompt_list.append(prompt)

            current_token_pos = backbone_inputs['input_ids'].shape[1]-sampling_skip
            backbone_inputs = {
            'input_ids' : torch.tensor([backbone_inputs['input_ids'][0][:current_token_pos].tolist()]),
            'attention_mask' : torch.tensor([backbone_inputs['attention_mask'][0][:current_token_pos].tolist()])
            }
            prompt = self.tokenizer.batch_decode(backbone_inputs['input_ids'])[0]
        ########
        return last_tokens_last_hidden_state_tensor, masked_token_index_tensor, masked_token_list, updated_prompt_list

    def __get_hidden_layer(self, prompt: list, original_backbone_inputs = None, use_gpu_:bool=False):
        backbone_inputs = copy.deepcopy(original_backbone_inputs)
        if backbone_inputs is None:
            backbone_inputs = self.tokenizer(prompt, return_tensors="pt")

        current_token_pos = (backbone_inputs['attention_mask'].shape[1]-1)-self.PONDER_CONTEXT_LENGTH
        (backbone_inputs['attention_mask'])[0][current_token_pos] = 0
        masked_token = self.tokenizer.batch_decode([backbone_inputs['input_ids'][0][current_token_pos]])[0]
        # masked_token_one_hot_encoding = torch.zeros((1, self.LLM_VOCAB_SIZE))
        # masked_token_one_hot_encoding[0][int(backbone_inputs['input_ids'][0][current_token_pos])] = 1
        token_index = torch.tensor([backbone_inputs['input_ids'][0][current_token_pos]])
        self.logger.debug(f"backbone_inputs = {backbone_inputs}")

        if use_gpu_:
            backbone_inputs['attention_mask'] = backbone_inputs['attention_mask'].to('cuda')
            backbone_inputs['input_ids'] = backbone_inputs['input_ids'].to('cuda')

            model_cuda = model.to('cuda')
            with LlamaBidirectionalSwitch(model_cuda):
                model_out = model_cuda(**backbone_inputs, output_hidden_states=True)
                last_token_last_hidden_state = model_out.hidden_states[-1][:,current_token_pos,:]
        else:
            with LlamaBidirectionalSwitch(model):
                model_out = model(**backbone_inputs, output_hidden_states=True)
                last_token_last_hidden_state = model_out.hidden_states[-1][:,current_token_pos,:]

        print(f"Prompt: {prompt}")
        print(f"Returned hidden layers shape: {model_out.hidden_states[-1].shape}")
        print(f"last_token_last_hidden_state: {last_token_last_hidden_state.shape}")
        print(f"Masked Token: {masked_token}")
        print(f"Masked Token idx in vocab: {token_index}")
        print(f"Masked token reverse (assert): {self.tokenizer.batch_decode([token_index])[0]}")
        print(f"Current position: {current_token_pos}")
        print(f"Backbone input: {backbone_inputs}")
        print()
        return last_token_last_hidden_state, token_index, masked_token
    

# Additional functions
def get_hugging_face_dataset(path : str):
    dataset = load_dataset(path)
    dataset_df = pd.DataFrame(dataset['train'])
    return dataset_df

def random_gen_bool(prob_true=0.2):
    return random.random() < prob_true

if __name__=="__main__":
    input_data = "data/input_data.csv"
    output_data_path = "data/output_data"
    if not Path(input_data).is_file():
        print("Getting the input data.")
        dataset_df = get_hugging_face_dataset(path="nvidia/HelpSteer")
        dataset_df.to_csv(input_data, index=False)

    gen_obj = GenData()
    gen_obj.generate_data(
        input_data_path=input_data,
        output_data_path=output_data_path,
        # use_gpu_=True
    )
    
    # last_token_last_hidden_state, masked_token = gen_obj.get_hidden_layer(["I am Iron man. Am I made of iron? Wait I think I am."])
    # print(last_token_last_hidden_state.shape)