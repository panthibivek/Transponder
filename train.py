from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import torch
from dataloader import MaskedTokenLoader
import torch.optim as optim


llm_checkpoint = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' 
llm = AutoModelForCausalLM.from_pretrained(llm_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(llm_checkpoint)

# Define the number of classes as the size of the vocabulary
num_labels = len(tokenizer.vocab)


class RegressionNetwork(nn.Module):
    def __init__(self, config,model):
        super().__init__()

        # pre-trained LlamaForCausalLM
        self.llama_model = model

        # Initialize regression layer with weights from lm_head
        self.regression_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.regression_head.weight = self.llama_model.lm_head.weight
        self.regression_head.bias = self.llama_model.lm_head.bias

    def forward(self,last_hidden_state):
        # Get distribution

        regression_output = self.regression_head(last_hidden_state)

        return regression_output

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__=='__main__':

    config_dict = {
        'vocab_size': 32000,
        'hidden_size': 2048,
    }

    input_data_path='data/output_data_hidden_state.pt'
    output_data_path='data/output_data_groundtruth_one_hot.pt'

    config=dotdict(config_dict)
    mlm=RegressionNetwork(config,llm)
    print(mlm)
    total_params = sum(p.numel() for p in mlm.regression_head.parameters())
    print(f"Number of parameters: {total_params}")

    #llama_token_classifier=LlamaForTokenClassification(llm,mlm)

    masked_token_loader=MaskedTokenLoader(input_data_path,output_data_path)
    criterion = nn.CrossEntropyLoss()
    ##only the regression head, 65M parameters
    optimizer = optim.Adam(mlm.regression_head.parameters(), lr=0.001)
    epoch=5
    for i in range(epoch):
        print(f'Training on Epoch {i}')
        for inputs,labels in masked_token_loader:
            optimizer.zero_grad()
            outputs = mlm(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    torch.save(mlm.state_dict(), 'token_classifier.pt')




