import torch
class MaskedTokenLoader:

    def __init__(self, data_path,label_path):
        self.x=self._extract_data(data_path)
        self.y=self._extract_data(label_path)


    def _extract_data(self,data_path):

        x=torch.load(data_path)
        return x
    
    def __len__(self):
        return(self.x.shape[0])


    def __getitem__(self,index):

        return self.x[index], self.y[index]
    
        

        

        

