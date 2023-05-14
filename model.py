import torch
import torch.nn as nn
from transformers import AdamW
import numpy as np
import pandas as pd

# include model_fin in the Reformer_Arch so it can be inherited?

# freeze all the parameters
for param in model.parameters():
    param.requires_grad = False
    class Reformer_Arch(nn.Module):

        def __init__(self, model):
            super(Reformer_Arch, self).__init__()
        
            self.model = model
        
            self.dropout = nn.Dropout(0.1)
      
            self.relu =  nn.ReLU()

            self.fc1 = nn.Linear(768,512)
      
            self.fc2 = nn.Linear(512,2)

            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, sent_id, mask):
          
            _, cls_hs = self.model(sent_id, attention_mask=mask, return_dict=False)
      
            x = self.fc1(cls_hs)

            x = self.relu(x)

            x = self.dropout(x)

            x = self.fc2(x)
      
            x = self.softmax(x)

            return x
    
model_fin = Reformer_Arch(model)

model_fin = model_fin.to(device)