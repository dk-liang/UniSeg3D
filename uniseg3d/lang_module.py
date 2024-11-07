import os
import sys
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mmdet3d.registry import MODELS
from mmdet3d.models.layers import MLP

from .openclip_backbone import OpenCLIPBackboneText

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

@MODELS.register_module()
class LangModule(nn.Module):
    def __init__(self,input_channels=768,
                 model_name = 'convnext_large_d_320', 
                 init_cfg=dict(type='clip_pretrain', 
                      checkpoint='laion2B-s29B-b131K-ft-soup'),
                 fix=True,
                 out_features = 32):
        super().__init__() 
        self.out_features = out_features
        self.input_channels = input_channels
        self.fix = fix
        
        self.text_encoder = OpenCLIPBackboneText(model_name=model_name, init_cfg=init_cfg)
        if self.fix:
            for name, param in self.text_encoder.named_parameters():
                    param.requires_grad = False
                    
        self.Mlp = Mlp(in_features = self.text_encoder.text_proj.shape[1], hidden_features = self.text_encoder.text_proj.shape[1], out_features = self.out_features)
        

    def forward(self, text_token):
        """
        encode the input descriptions
        """
        word_embeddings = []
        for i in range(len(text_token)): 
            if len(text_token[i])==0:
                word_embeddings.append(torch.tensor([]))
            else:
                lang_last = self.text_encoder(text_token[i])
                embedding = self.Mlp(lang_last)
                word_embeddings.append(embedding)

        return word_embeddings