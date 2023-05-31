import torch
from torch import nn

import numpy as np
import sys 
sys.path.append('.')
sys.path.append('..')
from model.generator4pt import GFPGANv1Clean

import pdb

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class BaseModel(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        load_model_path = kwargs['load_model_path']
        self.pre_state_dict = torch.load(load_model_path, map_location='cpu')
       
    def forward(self,x):
        pass

    @staticmethod
    def get_input(batch_size=1,img_size=256):
        x = torch.randn(batch_size,3,img_size,img_size,requires_grad=True)
        return x
    
    

class SrModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = GFPGANv1Clean(out_size=512,
                                channel_multiplier=2,
                                fix_decoder=False,
                                input_is_latent=True,
                                different_w=True,
                                sft_half=True
                                )

        self.model.load_state_dict(self.pre_state_dict['g_ema'], strict=True)
        self.model.eval()
        self.model = self.model

    def forward(self, x):
        
        hq_img,_ = self.model(x,return_latents=False, return_rgb=False)
        return hq_img

    @staticmethod
    def get_input(batch_size=1,img_size=512):
        x = torch.randn(batch_size,3,img_size,img_size)
        
        return x
    
    
def convert_model(Model,output_path,**kwargs):

    
    torch_model = Model(**kwargs)
    inputs = Model.get_input()
    # inputs2 = Model.get_input(batch_size=2)
    
    # model = torch_model.cuda()
    # # pdb.set_trace()
    # oup = model(inputs.cuda())
    traced_script_module = torch.jit.trace(torch_model, inputs)
    traced_script_module.save(output_path)   
   

def get_model(Model,model_path):
    model = torch.jit.load(model_path)
    # ort_session = ort.InferenceSession(model_path)
    model.eval()
    return model


def test_model(Model,model_path):

    model = get_model(Model,model_path)
    inputs = Model.get_input()
    
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
    outputs = model(inputs)
    return outputs
    

if __name__ == "__main__":
    
    model_path = 'checkpoint/GFPGAN/final.pth'
   
    convert_model(SrModel,"gfpgan1024.pt",
            load_model_path=model_path)
    
    # test_model(IDModel,"id_model.onnx")
    oup = test_model(SrModel,"gfpgan1024.pt")  
    