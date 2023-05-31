from model.generator import GFPGANv1Clean
import os 
import numpy as np 
import cv2 
import torch 

class Infer:
    def __init__(self,model_path):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.netG = GFPGANv1Clean(out_size=512,
                                channel_multiplier=2,
                                fix_decoder=False,
                                input_is_latent=True,
                                different_w=True,
                                sft_half=True
                                ).to(self.device)
        self.netG.load_state_dict(torch.load(model_path)['g_ema'])
        self.netG.eval()

    def run(self,img_paths,save):
        os.makedirs(save,exist_ok=True)
        for i,img_path in enumerate(img_paths):
            self.run_single(img_path,save)
            print('\rhave done %06d'%i,end='',flush=True)
        print()

    def run_single(self,img_path,save):

        inp = self.preprocess(img_path)
        with torch.no_grad():
            oup,_ = self.netG(inp,)
        oup = self.postprocess(oup)
        cv2.imwrite(os.path.join(save,os.path.basename(img_path)),oup)

    def preprocess(self,img):
        if isinstance(img,str):
            img = cv2.imread(img)
        img = cv2.resize(img,(512,512))
        img = img.astype(np.float32)[...,::-1] / 127.5 - 1
        return torch.from_numpy(img.transpose(2,0,1)[np.newaxis,...]).to(self.device)
    
    def postprocess(self,img):
       
        return (torch.clip(img,-1,1)[0].permute(1,2,0).cpu().numpy()[...,::-1]+1)*127.5


if __name__ == "__main__":
    model = Infer('checkpoint/GFPGAN/final.pth')

    base = ''

    fn = lambda x:[os.path.join(x,f) for f in os.listdir(x)]

    img_paths = fn(base)

    save = ''

    model.run(img_paths,save)