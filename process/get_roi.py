import sys 
sys.path.insert(0,'')
from LVT.LVT import Engine 
from LVT.utils import utils 
import time
from multiprocessing import Pool
import math
import argparse

import torch.distributed as dist
import torch
import numpy as np
import os
import cv2
import pdb

parser = argparse.ArgumentParser(description="InfoProcess")

parser.add_argument('--pool_num',default=5,type=int)



class InfoProcess:
    def __init__(self):
        model_dir = ''
        self.engine = Engine(face_lmk_path=model_dir + 'slpt-lmk.onnx')
        self.left_eye_index = list(range(60,68))
        self.right_eye_index = list(range(68,76))
        self.lip_index = list(range(76,88))

    def run(self,img_paths,save_base):
        os.makedirs(save_base,exist_ok=True)
        i = 0
        for img_path in img_paths:
            
            try:
                self.run_single(img_path,save_base)
            except:
                continue
            print('\r have done %06d'%i,end='',flush=True)
            i += 1
            
        print()

    def run_single(self,img_path,save_base):
       
        img = cv2.imread(img_path)

        # 98点
        h,w,_ = img.shape
        lmk = self.get_lmk(img,h,[0,0])[0]
        left_eye = self.get_area(lmk[self.left_eye_index])
        right_eye = self.get_area(lmk[self.right_eye_index])
        lip = self.get_area(lmk[self.lip_index])
        data = {'left_eye':left_eye,
                'right_eye':right_eye,
                'mouth':lip}
        name = os.path.splitext(os.path.basename(img_path))[0]
        np.save(os.path.join(save_base,name+'.npy'),data)
        
        
        return True


    def get_lmk(self,crop_img,crop_height,top):

        inp = self.engine.preprocess_lmk(crop_img)
        lmk = self.engine.get_lmk(inp)
        lmk = self.engine.postprocess_lmk(lmk,crop_height,top)
        return lmk 
    
    
    def get_area(self,lmk):
        left = np.min(lmk,0)
        right = np.max(lmk,0)
        mean = (left + right) / 2
        height = np.max(right-left)
        return np.array(mean.tolist()+[height]) 
    
    def draw_lmk(self,img,lmk):
        for p in lmk:
            cv2.circle(img,(int(p[0]),int(p[1])),2,[0,255,0])

        return img 
    
    def draw_rec(self,img,data):
        x,y,h = data 
        x1 = x - h/2
        y1 = y - h/2
        x2 = x + h/2 
        y2 = y + h/2 
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),[255,0,0],2)
        return img
    

def work(video_paths,save_base):
    process = InfoProcess()
    process.run(video_paths,save_base)


def print_error(value):
    print("error: ", value) 

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    args = parser.parse_args()
   
    fn = lambda x:[os.path.join(x,f) for f in os.listdir(x)]

    # img_paths = fn(base1) + fn(base2)
    base = ''
    img_paths = fn(base)
   
    save_base = ''
    
    # work(img_paths,save_base)
    length = len(img_paths)
   
    rank = int(os.environ.get('RANK','0'))
    world_size = int(os.environ.get('WORLD_SIZE','1'))
    print('*********************',rank,world_size)

    pool_num = args.pool_num

    dis1 = math.ceil(length / float(world_size))
    img_paths = img_paths[rank*dis1:(rank+1)*dis1]
    

    length = len(img_paths)
    dis = math.ceil(length/float(pool_num))

    if world_size > 1:
        dist.init_process_group(backend="nccl") # backbend='nccl'
        dist.barrier() # 用于同步训练
    signal = torch.tensor([0]).cuda()
    
    
    t1 = time.time()
    print('***************all length: %d ******************'%length)
    p = Pool(pool_num)
    for i in range(pool_num):
        p.apply_async(work, args = (
            img_paths[i*dis:(i+1)*dis],
            save_base,),error_callback=print_error)  
  
    p.close() 
    p.join()

    print("all the time: %s"%(time.time()-t1))
    signal = torch.tensor([1]).cuda()
    if world_size > 1:
        while True:

            dist.all_reduce(signal)
            value = signal.item()
            print('***************',value)
            if value >= world_size:
                break 
            else:
                dist.all_reduce(torch.tensor([0]).cuda())
                signal = torch.tensor([1]).cuda()
    
    