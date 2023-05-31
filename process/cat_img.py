import os 
import cv2 
import numpy as np 


def run(save,*bases,size=None):
    os.makedirs(save,exist_ok=True)
    names = os.listdir(bases[0])

    if size is None:
        img = cv2.imread(os.path.join(bases[0],names[0]))
        h,w,_ = img.shape 
        size = (w,h)
    
    for i,name in enumerate(names):
        img = []
        for base in bases:
            img.append(cv2.resize(cv2.imread(os.path.join(base,name)),size))

        img = np.concatenate(img,1)
        cv2.imwrite(os.path.join(save,name),img)
        print('\rhave done %06d'%i,end='',flush=True)
    print()

if __name__ == "__main__":
    base1 = ''
    base2 = ''
    base3 = ''
    base4 = ''
    save = ''
    size = (1024,1024)
    run(save,base1,base2,base3,base4,size=size)
