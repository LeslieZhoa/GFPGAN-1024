
'''
@author LeslieZhao
@date 20230531
'''
import torch 
from dataloader.GFPLoader import  GFPData
import numpy as np
from functools import partial


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def my_collate(batch,dataset):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    if len_batch > len(batch): # source all the required samples from the original dataset at random
        diff = len_batch - len(batch)
        while diff > 0:
            data = dataset[np.random.randint(0, len(dataset))]
            if data is not None:
                batch.append(data)
                diff -= 1
            

    return torch.utils.data.dataloader.default_collate(batch)

def requires_grad(model, flag=True):
    if model is None:
        return 
    for p in model.parameters():
        p.requires_grad = flag
def need_grad(x):
    x = x.detach()
    x.requires_grad_()
    return x

def init_weights(model,init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

        elif hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
                
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            
    model.apply(init_func)
    
def accumulate(model1, model2, decay=0.999,use_buffer=False):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
   
    if use_buffer:
        for p1,p2 in zip(model1.buffers(),model2.buffers()):
            p1.detach().copy_(decay*p1.detach()+(1-decay)*p2.detach())
def setup_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_data_loader(args):
    
    train_data = GFPData(dist=args.dist,
                    mean=args.mean,
                    std=args.std,
                    blur_kernel_size=args.blur_kernel_size,
                    kernel_list=args.kernel_list,
                    kernel_prob=args.kernel_prob,
                    blur_sigma=args.blur_sigma,
                    downsample_range=args.downsample_range,
                    noise_range=args.noise_range,
                    jpeg_range=args.jpeg_range,
                    use_flip=args.use_flip,
                    hq_root=args.train_hq_root,
                    lq_root=args.train_lq_root,
                    img_root=args.img_root,
                    crop_components=args.crop_components,
                    lmk_base=args.train_lmk_base,
                    size=args.size,
                    eye_enlarge_ratio=args.eye_enlarge_ratio
                    ,eval=False )
    
    test_data = None
    if args.eval: 
        test_data = GFPData(dist=args.dist,
                    mean=args.mean,
                    std=args.std,
                    blur_kernel_size=args.blur_kernel_size,
                    kernel_list=args.kernel_list,
                    kernel_prob=args.kernel_prob,
                    blur_sigma=args.blur_sigma,
                    downsample_range=args.downsample_range,
                    noise_range=args.noise_range,
                    jpeg_range=args.jpeg_range,
                    use_flip=args.use_flip,
                    hq_root=args.val_hq_root,
                    lq_root=args.val_lq_root,
                    crop_components=args.crop_components,
                    lmk_base=args.val_lmk_base,
                    size=args.size,
                    eye_enlarge_ratio=args.eye_enlarge_ratio
                    ,eval=True )

    use_collate = partial(my_collate,dataset=train_data)
    train_loader = torch.utils.data.DataLoader(train_data, 
                        collate_fn=use_collate,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True) #####myadd just test

    
    test_loader = None if test_data is None else \
        torch.utils.data.DataLoader(
                        test_data,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True
                    )
    return train_loader,test_loader,len(train_loader) 



def merge_args(args,params):
   for k,v in vars(params).items():
      setattr(args,k,v)
   return args

def convert_img(img,unit=False):
   
    img = (img + 1) * 0.5
    if unit:
        return torch.clamp(img*255+0.5,0,255)
    
    return torch.clamp(img,0,1)


def convert_flow_to_deformation(flow):
    r"""convert flow fields to deformations.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b,c,h,w = flow.shape
    flow_norm = 2 * torch.cat([flow[:,:1,...]/(w-1),flow[:,1:,...]/(h-1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0,2,3,1)
    return deformation

def make_coordinate_grid(flow):
    r"""obtain coordinate grid with the same size as the flow filed.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """    
    b,c,h,w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed    

    
def warp_image(source_image, deformation):
    r"""warp the input image according to the deformation

    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """ 
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear')
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation) 


def add_list(x):
    r = None
   
    for v in x:
        r = v if r is None else r + v
    return r 

def compute_cosine(x):
    mm = np.matmul(x,x.T)
    norm = np.linalg.norm(x,axis=-1,keepdims=True)
    dis = mm / np.matmul(norm,norm.T)
    return  dis - np.eye(*dis.shape)

def compute_graph(cos_dis):
    index = np.where(np.triu(cos_dis) >= 0.68)
   
    # dd存放最终的图
    # vis存放各节点以及他们的root
    dd = {}
    vis = {}


    for i in np.unique(index[0]):
        
        # 此步用来存放根，因为是上三角，如果存在vis中，必不为root
        if i not in vis:
            vis[i] = i
            dd[vis[i]] = [i]
        
        for j in index[1][index[0]==i]:
            # 遍历行，不存在vis中的才为没有加入的，要将其root指向最终的root
            # 如果i为root，则vis[vis[i]] 为本身，如果i为节点，则vis[vis[i]]必为root
            # vis存放的k:val 只有两种形式 root:root, val:root
            if j not in vis:
                vis[j] = vis[vis[i]]
                dd[vis[i]] = dd.get(vis[i],[]) + [j]
            
            # 如果两簇有关联，进行合并
            elif j in vis and vis[vis[j]] != vis[vis[i]]:
                old_root = vis[vis[j]]
                for v in dd[vis[vis[j]]]:
                    dd[vis[vis[i]]] += [v]
                    vis[v] = vis[vis[i]] 
                del dd[old_root]

    for k,v in dd.items():
        dd[k] = list(set(v+[k]))
    return dd,index