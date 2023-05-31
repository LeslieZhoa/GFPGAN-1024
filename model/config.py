'''
@author LeslieZhao
@date 20230531
'''
class Params:
    def __init__(self):
       
        self.name = 'GFPGAN'
        self.mode = 'decoder'
        self.pretrain_path = None
        self.scratch_gan_path = 'pretrained_models/GFPGANv1.4.pth'
        self.scratch_d_path = 'pretrained_models/d.pth'
        self.scratch_left_eye_path = 'pretrained_models/GFPGANv1_net_d_left_eye.pth'
        self.scratch_right_eye_path = 'pretrained_models/GFPGANv1_net_d_right_eye.pth'
        self.scratch_mouth_path = 'pretrained_models/GFPGANv1_net_d_mouth.pth'
        self.id_model = 'pretrained_models/arcface_resnet18.pth'
        self.img_root = "" # ffhq data
        self.train_hq_root = "" # 1024 data by some api
        self.train_lq_root = '' # custom data
        self.train_lmk_base = '' # lmk info
        self.val_lmk_base = ''
        self.val_lq_root = ''
        self.val_hq_root = ''
        # data 
        self.crop_components = False
        self.eye_enlarge_ratio = 1.4


        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.out_size = 512
        self.size = 1024

        self.blur_kernel_size = 41
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_sigma = [0.1, 10]
        self.downsample_range = [0.8, 8]
        self.noise_range = [0, 20]
        self.jpeg_range = [60, 100]
        self.use_flip = True
        
        self.use_buffer = False
        # model 
        self.channel_multiplier = 2
        self.fix_decoder = False 
        self.input_is_latent = True 
        self.different_w = True 
        self.sft_half = True

        self.id_block = 'IRBlock'
        self.id_layers = [2,2,2,2]
        self.id_use_se = False
       
        self.g_lr = 1e-4
        self.d_lr = 4e-5
        self.beta1 = 0.
        self.beta2 = 0.99

        # loss 
        self.perloss = True
        self.pixloss = True
        self.idloss = True
        self.featloss = True
        


        self.layer_weights = {'conv1_2': 0.1,
            'conv2_2': 0.1,
            'conv3_4': 1,
            'conv4_4': 1,
            'conv5_4': 1}
        # before relu
      
        self.vgg_type = 'vgg19'
        self.use_input_norm = True

        # self.lambda_perceptual = 1
        # self.lambda_style = 50
        # self.lambda_id = 10
        # self.lambda_fm = 1
        # self.lambda_gan_part = 1e-1
        # self.lambda_gan = 1e-1
        # self.lambda_l1 = 10
        # self.comp_style_weight = 10

        self.lambda_perceptual = 1
        self.lambda_style = 50
        self.lambda_id = 10
        self.lambda_fm = 1
        self.lambda_gan_part = 1e-1
        self.lambda_gan = 1e-1
        self.lambda_l1 = 10
        self.comp_style_weight = 20

        self.range_norm = True
        self.criterion = 'l1'

        self.gan_type = 'wgan_softplus'
        self.part_gan_type = 'vanilla'
        

        # optim 
        self.net_d_reg_every = 16
        self.r1_reg_weight = 10


