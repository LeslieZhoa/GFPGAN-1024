#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20230531
'''
import torch
from trainer.ModelTrainer import ModelTrainer
from model.generator import GFPGANv1Clean
from model.discriminator import StyleGAN2Discriminator,FacialComponentDiscriminator
from utils.utils import *
from model.loss import *
from model.third.arcface_arch import ResNetArcFace
import torch.nn.functional as F
import torch.nn as nn
import random
import torch.distributed as dist
from itertools import chain
from torchvision.ops import roi_align
import pdb

class GFPTrainer(ModelTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.netG = GFPGANv1Clean(out_size=args.out_size,
                                channel_multiplier=args.channel_multiplier,
                                fix_decoder=args.fix_decoder,
                                input_is_latent=args.input_is_latent,
                                different_w=args.different_w,
                                sft_half=args.sft_half
                                ).to(self.device)

        self.netD = StyleGAN2Discriminator(1024).to(self.device)
        self.left_eye_d = self.right_eye_d = self.mouth_d = None 

        if self.args.crop_components:
            self.left_eye_d = FacialComponentDiscriminator().to(self.device)
            self.right_eye_d = FacialComponentDiscriminator().to(self.device)
            self.mouth_d = FacialComponentDiscriminator().to(self.device)
            init_weights(self.left_eye_d,'xavier_uniform')
            init_weights(self.right_eye_d,'xavier_uniform')
            init_weights(self.mouth_d,'xavier_uniform')
            self.PartGANLoss = GANLoss(gan_type=args.part_gan_type,
                               loss_weight=args.lambda_gan_part)



        self.g_ema = GFPGANv1Clean(out_size=args.out_size,
                                channel_multiplier=args.channel_multiplier,
                                fix_decoder=args.fix_decoder,
                                input_is_latent=args.input_is_latent,
                                different_w=args.different_w,
                                sft_half=args.sft_half
                                ).to(self.device)
        self.g_ema.eval()
        
        init_weights(self.netD,'xavier_uniform')

        init_weights(self.netG,'xavier_uniform')

        self.optimG,self.optimD = self.create_optimizer() 

        if self.args.scratch:
            self.load_scratch()

        if args.pretrain_path is not None:
            self.loadParameters(args.pretrain_path)

        accumulate(self.g_ema,self.netG,0,args.use_buffer)
        
        self.netG,self.netG_module = self.use_ddp(self.netG,args.dist,True)
        self.netD,self.netD_module = self.use_ddp(self.netD,args.dist)
        self.left_eye_d,self.left_eye_d_module = self.use_ddp(self.left_eye_d,args.dist)
        self.right_eye_d,self.right_eye_d_module = self.use_ddp(self.right_eye_d,args.dist)
        self.mouth_d,self.mouth_d_module = self.use_ddp(self.mouth_d,args.dist)

        
       

        self.L1Loss = nn.L1Loss()

        if self.args.perloss:
            self.PerLoss = PerceptualLoss(args.layer_weights,args.vgg_type,
                                        args.use_input_norm,
                                        args.range_norm,
                                        perceptual_weight=args.lambda_perceptual,
                                        style_weight=args.lambda_style,
                                        criterion=args.criterion).to(self.device)
            requires_grad(self.PerLoss,False)
            self.PerLoss.eval()

        if self.args.idloss:
            self.idModel = ResNetArcFace(self.args.id_block,
                                         self.args.id_layers,
                                         self.args.id_use_se).to(self.device)
            self.idModel.load_state_dict(torch.load(self.args.id_model))
            requires_grad(self.idModel,False)
            self.idModel.eval()
        
        self.GANLoss = GANLoss(gan_type=args.gan_type,
                               loss_weight=args.lambda_gan)
        
        self.accum = 0.5 ** (32 / (10 * 1000))
    
    def create_optimizer(self):
        
        if self.args.mode == "decoder":
            g_params = self.netG.stylegan_decoder.parameters()
        elif self.args.mode == "encoder":
            requires_grad(self.netG.stylegan_decoder,False)
            g_params = [f for f in self.netG.parameters() if f.requires_grad]
        else:
            g_params = self.netG.parameters()
       
        g_optim = torch.optim.Adam(
                    g_params,
                    lr=self.args.g_lr,
                    betas=(self.args.beta1, self.args.beta2),
                    )
        
        if self.args.crop_components:
            d_params = chain(self.left_eye_d.parameters(),
                             self.right_eye_d.parameters(),
                             self.mouth_d.parameters(),
                             self.netD.parameters())
        else:
            d_params = self.netD.parameters()
        d_optim = torch.optim.Adam(
                    d_params,
                    lr=self.args.d_lr,
                    betas=(self.args.beta1, self.args.beta2),
                    )
        
        return  g_optim,d_optim

    
    def run_single_step(self, data, steps):
        
        self.netG.train()

        super().run_single_step(data,steps)
        

    def run_discriminator_one_step(self, data,step):
       
        requires_grad(self.netG, False)
        requires_grad(self.netD, True)
        requires_grad(self.left_eye_d, True)
        requires_grad(self.right_eye_d, True)
        requires_grad(self.mouth_d, True)

        lq,gt,loc_left_eye, loc_right_eye, loc_mouth = data
        fake, out_rgbs = self.netG(lq, return_rgb=False)
        
        
        self.optimD.zero_grad()
        d_loss,D_losses = self.compute_d_loss(fake,gt,loc_left_eye, loc_right_eye, loc_mouth)
        d_loss.backward()
        self.optimD.step()

        if step % self.args.net_d_reg_every == 0:
            gt.requires_grad = True
            real_pred,_ = self.netD(gt)
            l_d_r1 = r1_penalty(real_pred, gt)
            l_d_r1 = (self.args.r1_reg_weight / 2 * l_d_r1 * self.args.net_d_reg_every + 0 * real_pred[0])
            D_losses['l_d_r1'] = l_d_r1.detach().mean()
            l_d_r1.backward()
        
        self.d_losses = D_losses


    def run_generator_one_step(self, data,step):
        
    
        requires_grad(self.netG, True)
        
        requires_grad(self.netD, False)
        requires_grad(self.left_eye_d, False)
        requires_grad(self.right_eye_d, False)
        requires_grad(self.mouth_d, False)
       
        
        lq,gt,loc_left_eye, loc_right_eye, loc_mouth = data
        
        G_losses,loss,fake = \
            self.compute_g_loss(lq,gt,loc_left_eye, loc_right_eye, loc_mouth,step)
        # pdb.set_trace()
        self.optimG.zero_grad()
        loss.mean().backward()
        self.optimG.step()
        
        accumulate(self.g_ema,self.netG_module,self.accum,self.args.use_buffer)

        self.g_losses = G_losses
       

        self.generator = [lq.detach(),fake.detach(),gt.detach()]
        
    
    def evalution(self,test_loader,steps,epoch):
        
        loss_dict = {}
        index = random.randint(0,len(test_loader)-1)
        counter = 0
        with torch.no_grad():
            for i,data in enumerate(test_loader):
                
                data = self.process_input(data)
                lq,gt,loc_left_eye, loc_right_eye, loc_mouth = data
                G_losses,loss,fake = \
                        self.compute_g_loss(lq,gt,loc_left_eye, loc_right_eye, loc_mouth,steps)
                loss,D_losses = self.compute_d_loss(fake,gt,loc_left_eye, loc_right_eye, loc_mouth)
                G_losses = {**G_losses,**D_losses}
                for k,v in G_losses.items():
                    loss_dict[k] = loss_dict.get(k,0) + v.detach()
                if i == index and self.args.rank == 0 :
                    ema_oup,_ = self.g_ema(lq,return_rgb=False)
                    
                    show_data = [lq.detach(),
                                        fake.detach(),
                                        ema_oup.detach(),
                                        gt.detach()]
                    
                    self.val_vis.display_current_results(self.select_img(show_data,size=show_data[-1].shape[-1]),steps)
                counter += 1
        
       
        for key,val in loss_dict.items():
            loss_dict[key] /= counter

        if self.args.dist:
            # if self.args.rank == 0 :
            dist_losses = loss_dict.copy()
            for key,val in loss_dict.items():
                
                dist.reduce(dist_losses[key],0)
                value = dist_losses[key].item()
                loss_dict[key] = value / self.args.world_size

        if self.args.rank == 0 :
            self.val_vis.plot_current_errors(loss_dict,steps)
            self.val_vis.print_current_errors(epoch,steps,loss_dict,0)

        return loss_dict
    
    def compute_d_loss(self,fake,gt,loc_left_eye, loc_right_eye, loc_mouth):
    
        fake_d_pred,_ = self.netD(fake.detach())
        real_d_pred,_ = self.netD(gt)
        real_score = self.GANLoss(real_d_pred, True, is_disc=True)
        fake_score = self.GANLoss(fake_d_pred, False, is_disc=True)
        l_d = real_score + fake_score
        D_losses = {
            'd':l_d,
            'd_real':real_score,
            'd_fake':fake_score
        }


        if self.args.crop_components:
            real_left_eyes,\
            real_right_eyes,\
            real_mouths,\
            fake_left_eyes,\
            fake_right_eyes,\
            fake_mouths = self.get_roi_regions(fake,gt,[loc_left_eye, loc_right_eye, loc_mouth])
            # left eye
            fake_d_pred,_ = self.left_eye_d(fake_left_eyes.detach())
            real_d_pred,_ = self.left_eye_d(real_left_eyes)
            real_left_eye_score = self.PartGANLoss(real_d_pred,True, is_disc=True)
            fake_left_eye_score = self.GANLoss(fake_d_pred, False, is_disc=True)
            l_d_left_eye = real_left_eye_score + fake_left_eye_score 
            l_d += l_d_left_eye
            D_losses['d_left_eye_real'] = real_left_eye_score
            D_losses['d_left_eye_fake'] = fake_left_eye_score
            D_losses['d_left_eye'] = l_d_left_eye

            # right eye
            fake_d_pred,_ = self.right_eye_d(fake_right_eyes.detach())
            real_d_pred,_ = self.right_eye_d(real_right_eyes)
            real_right_eye_score = self.PartGANLoss(real_d_pred,True, is_disc=True)
            fake_right_eye_score = self.GANLoss(fake_d_pred, False, is_disc=True)
            l_d_right_eye = real_right_eye_score + fake_right_eye_score 
            l_d += l_d_right_eye
            D_losses['d_right_eye_real'] = real_right_eye_score
            D_losses['d_right_eye_fake'] = fake_right_eye_score
            D_losses['d_right_eye'] = l_d_right_eye

            # mouth
            fake_d_pred,_ = self.mouth_d(fake_mouths.detach())
            real_d_pred,_ = self.mouth_d(real_mouths)
            real_mouth_score = self.PartGANLoss(real_d_pred,True, is_disc=True)
            fake_mouth_score = self.GANLoss(fake_d_pred, False, is_disc=True)
            l_d_mouth = real_mouth_score + fake_mouth_score 
            l_d += l_d_mouth
            D_losses['d_mouth_real'] = real_mouth_score
            D_losses['d_mouth_fake'] = fake_mouth_score
            D_losses['d_mouth'] = l_d_mouth  
       
        return l_d,D_losses


    def compute_g_loss(self,lq,gt,loc_left_eye, loc_right_eye, loc_mouth,step):
        
        
        G_losses = {}
        loss = 0
        fake, out_rgbs = self.netG(lq, return_rgb=False)
        
        if self.args.pixloss:
            pixloss = self.L1Loss(fake,gt) * self.args.lambda_l1
            loss += pixloss 
            G_losses['g_pixloss'] = pixloss

        if self.args.perloss:

            l_g_percep, l_g_style = self.PerLoss(fake, gt)
            if l_g_percep is not None:
                loss += l_g_percep
                G_losses['g_percep_loss'] = l_g_percep
            if l_g_style is not None:
                loss += l_g_style
                G_losses['g_style_loss'] = l_g_style

        if self.args.idloss:
            out_gray = self.gray_resize_for_identity(fake)
            gt_gray = self.gray_resize_for_identity(gt)

            identity_gt = self.idModel(gt_gray).detach()
            identity_out = self.idModel(out_gray)
            l_identity = self.L1Loss(identity_out, identity_gt) * self.args.lambda_id
            loss += l_identity
            G_losses['g_id_loss'] = l_identity

        if self.args.crop_components:
            real_left_eyes,\
            real_right_eyes,\
            real_mouths,\
            fake_left_eyes,\
            fake_right_eyes,\
            fake_mouths = self.get_roi_regions(fake,gt,[loc_left_eye, loc_right_eye, loc_mouth])

            # left eye
            fake_left_eye, fake_left_eye_feats = self.left_eye_d(fake_left_eyes, return_feats=True)
            l_g_gan = self.PartGANLoss(fake_left_eye, True, is_disc=False)
            loss += l_g_gan
            G_losses['g_gan_left_eye'] = l_g_gan

            # right eye
            fake_right_eye, fake_right_eye_feats = self.right_eye_d(fake_right_eyes, return_feats=True)
            l_g_gan = self.PartGANLoss(fake_right_eye, True, is_disc=False)
            loss += l_g_gan
            G_losses['g_gan_right_eye'] = l_g_gan
            # mouth
            fake_mouth, fake_mouth_feats = self.mouth_d(fake_mouths, return_feats=True)
            l_g_gan = self.PartGANLoss(fake_mouth, True, is_disc=False)
            loss += l_g_gan
            G_losses['g_gan_mouth'] = l_g_gan

            if self.args.comp_style_weight  > 0:
                # get gt feat
                _, real_left_eye_feats = self.left_eye_d(real_left_eyes, return_feats=True)
                _, real_right_eye_feats = self.right_eye_d(real_right_eyes, return_feats=True)
                _, real_mouth_feats = self.mouth_d(real_mouths, return_feats=True)

                def _comp_style(feat, feat_gt, criterion):
                    return criterion(self._gram_mat(feat[0]), self._gram_mat(
                        feat_gt[0].detach())) * 0.5 + criterion(
                            self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

                # facial component style loss
                comp_style_loss = 0
                comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.L1Loss)
                comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.L1Loss)
                comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.L1Loss)
                comp_style_loss = comp_style_loss * self.args.comp_style_weight
                loss += comp_style_loss
                G_losses['g_comp_style_loss'] = comp_style_loss


        # gan loss
        fake_g_pred,fake_feat = self.netD(fake)
        l_g_gan = self.GANLoss(fake_g_pred, True, is_disc=False)
        loss += l_g_gan
        G_losses['g_gan'] = l_g_gan

        if self.args.featloss:
            _,real_feat = self.netD(gt)
            fm_loss = 0
            for r_f,f_f in zip(real_feat,fake_feat):
                fm_loss += self.L1Loss(r_f,f_f)

            fm_loss = fm_loss / len(real_feat) * self.args.lambda_fm
            loss += fm_loss 
            G_losses['g_fm_loss'] = fm_loss

        G_losses['loss'] = loss
       
        return G_losses,loss,fake


    
    
    def get_latest_losses(self):
        if not hasattr(self,'d_losses'):
            return self.g_losses
        return {**self.g_losses,**self.d_losses}

    def get_latest_generated(self):
        
        return self.generator
    
    def get_loss_from_val(self,loss):
        return loss['g_comp_style_loss'] + \
              loss['g_id_loss'] +\
              loss['g_percep_loss'] +\
              loss['g_pixloss'] + loss['g_fm_loss']
              

    def loadParameters(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)

        self.netG.load_state_dict(ckpt['G'],strict=True)
        
        self.netD.load_state_dict(ckpt['D'],strict=True)

        if self.args.crop_components:
            self.left_eye_d.load_state_dict(ckpt['left_eye_d'])
            self.right_eye_d.load_state_dict(ckpt['right_eye_d'])
            self.mouth_d.load_state_dict(ckpt['mouth_d'])
        try:
            self.optimG.load_state_dict(ckpt['g_optim'])
            self.optimD.load_state_dict(ckpt['d_optim'])
        except:
            print('you change train mode!!!')

    
    def load_scratch(self):

        state_dict = torch.load(self.args.scratch_gan_path)['params_ema']
        model_dict = self.netG.state_dict()
        state_dict = {k:v for k,v in state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.netG.load_state_dict(model_dict)

        self.netD.load_state_dict(torch.load(self.args.scratch_d_path))

        if self.args.crop_components:
            self.left_eye_d.load_state_dict(torch.load(self.args.scratch_left_eye_path)['params'])
            self.right_eye_d.load_state_dict(torch.load(self.args.scratch_right_eye_path)['params'])
            self.mouth_d.load_state_dict(torch.load(self.args.scratch_mouth_path)['params'])



    def saveParameters(self,path):

        save_dict = {
            "G": self.netG_module.state_dict(),
            'g_ema':self.g_ema.state_dict(),
            "g_optim": self.optimG.state_dict(),
            'D':self.netD_module.state_dict(),
            'd_optim': self.optimD.state_dict(),
            "args": self.args
        }

        if self.args.crop_components:
            save_dict['left_eye_d'] = self.left_eye_d_module.state_dict()
            save_dict['right_eye_d'] = self.right_eye_d_module.state_dict()
            save_dict['mouth_d'] = self.mouth_d_module.state_dict()
           
        torch.save(
                   save_dict,
                   path
                )

    def get_lr(self):
        return self.optimG.state_dict()['param_groups'][0]['lr']

    
    def select_img(self, data,size=None, name='fake', axis=2):
        if size is None:
            size = self.args.size
        data = [F.adaptive_avg_pool2d(x,size) for x in data]
        return super().select_img(data, name, axis)
    
    def freeze_models(self,frozen_params):
       
        for n in frozen_params:
            for p in self.netG.__getattr__(n).parameters():
                p.requires_grad = False 

    def get_roi_regions(self,fake,gt,location,eye_out_size=80, mouth_out_size=120):
        
        loc_left_eyes,loc_right_eyes,loc_mouths = location
        rois_eyes = []
        rois_mouths = []
        for b in range(loc_left_eyes.size(0)):  # loop for batch size
            # left eye and right eye
            img_inds = loc_left_eyes.new_full((2, 1), b)
            bbox = torch.stack([loc_left_eyes[b, :], loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
            rois_eyes.append(rois)
            # mouse
            img_inds = loc_left_eyes.new_full((1, 1), b)
            rois = torch.cat([img_inds, loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
            rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

        # real images
        all_eyes = roi_align(gt, boxes=rois_eyes, output_size=eye_out_size) 
        real_left_eyes = all_eyes[0::2, :, :, :]
        real_right_eyes = all_eyes[1::2, :, :, :]
        real_mouths = roi_align(gt, boxes=rois_mouths, output_size=mouth_out_size)
        # output
        all_eyes = roi_align(fake, boxes=rois_eyes, output_size=eye_out_size)
        fake_left_eyes = all_eyes[0::2, :, :, :]
        fake_right_eyes = all_eyes[1::2, :, :, :]
        fake_mouths = roi_align(fake, boxes=rois_mouths, output_size=mouth_out_size)
        return real_left_eyes,real_right_eyes,real_mouths,fake_left_eyes,fake_right_eyes,fake_mouths
    

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray


        


    
    

    
