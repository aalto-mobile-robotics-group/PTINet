import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet18_Weights
from model.clstm import*
from model.vae import*





class PTINet(nn.Module):
    def __init__(self, args):
        super(PTINet, self).__init__()

        if args.dataset=='jaad':
            self.size = 4
            self.ped_attribute_size=3
            self.ped_behavior_size=4
            self.scene_attribute_size=10

        elif args.dataset=='pie':
            self.size = 4
            self.ped_attribute_size=2
            self.ped_behavior_size=3
            self.scene_attribute_size=4

        elif args.dataset=='titan':
            self.size = 4
            self.ped_behavior_size=3
        else:
            'wrong dataset'

       
        self.num_layers=1
        self.latent_size=args.hidden_size
        
        self.speed_encoder = LSTMVAE(input_size=self.size, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)
        self.pos_encoder = LSTMVAE(input_size=self.size, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)
        


        if args.use_attribute == True:
            self.ped_behavior_encoder = LSTMVAE(input_size=self.ped_behavior_size, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)
            if args.dataset == 'jaad' or args.dataset == 'pie':         
                self.scene_attribute_encoder   =LSTMVAE(input_size=self.scene_attribute_size, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)
                self.mlp = nn.Sequential( nn.Linear(self.ped_attribute_size, 64),nn.ReLU(),nn.Linear(64, args.hidden_size),nn.ReLU() )

        if args.use_image==True:
            if args.image_network== 'resnet50':
                self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                self.resnet.fc = nn.Identity()
                self.img_encoder   = LSTMVAE(input_size=2048, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)

            elif args.image_network== 'resent18':
                self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                self.resnet.fc = nn.Identity()
                self.img_encoder   = nn.LSTM(input_size=512, hidden_size=args.hidden_size,num_layers=self.num_layers,batch_first=True)
            elif args.image_network == 'clstm':
                self.clstm=ConvLSTM(input_channels=3, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, conv_stride=1,pool_kernel_size=(2, 2), step=5, effective_step=[4])
                self.pooling_h = nn.AdaptiveAvgPool2d((1, 1))
                self.pooling_c = nn.AdaptiveAvgPool2d((1, 1))
                self.linear_c= nn.Linear(in_features=32, out_features=512)
                self.linear_h = nn.Linear(in_features=32, out_features=512)

        if args.use_opticalflow== True:
            self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Identity()
            self.op_encoder   = nn.LSTM(input_size=2048, hidden_size=args.hidden_size,num_layers=self.num_layers,batch_first=True)

      
        
        self.pos_embedding = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=self.size),
                                           nn.ReLU())
        
        self.speed_decoder    = nn.LSTMCell(input_size=self.size, hidden_size=args.hidden_size)
        self.crossing_decoder = nn.LSTMCell(input_size=self.size, hidden_size=args.hidden_size)
        self.attrib_decoder = nn.LSTMCell(input_size=self.size, hidden_size=args.hidden_size)
        
        self.fc_speed    = nn.Linear(in_features=args.hidden_size, out_features=self.size)
        self.fc_crossing = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=2), nn.ReLU())
        self.fc_attrib = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=3), nn.ReLU())
        
        self.hardtanh = nn.Hardtanh(min_val=-1*args.hardtanh_limit, max_val=args.hardtanh_limit)
        self.softmax = nn.Softmax(dim=1)
        
        self.args = args
        
    def forward(self, speed=None, pos=None,ped_attribute=None,ped_behavior=None,scene_attribute=None,images=None,optical=None, average=False):

        sloss, x_hat, zsp,hsp, (recon_loss, kld_loss) = self.speed_encoder(speed)
        hsp = hsp[0].squeeze(0)
        zsp=torch.mean(zsp,axis=1)
        # csp = csp.squeeze(0)
        
        ploss, x_hat, zpo,hpo, (recon_loss, kld_loss) = self.pos_encoder(pos)
        hpo = hpo[0].squeeze(0)
        zpo=torch.mean(zpo,axis=1)
        # cpo = cpo.squeeze(0)



        if self.args.use_attribute == True:
            pbloss, x_hat, zpa,hpa, (recon_loss, kld_loss)  = self.ped_behavior_encoder (ped_behavior)
            hpa = hpa[0].squeeze(0)
            zpa = torch.mean(zpa,axis=1)

            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':  

                psloss, x_hat, zsa,hsa, (recon_loss, kld_loss)  = self.scene_attribute_encoder(scene_attribute)
                hsa = hsa[0].squeeze(0)
                zsa =torch.mean(zsa,axis=1)

                pb=self.mlp(ped_attribute)

        if self.args.use_image==True:
            batch_size, seq_len, c, h, w = images.size()

            if self.args.image_network=='clstm':
                batch_size, seq_len, c, h, w = images.size()
                _,(himg, cimg)=self.clstm(images)
                himg=self.pooling_h(himg).view(himg.size(0), -1)
                himg=self.linear_h(himg)

                cimg=self.pooling_c(cimg).view(cimg.size(0), -1)
                cimg=self.linear_c(cimg)
            else:
                images = images.view(batch_size * seq_len, c, h, w)
                img_feats = self.resnet(images)
                img_feats = img_feats.view(batch_size, seq_len, -1)

                imgloss, x_hat, zim,him, (recon_loss, kld_loss) = self.img_encoder(img_feats)
                him = him[0].squeeze(0)
                zim = torch.mean(zim,axis=1)
        if self.args.use_opticalflow==True:
            batch_size_op, seq_len_op, c_op, h_op, w_op = optical.size()
            optical = optical.view(batch_size * seq_len_op, c_op, h_op, w_op)
            op_feats = self.resnet(optical)
            op_feats = op_feats.view(batch_size, seq_len_op, -1)

            _, (himg_op, cimg_op) = self.op_encoder(op_feats)
            himg_op = himg_op[-1,:,:].squeeze(0)
            cimg_op = cimg_op[-1,:,:].squeeze(0)


        outputs=[]

        if self.args.dataset == 'jaad' or self.args.dataset == 'pie':   

            outputs.append(ploss+sloss+pbloss+psloss)
        else:
            outputs.append(ploss+sloss+pbloss)



        #  _, (hsp, csp) = self.speed_encoder(speed.permute(1,0,2))
        # hsp = hsp.squeeze(0)
        # csp = csp.squeeze(0)
        
        # _, (hpo, cpo) = self.pos_encoder(pos.permute(1,0,2))
        # hpo = hpo.squeeze(0)
        # cpo = cpo.squeeze(0)
        # outputs = []

        #  if self.args.use_attribute == True:
        #     _, (hpa, cpa) = self.ped_behavior_encoder (ped_behavior)
        #     hpa = hpa[-1,:,:].squeeze(0)
        #     cpa = cpa[-1,:,:].squeeze(0)

        #     _, (hsa, csa) = self.scene_attribute_encoder(scene_attribute)
        #     hsa = hsa[-1,:,:].squeeze(0)
        #     csa = csa[-1,:,:].squeeze(0)

        #     pb=self.mlp(ped_attribute)

        # if self.args.use_image==True:
        #     batch_size, seq_len, c, h, w = images.size()
        #     images = images.view(batch_size * seq_len, c, h, w)
        #     img_feats = self.resnet(images)
        #     img_feats = img_feats.view(batch_size, seq_len, -1)

        #     _, (himg, cimg) = self.img_encoder(img_feats)
        #     himg = himg[-1,:,:].squeeze(0)
        #     cimg = cimg[-1,:,:].squeeze(0)

        # outputs = []
        
        
        speed_outputs    = torch.tensor([], device=self.args.device)
        in_sp = speed[:,-1,:]
        
        hds = hpo + hsp
        zds = zpo + zsp

        if self.args.use_attribute == True:
            hds = hds + hpa  
            zds = zds +zpa 
            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':  
                hds = hds+ hsa  + hpa  + pb 
                zds = zds +zpa + zsa + pb

        if self.args.use_image ==True:
            hds=hds + himg
            zds=zds + cimg 

        if self.args.use_opticalflow ==True:
            hds=hds + himg_op
            zds=zds + cimg_op 

        for i in range(self.args.output//self.args.skip):
            hds, zds         = self.speed_decoder(in_sp, (hds, zds))
            speed_output     = self.hardtanh(self.fc_speed(hds))
            speed_outputs    = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim = 1)
            in_sp            = speed_output.detach()
            
        outputs.append(speed_outputs)

        
        crossing_outputs = torch.tensor([], device=self.args.device)
        in_cr = pos[:,-1,:]
        
        hdc = hpo + hsp
        zdc = zpo + zsp

        if self.args.use_attribute == True:
            hdc = hdc  + hpa  
            zdc = zdc+ zpa 
            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':   
                hdc = hdc+ hsa  + hpa  + pb 
                zdc = zdc +zpa + zsa + pb


        if self.args.use_image ==True:
            hdc=hdc + himg
            zdc=zdc + cimg 

        if self.args.use_opticalflow ==True:
            hdc=hdc + himg_op
            zdc=zdc + cimg_op 

        for i in range(self.args.output//self.args.skip):
            hdc, zdc         = self.crossing_decoder(in_cr, (hdc, zdc))
            crossing_output  = self.fc_crossing(hdc)
            in_cr            = self.pos_embedding(hdc).detach()
            crossing_output  = self.softmax(crossing_output)
            crossing_outputs = torch.cat((crossing_outputs, crossing_output.unsqueeze(1)), dim = 1)

        outputs.append(crossing_outputs)
    
        if average:
            crossing_labels = torch.argmax(crossing_outputs, dim=2)
            intention = torch.max(crossing_labels,dim=1)[0]
            outputs.append(intention)
        


        return tuple(outputs)
