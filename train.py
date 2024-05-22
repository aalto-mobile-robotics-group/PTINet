import time
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score,f1_score
import datetime
import time
import datasets
# import network_trans as network
import model.network_image as network
# import network_attri
import utils
from utils import data_loader,calculate_score
from torch.utils.tensorboard import SummaryWriter
import visualization.display as viz

def parse_config_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PTINet network')
    
    parser.add_argument('--data_dir', type=str,
                        default='/home/farzeen/work/aa_postdoc/intent/JAAD/PN/',#'/scratch/project_2007864/JAAD/processed_annotations/',
                        required=False)
    parser.add_argument('--dataset', type=str, 
                        default='pie',
                        required=False)
    parser.add_argument('--out_dir', type=str, 
                        default='/home/farzeen/work/aa_postdoc/intent/PIE_bbox_image/bounding-box-prediction/output',#'/projappl/project_2007864/bbox_pred_image/output/',
                        required=False)  
    
    # data configuration
    parser.add_argument('--input', type=int,
                        default=16,
                        required=False)
    parser.add_argument('--output', type=int, 
                        default=32,
                        required=False)
    parser.add_argument('--stride', type=int, 
                        default=16,
                        required=False)  
    parser.add_argument('--skip', type=int, default=1)  
  

    # data loading / saving           
    parser.add_argument('--dtype', type=str, default='train')
    parser.add_argument("--from_file", type=bool, default=False)       
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--loader_workers', type=int, default=16)
    parser.add_argument('--loader_shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--prefetch_factor', type=int, default=3)

    # training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--lr_scheduler', type=bool, default=False)
    parser.add_argument('--local-rank', type=int, default=0)

    # network
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--use_image', type=bool, default=False,
                        help='Use input image as a feature',
                        required=False)
    parser.add_argument('--image_network', type=str, default='resnet50',
                        help='select backbone',
                        required=False)
    parser.add_argument('--use_attribute', type=bool, default=True,
                        help='Use input attribute as a feature',
                        required=False)
    parser.add_argument('--use_opticalflow', type=bool, default=True,
                        help='Use input emdedding as a feature',
                        required=False)
    
    args = parser.parse_args()

    return args



def train(args, train, val):
    print('='*100)
    print('Training ...')
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')


    dist.init_process_group(backend='nccl')
    torch.manual_seed(0)
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    verbose = dist.get_rank() == 0  # print only on global_rank==0
    net = network.PTINet(args).cuda()
    net = DistributedDataParallel(net, device_ids=[local_rank],find_unused_parameters=True)


    # Enable Tensor Core operations
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)


    optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=1e-7)
 

    if args.lr_scheduler:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, 
        #                                                 threshold = 1e-8, verbose=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    train_sampler = DistributedSampler(train)     
    dataloader_train = torch.utils.data.DataLoader(train, batch_size=args.batch_size,pin_memory=args.pin_memory, num_workers=0, drop_last=True,sampler=train_sampler)#,prefetch_factor=args.prefetch_factor)
    # init values
    mse = nn.MSELoss()
    # huber = torch.nn.HuberLoss(reduction='sum', delta=1.0)
    bce = nn.BCELoss()
    data = []
    best_ade=float('inf')
    writer = SummaryWriter()

    for epoch in range(args.n_epochs):
        start = time.time()
        
        avg_epoch_train_s_loss = 0
        avg_epoch_val_s_loss   = 0
        avg_epoch_train_c_loss = 0
        avg_epoch_val_c_loss   = 0
        avg_epoch_train_t_loss = 0
        avg_epoch_val_v_loss   = 0
        
        ade  = 0
        fde  = 0
        aiou = 0
        fiou = 0
        avg_acc = 0
        avg_rec = 0
        avg_pre = 0
        mAP = 0
        
        counter = 0
        for idx, inputs in enumerate(dataloader_train):
            counter += 1
            speed    = inputs['speed'].cuda(non_blocking=True)#obs_s
            future_speed = inputs['future_speed'].cuda(non_blocking=True) #target_s
            pos    = inputs['pos'].cuda(non_blocking=True) # obs_p
            future_pos = inputs['future_pos'].cuda(non_blocking=True) #target_p
            future_cross = inputs['future_cross'].cuda(non_blocking=True) #target_c
            optical=inputs['optical'].cuda(non_blocking=True)  
            ped_behavior = inputs['ped_behavior'].cuda(non_blocking=True) 
            images = inputs['image'].cuda(non_blocking=True) 
            label_c=inputs['cross_label'].cuda(non_blocking=True)
            ped_attribute = inputs['ped_attribute'].cuda(non_blocking=True) 
            scene_attribute = inputs['scene_attribute'].cuda(non_blocking=True)
            
            net.zero_grad()
            mloss,speed_preds, crossing_preds = net(speed=speed, pos=pos,ped_attribute=ped_attribute,ped_behavior=ped_behavior,scene_attribute=scene_attribute,images=images,optical=optical,average=False)
            # speed_preds, crossing_preds = net(image_list=images, velocity=speed, position=pos,average=False)
            # intentions=0
            # visual=viz.visualizer(inputs,speed_preds,crossing_preds,intentions,path='/home/farzeen/work/aa_postdoc/intent/Titan/titan_lstm_vae_clstm/bounding-box-prediction/visual')
            # visual.show_frame()
            # visual.show_sequence(save=True)

            # speed_loss  = huber(speed_preds, future_speed)
            speed_loss  = mse(speed_preds, future_speed)/100

            crossing_loss = 0
            for i in range(future_cross.shape[1]):
                crossing_loss += bce(crossing_preds[:,i], future_cross[:,i])
                
            crossing_loss /= future_cross.shape[1]
            
            loss = speed_loss + crossing_loss + mloss
            loss.backward()
            optimizer.step()
            
            avg_epoch_train_s_loss += float(speed_loss)
            avg_epoch_train_c_loss += float(crossing_loss)
            avg_epoch_train_t_loss += float(loss)
            torch.cuda.synchronize()
        if args.save:
            print('\nSaving ...')
            file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
            if args.lr_scheduler:
                filename = 'data_' + file + '_scheduler.csv'
                modelname = 'model_best' + file + '_scheduler.pkl'
            else:
                filename = 'data_' + file + str(epoch)+'.csv'
                modelname = 'model_best' + file +str(epoch)+ '.pkl'

            # df.to_csv(os.path.join(args.out_dir, args.log_name, filename), index=False)
            torch.save(net.state_dict(), os.path.join(args.out_dir, args.log_name, modelname))
            
            print('Training data and model saved to {}\n'.format(os.path.join(args.out_dir, args.log_name)))

            
        avg_epoch_train_s_loss /= counter
        avg_epoch_train_c_loss /= counter
        avg_epoch_train_t_loss /= counter
        writer.add_scalar("Loss_speed/train", avg_epoch_train_s_loss, epoch)
        writer.add_scalar("Loss_crossing/train", avg_epoch_train_c_loss, epoch)
        writer.add_scalar("Loss/train", avg_epoch_train_t_loss, epoch)
        



        val_sampler = DistributedSampler(val)     
        dataloader_val = torch.utils.data.DataLoader(val, batch_size=args.batch_size,pin_memory=args.pin_memory, num_workers=0, drop_last=True,sampler=val_sampler)#,prefetch_factor=args.prefetch_factor)

        counter=0
        state_preds = []
        state_targets = []
        intent_preds = []
        intent_targets = []
        f1_sc=[]
        pre=[]
        recall_sc=[]
        acc=[]


        for idx, val_in in enumerate(dataloader_val):
            counter+=1

            speed    = val_in['speed'].cuda(non_blocking=True) #obs_s
            future_speed = val_in['future_speed'].cuda(non_blocking=True) #target_s
            pos    = val_in['pos'].cuda(non_blocking=True) # obs_p
            future_pos = val_in['future_pos'].cuda(non_blocking=True) #target_p
            future_cross = val_in['future_cross'].cuda(non_blocking=True) #target_c
            ped_attribute = val_in['ped_attribute'].cuda(non_blocking=True)
            scene_attribute = val_in['scene_attribute'].cuda(non_blocking=True)
            optical=val_in['optical'].cuda(non_blocking=True) 
            ped_behavior = val_in['ped_behavior'].cuda(non_blocking=True)
            images = val_in['image'].cuda(non_blocking=True)
            label_c= val_in['cross_label'].cuda(non_blocking=True)
            
            with torch.no_grad():
                vloss,speed_preds, crossing_preds, intentions = net(speed=speed, pos=pos,ped_attribute=ped_attribute,ped_behavior=ped_behavior,scene_attribute=scene_attribute,images=images,optical=optical,average=True)
                # speed_loss    = huber(speed_preds, future_speed)
                # speed_preds, crossing_preds, intentions = net(image_list=images, velocity=speed, position=pos,average=True)
                speed_loss_v    = mse(speed_preds, future_speed)/100
                
                crossing_loss_v = 0
                for i in range(future_cross.shape[1]):
                    crossing_loss_v += bce(crossing_preds[:,i], future_cross[:,i])
                crossing_loss_v /= future_cross.shape[1]
                
                avg_epoch_val_s_loss += float(speed_loss_v)
                avg_epoch_val_c_loss += float(crossing_loss_v)
                avg_epoch_val_v_loss += float(vloss)
                
                preds_p = utils.speed2pos(speed_preds, pos)
                ade += float(utils.ADE(preds_p, future_pos))
                fde += float(utils.FDE(preds_p, future_pos))
                aiou += float(utils.AIOU(preds_p, future_pos))
                fiou += float(utils.FIOU(preds_p, future_pos))
                
                future_cross = future_cross[:,:,1].view(-1).cpu().numpy()
                crossing_preds = np.argmax(crossing_preds.view(-1,2).detach().cpu().numpy(), axis=1)
                precision,recall,f1,accuracy=calculate_score(crossing_preds,future_cross)
                pre.append(precision)
                recall_sc.append(recall)
                f1_sc.append(f1)
                acc.append(accuracy)
                
                label_c = label_c.view(-1).cpu().numpy()
                intentions = intentions.view(-1).detach().cpu().numpy()

             
                

                state_preds.extend(crossing_preds)
                state_targets.extend(future_cross)
                intent_preds.extend(intentions)
                intent_targets.extend(label_c)
                torch.cuda.synchronize()

            
        avg_epoch_val_s_loss /= counter
        avg_epoch_val_c_loss /= counter
        
        ade  /= counter
        fde  /= counter     
        aiou /= counter
        fiou /= counter

        v_loss= avg_epoch_val_s_loss + avg_epoch_val_c_loss + avg_epoch_val_v_loss

        writer.add_scalar("Loss_speed/val", avg_epoch_val_s_loss, epoch)
        writer.add_scalar("Loss_crossing/val", avg_epoch_val_c_loss, epoch)
        # writer.add_scalar("Loss/val", v_loss, epoch)


        pre_int,recall_int,f1_intt,acc_int=calculate_score(np.array(intent_preds),np.array(intent_targets))
        pre=np.sum(pre)/counter
        recall_sc=np.sum(recall_sc)/counter
        f1_sc=np.sum(f1_sc)/counter
        acc=np.sum(acc)/counter

        avg_acc = accuracy_score(state_targets, state_preds)
        f1_state = f1_score(state_targets, state_preds)
        avg_rec = recall_score(state_targets, state_preds, average='binary', zero_division=1)
        avg_pre = precision_score(state_targets, state_preds, average='binary', zero_division=1)
        mAP = average_precision_score(state_targets, state_preds, average=None)
        intent_acc = accuracy_score(intent_targets, intent_preds)
        f1_int = f1_score(intent_targets, intent_preds)
        intent_mAP = average_precision_score(intent_targets, intent_preds, average=None)
        
        data.append([epoch, avg_epoch_train_s_loss, avg_epoch_val_s_loss, \
                    avg_epoch_train_c_loss, avg_epoch_val_c_loss, \
                    ade, fde, aiou, fiou, intent_acc])

        if args.lr_scheduler:
            scheduler.step(avg_epoch_train_t_loss)

        if ade < best_ade:
            best_ade = ade
            file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
            if args.lr_scheduler:
                 modelname = 'model_best' + file + '_scheduler.pkl'
            else:
                 modelname = 'model_best' + file + '.pkl'   
            #  print(modelname)
            torch.save(net.state_dict(), os.path.join(args.out_dir, args.log_name, modelname))
        
        print('e:', epoch, 
             '| ade: %.4f'% ade, 
            '| fde: %.4f'% fde, '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| state_acc: %.4f'% avg_acc,acc, 
            '| intention_acc: %.4f'% intent_acc,acc_int, '| f1_int: %.4f'%f1_int,f1_intt, '| f1_state: %.4f'% f1_state,f1_sc, '| pre: %.4f'% pre, '| recall_sc: %.4f'% recall_sc,'| pre_int: %.4f'% pre_int, '| recall_int: %.4f'% recall_int,
            )
   

    df = pd.DataFrame(data, columns =['epoch', 'train_loss_s', 'val_loss_s', 'train_loss_c', 'val_loss_c',\
                'ade', 'fde', 'aiou', 'fiou', 'intention_acc']) 

    if args.save:
        print('\nSaving ...')
        file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
        if args.lr_scheduler:
            filename = 'data_final' + file + '_scheduler.csv'
            modelname = 'model_final' + file + '_scheduler.pkl'
        else:
            filename = 'data_final' + file + '.csv'
            modelname = 'model_final' + file + '.pkl'

        df.to_csv(os.path.join(args.out_dir, args.log_name, filename), index=False)
        torch.save(net.state_dict(), os.path.join(args.out_dir, args.log_name, modelname))
        
        print('Training data and model saved to {}\n'.format(os.path.join(args.out_dir, args.log_name)))

    print('='*100)
    print('Done !')




if __name__ == '__main__':

    print("Date and time:",datetime.datetime.now())

    args = parse_args()
    config = parse_config_file('/home/farzeen/work/aa_postdoc/intent/PTINet/PTINet/config.yml')
    if config.get('use_argument_parser')== False:
    # Override command-line arguments with values from the configuration file
        for arg in vars(args):
            if arg in config:
                setattr(args, arg, config[arg])
    print(args)
    # create output dir
    # if not args.log_name:
    #     args.log_name = '{}_{}_{}_{}'.format(args.dataset, str(args.input),\
    #                             str(args.output), str(args.stride)) 
    # if not os.path.isdir(os.path.join(args.out_dir, args.log_name)):
    #     os.mkdir(os.path.join(args.out_dir, args.log_name))

    # select dataset
   

    # load data
    train_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=os.path.join(args.out_dir, args.log_name),
                dtype='val',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                from_file=args.from_file,
                save=args.save,
                use_images=args.use_image,
                use_attribute=args.use_attribute,
                use_opticalflow=args.use_opticalflow
                )

    # train_loader = data_loader(args, train_set)

    val_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=os.path.join(args.out_dir, args.log_name),
                dtype='val',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                from_file=args.from_file,
                save=args.save,
                use_images=args.use_image,
                use_attribute=args.use_attribute,
                use_opticalflow=args.use_opticalflow
                )
    # val_loader = data_loader(args, val_set)

   
    
    train(args,  train_set, val_set)
    