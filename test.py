import time
import os
import argparse

import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import torchvision
# import torchvision.transforms as transforms
    
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score,f1_score

# import DataLoader
import datasets
import model.network_image as network
import utils
from utils import data_loader,calculate_score
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PV-LSTM network')
    
    parser.add_argument('--data_dir', type=str,
                        default= '/scratch/project_2007864/PIE/PN/',
                        required=False)
    parser.add_argument('--dataset', type=str, 
                        default='pie',
                        required=False)
    parser.add_argument('--out_dir', type=str, 
                        default= '/projappl/project_2007864/PIE_lstm_vae_clstm/bounding-box-prediction/output_32/',
                        required=False)  
    parser.add_argument('--task', type=str, 
                        default='2D_bounding_box-intention',
                        required=False)
    
    # data configuration
    parser.add_argument('--input', type=int,
                        default=16,
                        required=False)
    parser.add_argument('--output', type=int, 
                        default=48,
                        required=False)
    parser.add_argument('--stride', type=int, 
                        default=16,
                        required=False)  
    parser.add_argument('--skip', type=int, default=1)  
    parser.add_argument('--is_3D', type=bool, default=False) 
    # data loading / saving           
    parser.add_argument('--dtype', type=str, default='val')
    parser.add_argument("--from_file", type=bool, default=False)       
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--loader_workers', type=int, default=10)
    parser.add_argument('--loader_shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--prefetch_factor', type=int, default=3)

    # training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=int, default= 0.001)
    parser.add_argument('--lr_scheduler', type=bool, default=False)

    # network
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--use_image', type=bool, default=True,
                        help='Use input image as a feature',
                        required=False)
    parser.add_argument('--image_network', type=str, default='clstm',
                        help='select backbone',
                        required=False)
    parser.add_argument('--use_attribute', type=bool, default=True,
                        help='Use input attribute as a feature',
                        required=False)
    parser.add_argument('--use_embedding', type=bool, default=False,
                        help='Use input emdedding as a feature',
                        required=False)
    parser.add_argument('--embedding_dim', type=int, default=10,
                        help='Use input attribute as a feature',
                        required=False)
   

    args = parser.parse_args()

    return args


# For 2D datasets
def test_2d(args, test):
    print('='*100)
    print('Testing ...')
    print('Task: ' + str(args.task))
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
    net = network.PV_LSTM(args).cuda()
    net = DistributedDataParallel(net, device_ids=[local_rank],find_unused_parameters=True)


    file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
    if args.lr_scheduler:
       
            modelname = 'model_' + file + '_scheduler.pkl'
    else:
         
            modelname = 'model_best' + file + '.pkl'
    # modelname = 'model_' + file + '.pkl'
    print(os.path.join(args.out_dir, args.log_name, modelname))
    net.load_state_dict(torch.load(os.path.join(args.out_dir, args.log_name, modelname)))
    net.eval()
    test_sampler = DistributedSampler(test) 
    dataloader_test = torch.utils.data.DataLoader(test, batch_size=args.batch_size,pin_memory=args.pin_memory,sampler=test_sampler, num_workers=20, drop_last=True,prefetch_factor=args.prefetch_factor)
    # mse = nn.MSELoss()
    huber = torch.nn.HuberLoss(reduction='sum', delta=1.0)
    bce = nn.BCELoss()
    val_s_scores   = []
    val_c_scores   = []

    ade  = 0
    fde  = 0
    aiou = 0
    fiou = 0
    avg_acc = 0
    avg_rec = 0
    avg_pre = 0
    mAP = 0

    avg_epoch_val_s_loss   = 0
    avg_epoch_val_c_loss   = 0

    counter=0
    state_preds = []
    state_targets = []
    intent_preds = []
    intent_targets = []
    f1_sc=[]
    pre=[]
    recall_sc=[]
    acc=[]

    start = time.time()

    for idx, inputs in enumerate(dataloader_test):
        counter+=1
        speed    = inputs['speed'].cuda(non_blocking=True)#obs_s
        future_speed = inputs['future_speed'].cuda(non_blocking=True) #target_s
        pos    = inputs['pos'].cuda(non_blocking=True) # obs_p
        future_pos = inputs['future_pos'].cuda(non_blocking=True) #target_p
        future_cross = inputs['future_cross'].cuda(non_blocking=True) #target_c
        ped_attribute = inputs['ped_attribute'].cuda(non_blocking=True) 
        scene_attribute = inputs['scene_attribute'].cuda(non_blocking=True) 
        ped_behavior = inputs['ped_behavior'].cuda(non_blocking=True) 
        images = inputs['image'].cuda(non_blocking=True) 
        label_c=inputs['cross_label'].cuda(non_blocking=True)

        with torch.no_grad():
            mloss,speed_preds, crossing_preds, intentions = net(speed=speed, pos=pos,ped_attribute=ped_attribute,ped_behavior=ped_behavior,scene_attribute=scene_attribute,images=images,average=True)
            speed_loss    = huber(speed_preds, future_speed)
            # speed_loss    = mse(speed_preds, future_speed)/100

            crossing_loss = 0
            for i in range(future_cross.shape[1]):
                crossing_loss += bce(crossing_preds[:,i], future_cross[:,i])
            crossing_loss /= future_cross.shape[1]

            avg_epoch_val_s_loss += float(speed_loss)
            avg_epoch_val_c_loss += float(crossing_loss)

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

        avg_epoch_val_s_loss += float(speed_loss)
        avg_epoch_val_c_loss += float(crossing_loss)

    avg_epoch_val_s_loss /= counter
    avg_epoch_val_c_loss /= counter
    val_s_scores.append(avg_epoch_val_s_loss)
    val_c_scores.append(avg_epoch_val_c_loss)

    ade  /= counter
    fde  /= counter     
    aiou /= counter
    fiou /= counter

    pre_int,recall_int,f1_intt,acc_int=calculate_score(np.array(intent_preds),np.array(intent_targets))
    pre=np.sum(pre)/counter
    recall_sc=np.sum(recall_sc)/counter
    f1_sc=np.sum(f1_sc)/counter
    acc=np.sum(acc)/counter

    avg_acc = accuracy_score(state_targets, state_preds)
    f1_state = f1_score(state_targets, state_preds)
    avg_rec = recall_score(state_targets, state_preds, average='binary', zero_division=1)
    avg_pre = precision_score(state_targets, state_preds, average='binary', zero_division=1)
    intent_acc = accuracy_score(intent_targets, intent_preds)
    f1_int = f1_score(intent_targets, intent_preds)

    print( '| ade: %.4f'% ade, 
        '| fde: %.4f'% fde, '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| state_acc: %.4f'% avg_acc,acc, 
        '| int_acc: %.4f'% intent_acc,acc_int, '| f1_int: %.4f'%f1_int,f1_intt, '| f1_state: %.4f'% f1_state,f1_sc, '| pre: %.4f'% pre, '| recall_sc: %.4f'% recall_sc,'| pre_int: %.4f'% pre_int, '| recall_int: %.4f'% recall_int,
        '| t:%.4f'%(time.time()-start))




if __name__ == '__main__':
    args = parse_args()

    # create output dir
    # if not args.log_name:
    #     args.log_name = '{}_{}_{}_{}'.format(args.dataset, str(args.input),\
    #                             str(args.output), str(args.stride)) 
    # if not os.path.isdir(os.path.join(args.out_dir, args.log_name)):
    #     os.mkdir(os.path.join(args.out_dir, args.log_name))

    # select dataset
    if args.dataset == 'jaad':
        args.is_3D = False
    elif args.dataset == 'jta':
        args.is_3D = True
    elif args.dataset == 'nuscenes':
        args.is_3D = True
    else:
        print('Unknown dataset entered! Please select from available datasets: jaad, jta, nuscenes...')

    # load data
    test_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=os.path.join(args.out_dir, args.log_name),
                dtype='val',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                task=args.task,
                from_file=args.from_file,
                save=args.save,
                use_images=args.use_image,
                use_attribute=args.use_attribute
                )

    # test_loader = data_loader(args, test_set)

    # initiate network
    # net = network.PV_LSTM(args).to(args.device)

    # training
    test_2d(args,  test_set)