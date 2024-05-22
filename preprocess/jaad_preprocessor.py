import os
import sys
import argparse
import numpy as np
import pandas as pd
import jaad_data


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/farzeen/work/aa_postdoc/intent/JAAD',help='Path to cloned JAAD repository')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of train video between [0.1]')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of val video between [0.1]')
parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test video between [0.1]')

args = parser.parse_args()

data_path = args.data_path
sys.path.insert(1, data_path+'/')


if not os.path.isdir(os.path.join(data_path, 'PN_ego')):
    os.mkdir(os.path.join(data_path, 'PN_ego'))
    
if not os.path.isdir(os.path.join(data_path, 'PN_ego', 'train')):    
    os.mkdir(os.path.join(data_path, 'PN_ego', 'train'))

if not os.path.isdir(os.path.join(data_path, 'PN_ego', 'val')):
    os.mkdir(os.path.join(data_path, 'PN_ego', 'val'))

if not os.path.isdir(os.path.join(data_path, 'PN_ego', 'test')):
    os.mkdir(os.path.join(data_path, 'PN_ego', 'test'))

jaad = jaad_data.JAAD(data_path=data_path)
dataset = jaad.generate_database()

n_train_video = int(args.train_ratio * 346)
n_val_video = int(args.val_ratio * 346)
n_test_video = int(args.test_ratio * 346)

videos = list(dataset.keys())
train_videos = videos[:n_train_video]
val_videos = videos[n_train_video:n_train_video+n_val_video]
test_videos = videos[n_train_video+n_val_video:]


for video in dataset:
    print('Processing', video, '...')
    vid = dataset[video]
    data = np.empty((0,25))
    for ped in vid['ped_annotations']:
        if vid['ped_annotations'][ped]['behavior']:
            frames = np.array(vid['ped_annotations'][ped]['frames']).reshape(-1,1)
            ids = np.repeat(vid['ped_annotations'][ped]['old_id'], frames.shape[0]).reshape(-1,1)
            bbox = np.array(vid['ped_annotations'][ped]['bbox'])
            x = bbox[:,0].reshape(-1,1)
            y = bbox[:,1].reshape(-1,1)
            w = np.abs(bbox[:,0] - bbox[:,2]).reshape(-1,1)
            h = np.abs(bbox[:,1] - bbox[:,3]).reshape(-1,1)
            imagefolderpath = np.array([os.path.join(data_path, 'images', video, '%05d'%int(frames[fr][0])+'.png') for fr in range(0, frames.shape[0])]).reshape(-1,1)
             

            cross = np.array(vid['ped_annotations'][ped]['behavior']['cross']).reshape(-1,1)

            reaction = np.array(vid['ped_annotations'][ped]['behavior']['reaction']).reshape(-1,1)
            hand_gesture = np.array(vid['ped_annotations'][ped]['behavior']['hand_gesture']).reshape(-1,1)
            look = np.array(vid['ped_annotations'][ped]['behavior']['look']).reshape(-1,1)
            nod = np.array(vid['ped_annotations'][ped]['behavior']['nod']).reshape(-1,1)


            ####[age,gender,group_size]

            age=np.repeat(vid['ped_annotations'][ped]['attributes']['age'],frames.shape[0]).reshape(-1,1)
            gender=np.repeat(vid['ped_annotations'][ped]['attributes']['gender'],frames.shape[0]).reshape(-1,1)
            group_size=np.repeat(vid['ped_annotations'][ped]['attributes']['group_size'],frames.shape[0]).reshape(-1,1)
           

            #### scene_attribute [road_type,designated,motion_direction,num_lanes,signalized,traffic_direction]
            
           
            designated=np.repeat(vid['ped_annotations'][ped]['attributes']['designated'],frames.shape[0]).reshape(-1,1)
            motion_direction=np.repeat(vid['ped_annotations'][ped]['attributes']['motion_direction'],frames.shape[0]).reshape(-1,1)
            num_lanes=np.repeat(vid['ped_annotations'][ped]['attributes']['num_lanes'],frames.shape[0]).reshape(-1,1)
            signalized=np.repeat(vid['ped_annotations'][ped]['attributes']['signalized'],frames.shape[0]).reshape(-1,1)
            traffic_direction=np.repeat(vid['ped_annotations'][ped]['attributes']['traffic_direction'],frames.shape[0]).reshape(-1,1)
            road_type=np.repeat(vid['traffic_annotations']['road_type'],frames.shape[0]).reshape(-1,1)
            
            
            
            ped_crossing,ped_sign,stop_sign,traffic_light=[],[],[],[] ##[ped_crossing,ped_sign,stop_sign,traffic_light]
            for f in frames:
                ped_crossing.append(vid['traffic_annotations'][f[0]]['ped_crossing'])
                ped_sign.append(vid['traffic_annotations'][f[0]]['ped_sign'])
                stop_sign.append(vid['traffic_annotations'][f[0]]['stop_sign'])
                traffic_light.append(vid['traffic_annotations'][f[0]]['traffic_light'])
        

            ### add other behavior attributes here, reaction, hand_gesture,Look, action, nod
            ### add attributes, age, crossing, crossing_point, desicion point, designated, gender, group_soze, motion direction, num_lanes, signalized, traffic direction
            ped_crossing=np.array(ped_crossing).reshape(-1,1)
            ped_sign=np.array(ped_sign).reshape(-1,1)
            stop_sign=np.array(stop_sign).reshape(-1,1)
            traffic_light=np.array(traffic_light).reshape(-1,1)


            ped_data = np.hstack((frames, ids, x, y, w, h, imagefolderpath, cross,reaction,hand_gesture,look,nod,age,gender,group_size,designated,motion_direction,num_lanes,signalized,traffic_direction,ped_crossing,ped_sign,stop_sign,traffic_light,road_type))
            data = np.vstack((data, ped_data))
    data_to_write = pd.DataFrame({'frame': data[:,0].reshape(-1), 
                                  'ID': data[:,1].reshape(-1), 
                                  'x': data[:,2].reshape(-1), 
                                  'y': data[:,3].reshape(-1), 
                                  'w': data[:,4].reshape(-1), 
                                  'h': data[:,5].reshape(-1), 
                                  'imagefolderpath': data[:,6].reshape(-1), 
                                  'crossing_true': data[:,7].reshape(-1), 
                                  'reaction': data[:,8].reshape(-1), 
                                  'hand_gesture': data[:,9].reshape(-1), 
                                  'look': data[:,10].reshape(-1), 
                                  'nod': data[:,11].reshape(-1), 
                                  'age': data[:,12].reshape(-1), 
                                  'gender': data[:,13].reshape(-1), 
                                  'group_size': data[:,14].reshape(-1), 
                                  'designated': data[:,15].reshape(-1), 
                                  'motion_direction': data[:,16].reshape(-1), 
                                  'num_lanes': data[:,17].reshape(-1), 
                                  'signalized': data[:,18].reshape(-1), 
                                  'traffic_direction': data[:,19].reshape(-1), 
                                  'ped_crossing': data[:,20].reshape(-1), 
                                  'ped_sign': data[:,21].reshape(-1), 
                                  'stop_sign': data[:,22].reshape(-1), 
                                  'traffic_light': data[:,23].reshape(-1), 
                                  'road_type': data[:,24].reshape(-1),})
    data_to_write['filename'] = data_to_write.frame
    data_to_write.filename = data_to_write.filename.apply(lambda x: '%05d'%int(x)+'.png')
    
    if video in train_videos:
        data_to_write.to_csv(os.path.join(data_path, 'PN_ego', 'train', video+'.csv'), index=False)
    elif video in val_videos:
        data_to_write.to_csv(os.path.join(data_path, 'PN_ego', 'val', video+'.csv'), index=False)
    elif video in test_videos:
        data_to_write.to_csv(os.path.join(data_path, 'PN_ego', 'test', video+'.csv'), index=False)