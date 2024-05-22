import os
import sys
import argparse
import numpy as np
import pandas as pd
import pie_data


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to cloned PIE repository',default='/home/farzeen/work/aa_postdoc/datasets/Dataset/PIE/')


args = parser.parse_args()

data_path = args.data_path
sys.path.insert(1, data_path+'/')


if not os.path.isdir(os.path.join(data_path, 'PN_imu')):
    os.mkdir(os.path.join(data_path, 'PN_imu'))
    
if not os.path.isdir(os.path.join(data_path, 'PN_imu', 'train')):    
    os.mkdir(os.path.join(data_path, 'PN_imu', 'train'))

if not os.path.isdir(os.path.join(data_path, 'PN_imu', 'val')):
    os.mkdir(os.path.join(data_path, 'PN_imu', 'val'))

if not os.path.isdir(os.path.join(data_path, 'PN_imu', 'test')):
    os.mkdir(os.path.join(data_path, 'PN_imu', 'test'))

pie = pie_data.PIE(data_path=data_path)
# pie.extract_and_save_images(extract_frame_type='annotated')
dataset = pie.generate_database()

videos = list(dataset.keys())
train_videos = [ 'set02', 'set04','set05']
val_videos = ['set06']
test_videos = ['set03']

ped_id=1
for video in dataset:
    print('Processing', video, '...')
    vid = dataset[video]
    seq=list(vid.keys())
    for k in seq:
        print('Processing', k, '...')
        vi_seq=vid[k]
        for ped in vi_seq['ped_annotations']:
            data = np.empty((0,23))
            if vi_seq['ped_annotations'][ped]['behavior']:
                frames = np.array(vi_seq['ped_annotations'][ped]['frames']).reshape(-1,1)
                ids = np.repeat(ped_id, frames.shape[0]).reshape(-1,1)
                bbox = np.array(vi_seq['ped_annotations'][ped]['bbox'])
                x = bbox[:,0].reshape(-1,1)
                y = bbox[:,1].reshape(-1,1)
                w = np.abs(bbox[:,0] - bbox[:,2]).reshape(-1,1)
                h = np.abs(bbox[:,1] - bbox[:,3]).reshape(-1,1)
                imagefolderpath = np.repeat(os.path.join(data_path, 'images', video.replace('video_', ''),k), frames.shape[0]).reshape(-1,1)

                 #behavior 
                converted_cross= [0 if x == -1 else x for x in vi_seq['ped_annotations'][ped]['behavior']['cross']]
                cross = np.array(converted_cross).reshape(-1,1)
                look = np.array(vi_seq['ped_annotations'][ped]['behavior']['look']).reshape(-1,1)
                action = np.array(vi_seq['ped_annotations'][ped]['behavior']['action']).reshape(-1,1)
                gesture = np.array(vi_seq['ped_annotations'][ped]['behavior']['gesture']).reshape(-1,1)

                # attributes
                age=np.repeat(vi_seq['ped_annotations'][ped]['attributes']['age'],frames.shape[0]).reshape(-1,1)
                gender=np.repeat(vi_seq['ped_annotations'][ped]['attributes']['gender'],frames.shape[0]).reshape(-1,1)

                #scene_attributes

                intersection=np.repeat(vi_seq['ped_annotations'][ped]['attributes']['intersection'],frames.shape[0]).reshape(-1,1)
                traffic_direction=np.repeat(vi_seq['ped_annotations'][ped]['attributes']['traffic_direction'],frames.shape[0]).reshape(-1,1)
                num_lanes=np.repeat(vi_seq['ped_annotations'][ped]['attributes']['num_lanes'],frames.shape[0]).reshape(-1,1)
                signalized=np.repeat(vi_seq['ped_annotations'][ped]['attributes']['signalized'],frames.shape[0]).reshape(-1,1)

                intention_prob=np.repeat(vi_seq['ped_annotations'][ped]['attributes']['intention_prob'],frames.shape[0]).reshape(-1,1) 

                #ego vehicle 
                accelx=[]
                accely=[]
                accelz=[]
                odb_speed=[]
                heading_angle=[]
                   
                for fr in range(len(frames)):
                    accelx.append(vi_seq['vehicle_annotations'][frames[fr][0]]['accX'])
                    accely.append(vi_seq['vehicle_annotations'][frames[fr][0]]['accY'])
                    accelz.append(vi_seq['vehicle_annotations'][frames[fr][0]]['accZ'])
                    odb_speed.append(vi_seq['vehicle_annotations'][frames[fr][0]]['OBD_speed'])
                    heading_angle.append(vi_seq['vehicle_annotations'][frames[fr][0]]['heading_angle'])
                
                accx=np.array(accelx).reshape(-1,1)
                accy=np.array(accely).reshape(-1,1)
                accz=np.array(accelz).reshape(-1,1)
                o_speed=np.array(odb_speed).reshape(-1,1)
                h_angle=np.array(heading_angle).reshape(-1,1)
                
                
                ped_data = np.hstack((frames, ids, x, y, w, h, imagefolderpath, cross,look,action,gesture,age,gender,intersection,traffic_direction,num_lanes,signalized,intention_prob,accx,accy,accz,o_speed,h_angle))
                data = np.vstack((data, ped_data))
            data_to_write = pd.DataFrame({'frame': data[:,0].reshape(-1), 
                                        'ID': data[:,1].reshape(-1), 
                                        'x': data[:,2].reshape(-1), 
                                        'y': data[:,3].reshape(-1), 
                                        'w': data[:,4].reshape(-1), 
                                        'h': data[:,5].reshape(-1), 
                                        'imagefolderpath': data[:,6].reshape(-1), 
                                        'crossing_true': data[:,7].reshape(-1),
                                        'look': data[:,8].reshape(-1),
                                        'action': data[:,9].reshape(-1),
                                        'gesture': data[:,10].reshape(-1),
                                        'age': data[:,11].reshape(-1),
                                        'gender': data[:,12].reshape(-1),
                                        'intersection': data[:,13].reshape(-1),
                                        'traffic_direction': data[:,14].reshape(-1),
                                        'num_lanes': data[:,15].reshape(-1),
                                        'signalized': data[:,16].reshape(-1),
                                        'intention_prob': data[:,17].reshape(-1),
                                        'accx': data[:,18].reshape(-1),
                                        'accy': data[:,19].reshape(-1),
                                        'accz': data[:,20].reshape(-1),
                                        'o_speed': data[:,21].reshape(-1),
                                        'h_angle': data[:,22].reshape(-1),
                                        })
            data_to_write['filename'] = data_to_write.frame
            data_to_write.filename = data_to_write.filename.apply(lambda x: '%05d'%int(x)+'.png')
            
            if video in train_videos:
                data_to_write.to_csv(os.path.join(data_path, 'PN_imu', 'train', '%05d'%int(ped_id)+'.csv'), index=False)
            elif video in val_videos:
                data_to_write.to_csv(os.path.join(data_path, 'PN_imu', 'val', '%05d'%int(ped_id)+'.csv'), index=False)
            elif video in test_videos:
                data_to_write.to_csv(os.path.join(data_path, 'PN_imu', 'test', '%05d'%int(ped_id)+'.csv'), index=False)
            ped_id=ped_id+1