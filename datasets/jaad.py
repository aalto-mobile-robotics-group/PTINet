import torch
import torchvision.transforms.functional as TF
import pandas as pd
from ast import literal_eval
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import utils
import copy
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
class JAAD(torch.utils.data.Dataset):
    def __init__(self, 
                data_dir,
                out_dir,
                dtype,
                input,
                output,
                stride,
                skip=1,
                task='bounding_box',
                from_file=False,
                save=True,
                use_images=False,
                use_attribute=False,
                use_opticalflow=False,
                image_resize=[240, 426]
                ):
        
        print('*'*30)
        print('Loading JAAD', dtype, 'data ...')

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.input = input
        self.output = output
        self.stride = stride
        self.skip = skip
        self.dtype = dtype
        self.task = task
        self.use_image = use_images
        self.use_attribute=use_attribute
        self.use_opticalflow=use_opticalflow
        self.image_resize = image_resize
        self.max_threads = 16


        self.filename = 'jaad_{}_{}_{}_{}.csv'.format(dtype, str(input),\
                                str(output), str(stride)) 
        
        if(from_file):
            sequence_centric = pd.read_csv(os.path.join(self.out_dir, self.filename))
            df = sequence_centric.copy()      
            for v in list(df.columns.values):
                print(v+' loaded')
                try:
                    df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
                except:
                    continue
            sequence_centric[df.columns] = df[df.columns]
            
        else:
            #read data
        #     print('Reading data files ...')
        #     df = pd.DataFrame()
        #     new_index=0
        #     for file in glob.glob(os.path.join(data_dir, dtype,"*")):
        #         temp = pd.read_csv(file)
        #         if not temp.empty:
        #             temp['file'] = [file for t in range(temp.shape[0])]

        #             #assign unique ID to each 
        #             for index in temp.ID.unique():
        #                 new_index += 1
        #                 temp.ID = temp.ID.replace(index, new_index)

        #             #sort rows by ID and frames
        #             temp = temp.sort_values(['ID', 'frame'], axis=0)

        #             # df = df.append(temp, ignore_index=True)
        #             df = pd.concat((df,temp),ignore_index=True)
            
        #     print('Processing data ...')
        #     #create sequence column
        #     df.insert(0, 'sequence', df.ID)
            
        #     df = df.apply(lambda row: utils.compute_center(row), axis=1)

        #     #reset index
        #     df = df.reset_index(drop = True)
            
        #     df['bounding_box'] = df[['x', 'y', 'w', 'h']].apply(lambda row: [row.x, row.y, row.w, row.h], axis=1)
            
        #     bb = df.groupby(['ID'])['bounding_box'].apply(list).reset_index(name='bounding_box')
        #     s = df.groupby(['ID'])['scenefolderpath'].apply(list).reset_index(name='scenefolderpath').drop(columns='ID')
        #     f = df.groupby(['ID'])['filename'].apply(list).reset_index(name='filename').drop(columns='ID')
        #     c = df.groupby(['ID'])['crossing_true'].apply(list).reset_index(name='crossing_true').drop(columns='ID')
        #     d = bb.join(s).join(f).join(c)
            
        #     d['label'] = d['crossing_true']
        #     d.label = d.label.apply(lambda x: 1 if 1 in x else 0)
            
        #     d = d.drop(d[d.bounding_box.apply(lambda x: len(x) < input + output)].index)
        #     d = d.reset_index(drop=True)
            
        #     bounding_box_o = np.empty((0,input,4))
        #     bounding_box_t = np.empty((0,output,4))
        #     scene_o = np.empty((0,input))
        #     file = np.empty((0,input))   
        #     cross_o = np.empty((0,input))
        #     cross = np.empty((0,output))
        #     ind = np.empty((0,1))

        #     for i in range(d.shape[0]):
        #         ped = d.loc[i]
        #         k = 0
        #         while (k+input+output) <= len(ped.bounding_box):
        #             ind = np.vstack((ind, ped['ID']))
        #             bounding_box_o = np.vstack((bounding_box_o, np.array(ped.bounding_box[k:k+input]).reshape(1,input,4)))
        #             bounding_box_t = np.vstack((bounding_box_t, np.array(ped.bounding_box[k+input:k+input+output]).reshape(1,output,4)))  
        #             scene_o = np.vstack((scene_o, np.array(ped.scenefolderpath[k:k+input]).reshape(1,input)))
        #             file = np.vstack((file, np.array(ped.filename[k:k+input]).reshape(1,input)))     
        #             cross_o = np.vstack((cross_o, np.array(ped.crossing_true[k:k+input]).reshape(1,input)))
        #             cross = np.vstack((cross, np.array(ped.crossing_true[k+input:k+input+output]).reshape(1,output)))

        #             k += stride
            
        #     dt = pd.DataFrame({'ID':ind.reshape(-1)})
        #     data = pd.DataFrame({'bounding_box':bounding_box_o.reshape(-1, 1, input, 4).tolist(),
        #                          'future_bounding_box':bounding_box_t.reshape(-1, 1, output, 4).tolist(),
        #                          'scenefolderpath':scene_o.reshape(-1,input).tolist(),
        #                          'filename':file.reshape(-1,input).tolist(),
        #                          'crossing_obs':cross_o.reshape(-1, input).tolist(),
        #                          'crossing_true':cross.reshape(-1,output).tolist()})
        #     data.bounding_box = data.bounding_box.apply(lambda x: x[0])
        #     data.future_bounding_box = data.future_bounding_box.apply(lambda x: x[0])
        #     data = dt.join(data)
            
        #     data = data.drop(data[data.crossing_obs.apply(lambda x: 1. in x)].index)
        #     data['label'] = data.crossing_true.apply(lambda x: 1. if 1. in x else 0.)
            
        #     if save:
        #         data.to_csv(os.path.join(self.out_dir, self.filename), index=False)
                
        #     sequence_centric = data.copy()
            
 
        # self.data = sequence_centric.copy().reset_index(drop=True)
            
        # print(dtype, "set loaded")
         #read data
            print('Reading data files ...')
            df = pd.DataFrame()
            new_index=0
            for file in glob.glob(os.path.join(data_dir, dtype,"*")):
                temp = pd.read_csv(file)
                if not temp.empty:
                    temp['file'] = [file for t in range(temp.shape[0])]

                    #assign unique ID to each 
                    for index in temp.ID.unique():
                        new_index += 1
                        temp.ID = temp.ID.replace(index, new_index)

                    #sort rows by ID and frames
                    temp = temp.sort_values(['ID', 'frame'], axis=0)

                    df = pd.concat((df,temp),ignore_index=True)
            
            print('Processing data ...')
            #create sequence column
            df.insert(0, 'sequence', df.ID)
            
            df = df.apply(lambda row: utils.compute_center(row), axis=1)

            #reset index
            df = df.reset_index(drop = True)
            
            df['bounding_box'] = df[['x', 'y', 'w', 'h']].apply(lambda row: [row.x, row.y, row.w, row.h], axis=1)

            df['ped_attribute'] = df[['age', 'gender', 'group_size']].apply(lambda row: [row.age, row.gender, row.group_size], axis=1)
            df['ped_behavior'] = df[['reaction','hand_gesture','look','nod']].apply(lambda row: [row.reaction, row.hand_gesture, row.look, row.nod], axis=1)
            df['scene_attribute'] = df[['designated', 'motion_direction', 'num_lanes','signalized','traffic_direction','ped_crossing','ped_sign','stop_sign','traffic_light','road_type']].apply(lambda row: [row.designated, row.motion_direction, row.num_lanes,row.signalized,row.traffic_direction,row.ped_crossing,row.ped_sign,row.stop_sign,row.traffic_light,row.road_type], axis=1)
            
            bb = df.groupby(['ID'])['bounding_box'].apply(list).reset_index(name='bounding_box')
            s = df.groupby(['ID'])['imagefolderpath'].apply(list).reset_index(name='imagefolderpath').drop(columns='ID')
            f = df.groupby(['ID'])['filename'].apply(list).reset_index(name='filename').drop(columns='ID')
            c = df.groupby(['ID'])['crossing_true'].apply(list).reset_index(name='crossing_true').drop(columns='ID')
            t = df.groupby(['ID'])['ped_attribute'].apply(list).reset_index(name='ped_attribute').drop(columns='ID')
            h = df.groupby(['ID'])['ped_behavior'].apply(list).reset_index(name='ped_behavior').drop(columns='ID')
            w = df.groupby(['ID'])['scene_attribute'].apply(list).reset_index(name='scene_attribute').drop(columns='ID')
            d = bb.join(s).join(f).join(c).join(t).join(h).join(w)
            
            d['label'] = d['crossing_true']
            d.label = d.label.apply(lambda x: 1 if 1 in x else 0)
            
            d = d.drop(d[d.bounding_box.apply(lambda x: len(x) < input + output)].index)
            d = d.reset_index(drop=True)
            
            bounding_box_o = np.empty((0,input,4))
            bounding_box_t = np.empty((0,output,4))
            scene_o = np.empty((0,input))
            file = np.empty((0,input))   
            cross_o = np.empty((0,input))
            cross = np.empty((0,output))
            ind = np.empty((0,1))
            p_attribute=np.empty((0,3))
            p_behavior=np.empty((0,input,4))
            s_attribute=np.empty((0,input,10))

            for i in range(d.shape[0]):
                ped = d.loc[i]
                k = 0
                while (k+input+output) <= len(ped.bounding_box):
                    ind = np.vstack((ind, ped['ID']))
                    p_attribute=np.vstack((p_attribute,ped['ped_attribute'][0]))
                    bounding_box_o = np.vstack((bounding_box_o, np.array(ped.bounding_box[k:k+input]).reshape(1,input,4)))
                    bounding_box_t = np.vstack((bounding_box_t, np.array(ped.bounding_box[k+input:k+input+output]).reshape(1,output,4)))  
                    scene_o = np.vstack((scene_o, np.array(ped.imagefolderpath[k:k+input]).reshape(1,input)))
                    p_behavior = np.vstack((p_behavior, np.array(ped.ped_behavior[k:k+input]).reshape(1,input,4)))
                    s_attribute = np.vstack((s_attribute, np.array(ped.scene_attribute[k:k+input]).reshape(1,input,10)))
                    file = np.vstack((file, np.array(ped.filename[k:k+input]).reshape(1,input)))     
                    cross_o = np.vstack((cross_o, np.array(ped.crossing_true[k:k+input]).reshape(1,input)))
                    cross = np.vstack((cross, np.array(ped.crossing_true[k+input:k+input+output]).reshape(1,output)))

                    k += stride
            
            dt = pd.DataFrame({'ID':ind.reshape(-1)})
            ped_dt = pd.DataFrame({'ped_attribute':p_attribute.reshape(-1,3).tolist()})
            data = pd.DataFrame({'bounding_box':bounding_box_o.reshape(-1, 1, input, 4).tolist(),
                                 'future_bounding_box':bounding_box_t.reshape(-1, 1, output, 4).tolist(),
                                 'ped_behavior':p_behavior.reshape(-1, 1, input, 4).tolist(),
                                 'scene_attribute':s_attribute.reshape(-1, 1, input, 10).tolist(),
                                 'imagefolderpath':scene_o.reshape(-1,input).tolist(),
                                 'filename':file.reshape(-1,input).tolist(),
                                 'crossing_obs':cross_o.reshape(-1, input).tolist(),
                                 'crossing_true':cross.reshape(-1,output).tolist()})
            data.bounding_box = data.bounding_box.apply(lambda x: x[0])
            data.future_bounding_box = data.future_bounding_box.apply(lambda x: x[0])
            data.ped_behavior = data.ped_behavior.apply(lambda x: x[0])
            data.scene_attribute= data.scene_attribute.apply(lambda x: x[0])
            data=ped_dt.join(data)
            data = dt.join(data)
            
            data = data.drop(data[data.crossing_obs.apply(lambda x: 1. in x)].index) ### remove cases in which the pedestrian is already on the road, and there is no chnage from crossing to non_crossing.
            data['label'] = data.crossing_true.apply(lambda x: 1. if 1. in x else 0.)
            
            if save:
                data.to_csv(os.path.join(self.out_dir, self.filename), index=False)
                
            sequence_centric = data.copy()
            
 
        self.data = sequence_centric.copy().reset_index(drop=True)
            
        print(dtype, "set loaded")
        


    def __len__(self):
        return len(self.data)

    def _read_images(self,image_paths):
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            images_list = list(executor.map(self._load_image, image_paths))
       
        # images_list=[transforms.functional.to_tensor(self._load_image(path).resize((self.image_resize[0], self.image_resize[1]))) for path in image_paths]
        images=torch.stack(images_list)
        return images

    def _load_image(self, image_path):
        image = Image.open(image_path).resize((self.image_resize[0], self.image_resize[1]))
        image=transforms.functional.to_tensor(image)
        return image

    def _read_images_op(self,image_paths_op):
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            images_list_op = list(executor.map(self._load_image_op, image_paths_op))
       
        # images_list=[transforms.functional.to_tensor(self._load_image(path).resize((self.image_resize[0], self.image_resize[1]))) for path in image_paths]
        images_op=torch.stack(images_list_op)
        return images_op
    
    def _load_image_op(self, image_path_op):
        image_op = Image.open(image_path_op).resize((self.image_resize[0], self.image_resize[1]))
        image_op=transforms.functional.to_tensor(image_op)
        return image_op
    

    def _read_images_cc(self,image_paths,bbox):
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            images_list = list(executor.map(self._load_image_CC, image_paths,bbox))
       
        # images_list=[transforms.functional.to_tensor(self._load_image(path).resize((self.image_resize[0], self.image_resize[1]))) for path in image_paths]
        images=torch.stack(images_list)
        return images
    
    def _load_image_CC(self, image_path,bbox):
        x,y,width,height=bbox
        x=x-(width/2)
        y=y-(height/2)
        x,y,width,height=int(x),int(y),int(width),int(height)
        img_cc = cv2.resize(np.array(Image.open(image_path))[y:y + height, x:x + width],(self.image_resize[0], self.image_resize[1]))
        # img_cc = image[y:y + height, x:x + width].resize((self.image_resize[0], self.image_resize[1])) 
        image_cc=transforms.functional.to_tensor(img_cc)
        # image_cc=self.normalize_imagenet(image_cc)
        return image_cc


        
    def __getitem__(self, index):
        seq = self.data.iloc[index]
        outputs = {}

        # observed = torch.tensor(np.array(seq.bounding_box))
        # future = torch.tensor(np.array(seq.future_bounding_box))
        
        obs = torch.tensor([seq.bounding_box[i] for i in range(0,self.input,self.skip)])
        obs_speed = (obs[1:] - obs[:-1])
    
        # outputs.append(obs_speed.type(torch.float32))
        
        
        true = torch.tensor([seq.future_bounding_box[i] for i in range(0,self.output,self.skip)])
        true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1]))
        # outputs.append(true_speed.type(torch.float32))
        # outputs.append(obs.type(torch.float32))
        # outputs.append(true.type(torch.float32))
        
        
        true_cross = torch.tensor([seq.crossing_true[i] for i in range(0,self.output,self.skip)])
        true_non_cross = torch.ones(true_cross.shape, dtype=torch.int64)-true_cross
        true_cross = torch.cat((true_non_cross.unsqueeze(1), true_cross.unsqueeze(1)), dim=1)
        cross_label = torch.tensor(seq.label)
        # outputs.append(true_cross.type(torch.float32))
        # outputs.append(cross_label.type(torch.float32))
        if self.use_attribute==True:
            ped_behavior = torch.tensor([seq.ped_behavior[i] for i in range(0,self.input,self.skip)])
            scene_attribute = torch.tensor([seq.scene_attribute[i] for i in range(0,self.input,self.skip)])
            ped_attribute=torch.tensor(seq.ped_attribute)
        else:
            ped_behavior=torch.empty(1,1)
            scene_attribute=torch.empty(1,1)
            ped_attribute=torch.empty(1,1)
            

        if self.use_image ==True:   
              
            image_paths = [seq.imagefolderpath[frame] for frame in range(0,self.input,self.skip)]
            images =self._read_images(image_paths)
            # optical =self._read_images_op(image_paths_optical)
            # images =self._read_images_cc(image_paths,seq.bounding_box)
            # optical =self._read_images_cc(image_paths_optical,seq.bounding_box)
            # images = torch.tensor([])
            # for i, path in enumerate(image_paths):
            #     image = Image.open(path)
            #     #bb = obs[i,:]
            #     #img = ImageDraw.Draw(scene)   
            #     #utils.drawrect(img, ((bb[0]-bb[2]/2, bb[1]-bb[3]/2), (bb[0]+bb[2]/2, bb[1]+bb[3]/2)), width=5)
            #     image = self.scene_transforms(image)
            #     images = torch.cat((images, image.unsqueeze(0)))
        else:
            images=torch.empty(1,1)

        if self.use_opticalflow==True:
            image_paths = [seq.imagefolderpath[frame] for frame in range(0,self.input,self.skip)]
            op_path=copy.deepcopy(image_paths)
            image_paths_optical= []
            for ul in op_path:
                image_paths_optical.append(ul.replace('images', 'opticalflow'))
            optical =self._read_images_op(image_paths_optical)

        else:
            optical=torch.empty(1,1)


        

            
        outputs={'image':images,
                'optical':optical,
                'ped_attribute':ped_attribute,
                'scene_attribute':scene_attribute,
                'ped_behavior':ped_behavior,
                'cross_label':cross_label, 
                'future_cross':true_cross,
                'future_speed':true_speed,
                'speed':obs_speed,
                'pos':obs,
                'future_pos':true,
                'id':seq.ID}
        
        
        
        return outputs


    def scene_transforms(self, scene):  
        scene = TF.resize(scene, size=(self.image_resize[0], self.image_resize[1]))
        scene = TF.to_tensor(scene)
        
        return scene