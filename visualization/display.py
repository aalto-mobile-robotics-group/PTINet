from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import random
import os
import torchvision.transforms as trans


def plot_image_tensor(img_tensor):
    Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')
    pil_img = Tensor2PIL(img_tensor)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    plt.imshow(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), aspect='equal')


def draw_ped_bbox(image, bbox,color=(0, 0, 255)):
    thickness = 2
    # color = (0, 0, 255)
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image
def convert_cwh(bbox):
  
    x_center, y_center, width, height = bbox
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    
    return [x_min, y_min, x_max, y_max]

def ped_attribute_text (value):

    labels=[]   
    ped_attribute_label=['age', 'gender', 'group_size']
    color=[(66,104,124),(132,165,184),(179,218,241)]
    map_dic = {
                   'age': { 0:'child', 1:'young',  2:'adult',  3:'senior'},
                   'gender': {0:'n/a', 1:'female', 2:'male'},
                   }
    lab1=map_dic[ped_attribute_label[0]][value[0]]
    lab2=map_dic[ped_attribute_label[1]][value[1]]
    labels.append(('age',lab1,color[0]))
    labels.append(('gender',lab2,color[1]))
    labels.append(('group_size',value[2],color[2]))

    return labels

def ped_behavior_text (value):

    ped_behavior_label=['reaction','hand_gesture','look','nod']
    color=[(66,133,244),(52,168,83),(251,188,5),(234,67,53)]
    map_dic = {
                  
                   'nod': {0: 'n/a', 1: 'nodding'},
                   'look': {0: 'not-looking', 1: 'looking'},
                   'hand_gesture': {0: 'n/a', 1: 'greet',
                                    2: 'yield', 3: 'rightofway',
                                    4: 'other'},
                   'reaction': {0: 'n/a', 1: 'clear_path',
                                2: 'speed_up', 3: 'slow_down'},
                   }
    lab=[]
    for i in range(len(value)):
        ds=map_dic[ped_behavior_label[i]][value[i]]
        lab.append((ped_behavior_label[i],ds,color[i]))

    return lab

def ped_scene_text (value):

    ped_behavior_label=['designated', 'motion_dir', 'num_lanes','signalized','traffic_dir','ped_crossing','ped_sign','stop_sign','traffic_light','road_type']
    color=[(255,168,0),(154,205,50),(70,130,180),(234,67,53),(86,180,233),(0,158,115),(240,228,66),(0,114,178),(66,133,244),(52,168,83)]
    map_dic = {
                  
                  'designated': {0: 'no', 1: 'yes'},
                  'ped_sign': {0: 'no', 1: 'yes'},
                  'stop_sign': {0: 'no', 1: 'yes'},
                  'ped_crossing': {0: 'no', 1: 'yes'},
                   'motion_dir': {0: 'n/a', 1: 'lat', 2: 'long'},
                   'traffic_dir': {0: 'ow', 1: 'tw'},
                   'signalized': {0: 'n/a', 1: 'no', 2: 'yes'},
                   'road_type': {0: 'street', 1: 'parking_lot', 2: 'garage'},
                   'traffic_light': {0: 'n/a', 1: 'R', 2: 'G'}
                   }
    lab=[]
    for i in range(len(value)):

        try:
            ds=map_dic[ped_behavior_label[i]][value[i]]
        except:
            ds=int(value[i])

        lab.append((ped_behavior_label[i],ds,color[i]))

    return lab

def draw_ped_attribute(cv2_img,labels,offset,thickness):
    h,w,_=cv2_img.shape
    div= len(labels)
    seg=w/div
    thickness = thickness
    font=cv2.FONT_ITALIC
    for i in range(div):
        rect_x = seg*i
        rect_y = h-offset-thickness 
        rect_width = seg*(i+1)
        rect_height = h-offset
        color = color=random.sample(range(0, 256),3)
        start_point = (int(rect_x), int(rect_y))
        end_point = (int(rect_width),int(rect_height ))
        image = cv2.rectangle(cv2_img, start_point, end_point, labels[i][2], thickness)
        image=cv2.putText((cv2_img),labels[i][0]+':'+str(labels[i][1]), (start_point[0],start_point[1]+20), font, (0.7), (0,0,0), 2, cv2.LINE_AA)

    return image

def draw_bar_plots(image,cross_future,cross_pred,bbox):
    ped_c=np.argmax(cross_pred.view(-1,2).detach().cpu().numpy(), axis=1)
    # Count instances for each category
    gt_crossing_count = np.sum(np.array(cross_future[:, 0]))
    gt_non_crossing_count = np.sum(np.array(cross_future[:, 1]))
    pred_crossing_count = np.count_nonzero(ped_c == 0)
    pred_non_crossing_count = np.count_nonzero(ped_c == 1)

    # Scale counts to max length of 16
    max_bar_length = 160
    gt_crossing_len = int((gt_crossing_count / 16) * max_bar_length)
    gt_non_crossing_len = int((gt_non_crossing_count / 16) * max_bar_length)
    pred_crossing_len = int((pred_crossing_count / 16) * max_bar_length)
    pred_non_crossing_len = int((pred_non_crossing_count / 16) * max_bar_length)

    # Initialize bar location and dimensions
    initial_y = int(bbox[1] -bbox[3]//2)-30
    middle_x =int( bbox[0] + bbox[2] // 2)

    # Draw bars (each bar is 20 pixels high)
    cv2.rectangle(image, (middle_x - gt_crossing_len, initial_y), (middle_x, initial_y + 20),  (255, 0, 0), -1) #ground truth
    cv2.rectangle(image, (middle_x, initial_y), (middle_x + gt_non_crossing_len, initial_y + 20), (0, 255, 0), -1) #ground truth non-crossing
    cv2.rectangle(image, (middle_x - pred_crossing_len, initial_y - 20), (middle_x, initial_y), (0, 0, 255), -1) # pred crosss
    cv2.rectangle(image, (middle_x, initial_y - 20), (middle_x + pred_non_crossing_len, initial_y), (255, 255, 0), -1) # pred_non crossing

    # cv2.line(image, (middle_x, initial_y - 21), (middle_x, initial_y + 21), (255, 255, 255), 1)    # Add text labels
    cv2.line(image, (middle_x - max_bar_length, initial_y), (middle_x + max_bar_length, initial_y), (255, 255, 255), 1)

    cv2.putText(image, 'GT', (middle_x - 5, initial_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, 'Pred', (middle_x - 20, initial_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, 'Crossing', (middle_x - max_bar_length, initial_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, 'Not Crossing', (middle_x + 5, initial_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw a black boundary around the bars
    cv2.rectangle(image, (middle_x - max_bar_length, initial_y - 20), (middle_x + max_bar_length, initial_y + 20), (255, 255, 255), 1)



    return image

  

class visualizer:
    def __init__(self, sample,speed_preds,crossing_preds,intentions,path='/home/farzeen/work/aa_postdoc/intent/intention_prediction/visual/'):
        self.sample = sample
        self.path=path
        self.speed_pred=speed_preds
        self.crossing=crossing_preds
        self.intent=intentions

    def show_frame(self, k=0, title=None,save=True):
        """
        Visualize kth frame in history sample
        """
        font=cv2.FONT_ITALIC
        sample = self.sample
        cross_pred=self.crossing
        cross_future=sample['future_cross'][0]
        pred=self.speed_pred.cpu()
        gt=sample['future_pos'][0]
        bbox = convert_cwh(sample['future_pos'][0][k])
        bbox_p = convert_cwh(pred[0][k])
    
        img_path = sample['image_path_p'][k][0]
        image = Image.open(img_path)
        # transforms = trans.Resize(size = (1080,1920))
        # img_tensor = transforms(img_tensor)
        pid = sample['id']
        # Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')
        # pil_img = Tensor2PIL(img_tensor)
        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = draw_ped_bbox(cv2_img, bbox, color=(255, 255, 0))
        image = draw_ped_bbox(cv2_img, bbox_p,color=(0, 0, 255))
        for jj in range(0,16):
            cv2.circle(image, (int(pred[0][jj][0]), int(pred[0][jj][1])), 4, (0,0,255), -1)
            cv2.circle(image, (int(gt[jj][0]), int(gt[jj][1])), 4, (255,0,5), -1)
        # labels=ped_attribute_text (sample['ped_attribute'].tolist()[0])
        image=draw_bar_plots(image,cross_future,cross_pred,gt[0])
       
        # image=draw_ped_attribute(cv2_img,labels,0,30)
        # label=ped_behavior_text (sample['ped_behavior'][0][k].tolist())
        # image=draw_ped_attribute(cv2_img,label,50,20)
        # label2=ped_scene_text (sample['scene_attribute'][0][k].tolist())
        # image=draw_ped_attribute(cv2_img,label2,90,20)
        # image=cv2.putText((cv2_img),'ped_id:'+str(int(sample['id'])), (100,100), font, (1), (0,0,0), 2, cv2.LINE_AA)
        # if int(sample['cross_label'])==0:
        #     image=cv2.putText((cv2_img),'Not Crossing', (250,100), font, (1), (0,0,0), 2, cv2.LINE_AA)
        # else:
        #     image=cv2.putText((cv2_img),'Crossing', (250,100), font, (1), (0,0,0), 2, cv2.LINE_AA)
        # image=draw_ped_behavior(cv2_img,label)
       
        # plt.title(title)
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), aspect='equal')
        if save==True:
            # os.mkdir(os.path.join(self.path,str(int(sample['id'])) ))
            plt.imsave(self.path+'ped_id_'+str(int(sample['id']))+'_seq_'+str(k)+'_'+img_path.split('/')[-2]+'_'+img_path.split('/')[-1],image[:,:,::-1])

        return image

    def show_sequence(self,save=True ):
        """
        Visualize the whole sequence sample
        """
        sample = self.sample
        pid = sample['id']
        for i in range(sample['image'].size()[1]):
            image=self.show_frame(k=i, title=None,save=save)

       