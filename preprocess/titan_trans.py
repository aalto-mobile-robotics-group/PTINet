import pandas as pd
import os
import numpy as np
import copy
import copy

map_dic = {'Communicative': {0: 'none of the above', 1: 'talking on phone', 2: 'looking at phone', 3: 'talking in group'},
                   'Atomic Actions': {0: 'laying down', 1: 'walking',2:'sitting',3:'standing',4: 'kneeling',5:'squatting',6:'running',7:'bending'},
                   'look': {0: 'not-looking', 1: 'looking'},
                   'Simple Context': {0: 'none of the above', 1: 'crossing a street at pedestrian crossing',
                                    2: 'walking on the road', 3: 'walking along the side of the road',
                                    4: 'jaywalking (illegally crossing NOT at pedestrian crossing)', 5: 'waiting to cross street'},
                   'age': {0: 'child', 2: 'adult', 3: 'senior'},}
  # Annotation processing helpers
def _map_text_to_scalar( label_type, value):
        """
        Maps a text label in XML file to scalars
        :param label_type: The label type
        :param value: The text to be mapped
        :return: The scalar value
        """
        map_dic = {'Communicative': {'none of the above': 0, 'talking on phone': 1, 'looking into phone': 2,'talking in group':3},
                   'Atomic Actions': {'laying down': 0, 'walking': 1,'sitting':2,'standing':3,'kneeling':4,'squatting':5,'running':6,'bending':7},
                   'Simple Context': {'none of the above': 0, 'crossing a street at pedestrian crossing': 1,'walking on the road':2,'walking along the side of the road':3,'jaywalking (illegally crossing NOT at pedestrian crossing)':4,'waiting to cross street':5,'cleaning an object':0,'exiting a building':0,'entering a building':0,'biking':0,'closing':0,'opening':0,'motorcycling':0},
                   'age': {'child': 0, 'adult': 2, 'senior': 3,'senior over 65':4},}

        return map_dic[label_type][value]
def get_split_vids_titan(split_vids_path, image_set="all") -> list:
    """
        Returns a list of video ids for a given data split
        :param:  split_vids_path: path of TITAN split
                image_set: Data split, train, test, val
        :return: The list of video ids
        """
    assert image_set in ["train", "test", "val", "all"]
    vid_ids = []
    sets = [image_set ] if image_set != 'all' else ['train_set', 'test_set', 'val_set']
    for s in sets:
        vid_id_file = os.path.join(split_vids_path, s + '.txt')
        with open(vid_id_file, 'rt') as fid:
            vid_ids.extend([x.strip() for x in fid.readlines()])

    return vid_ids
def is_sorted(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

def read_csv_titan(anns_dir, vid):
    video_number = int(vid.split("_")[1])
    df = pd.read_csv(os.path.join(anns_dir, vid + '.csv'))
    veh_rows = df[df['label'] != "person"].index
    df.drop(veh_rows, inplace=True)
    df.drop([df.columns[1], df.columns[7], df.columns[8],df.columns[9],
             df.columns[11], df.columns[14]],
            axis='columns', inplace=True)
    df.sort_values(by=['obj_track_id', 'frames'], inplace=True)
    ped_info_raw = df.values.tolist()
    pids = df['obj_track_id'].values.tolist()
    
    pids = list(set(list(map(int, pids))))

    if is_sorted(pids):
        return video_number, ped_info_raw, pids
    else:
        pids.sort()
        return video_number, ped_info_raw, pids


# def get_ped_info_titan(anns_dir, vids) -> dict:
#     ped_info = {}
#     for vid in vids:
#         video_number, ped_info_raw, pids = read_csv_titan(anns_dir, vid)
#         video_number_copy=copy.deepcopy(video_number)
#         n = len(pids)
#         ped_info[vid] = {}
#         flag = 0
#         for i in range(n):
#             idx = f"ped_{video_number_copy}_{i + 1}"
#             ped_info[vid][idx] = {}
#             ped_info[vid][idx]["frames"] = []
#             ped_info[vid][idx]["bbox"] = []
#             ped_info[vid][idx]["crossing"] = []
#             ped_info[vid][idx]['behavior'] = []
#             ped_info[vid][idx]['age'] = []
#             # anns[vid][idx]["cross"] = []
#             for j in range(flag, len(ped_info_raw)):
#                 if ped_info_raw[j][1] == pids[i]:
#                     ele = ped_info_raw[j]
#                     # t = int(ele[0].split('.')[0])
#                     # box = list([ele[3], ele[2], ele[3] + ele[5], ele[2] + ele[4]])
#                     box = list(map(round, [ ele[2], ele[3] ,ele[4], ele[5]]))
#                     box = list(map(float, box))
#                     action = 1 if ele[6] == "walking" else 0
#                     ped_info[vid][idx]['frames'].append(ele[0])
#                     ped_info[vid][idx]['bbox'].append(box)
#                     ped_info[vid][idx]['crossing'].append(action)
#                     ped_info[vid][idx]['behavior'].append([-1,-1,-1,-1])
#                     ped_info[vid][idx]['age'].append([-1,-1,-1,-1])
#                 else:
#                     flag += len(ped_info[vid][idx]["frames"])
#                     break
#             ped_info[vid][idx]['old_id'] = vid + f'_{pids[i]}'
#             ped_info[vid][idx]['attributes'] = [-1,-1,-1,-1,-1,-1]

#     return ped_info


def convert_anns_titan(anns_dir, vids) -> dict:
    anns = {}
    path='/media/farzeen/sam/sam/titan_data/dataset/images_anonymized/'
    data_path='//media/farzeen/sam/sam/titan_data/dataset/PN/'
    ped_count=1
    for vid in vids:
        print('processing',vid)
        video_number, ped_info_raw, pids = read_csv_titan(anns_dir, vid)
        video_number_copy=copy.deepcopy(video_number)
        n = len(pids)
        flag = 0
        for i in range(n):
            data = np.empty((0,11))
            idx = f"ped_{video_number_copy}_{i + 1}"
            print(idx)
            anns[idx] = {}
            anns[idx]["frames"] = []
            anns[idx]["bbox"] = []
            anns[idx]["crossing"] = []
            anns[idx]['age'] = []
            anns[idx]['communication'] = []
            anns[idx]['action_s'] = []
            anns[idx]['ID'] = []
            anns[idx]['imagefolderpath'] = []
            # anns[idx]['video_id'] = []
            
            # anns[vid][idx]["cross"] = []
            for j in range(flag, len(ped_info_raw)):
                if ped_info_raw[j][1] == pids[i]:
                    ele = ped_info_raw[j]
                    t = int(ele[0].split('.')[0])
                    # box = list([ele[3], ele[2], ele[3] + ele[5], ele[2] + ele[4]])
                    box = list(map(round, [ ele[2], ele[3],ele[4] , ele[5]]))
                    box = list(map(float, box))
                    if ele[7] in ["walking", 'running']:
                        action = 1
                    elif ele[7] in ["standing"]:
                        action = 0
                    else:
                        action = 0
                    age=_map_text_to_scalar('age',ele[9])
                    communication=_map_text_to_scalar('Communicative',ele[6])
                    action_s=_map_text_to_scalar('Simple Context',ele[8])

                    anns[idx]['frames'].append(ele[0])
                    anns[idx]['bbox'].append(box)
                    anns[idx]['crossing'].append(action)
                    anns[idx]['age'].append(age)
                    anns[idx]['communication'].append(communication)
                    anns[idx]['action_s'].append(action_s)
                    anns[idx]['ID'].append(ped_count)
                    # anns[idx]['video_id'].append(vid)
                    anns[idx]['imagefolderpath'].append(os.path.join(path, vid,'images'))
                    
                
                else:
                    flag += len(anns[idx]["frames"])
                    
                    break
            # anns[idx]['video_number'] = vid
            # anns[idx]['old_id'] = vid + f'_{pids[i]}'
                
            frames = np.array(anns[idx]['frames']).reshape(-1,1)   
            bbox = np.array(anns[idx]['bbox'])
            x = bbox[:,1].reshape(-1,1)
            y = bbox[:,0].reshape(-1,1)
            w = bbox[:,3].reshape(-1,1)
            h = bbox[:,2].reshape(-1,1)
            crossing = np.array(anns[idx]['crossing']).reshape(-1,1)   
            age = np.array(anns[idx]['age']).reshape(-1,1)   
            communication = np.array(anns[idx]['communication']).reshape(-1,1)   
            action_s = np.array(anns[idx]['action_s']).reshape(-1,1)   
            ids = np.array(anns[idx]['ID']).reshape(-1,1)   
            # video_number = np.array(anns[idx]['video_id']).reshape(-1,1)   
            imagefolderpath = np.array(anns[idx]['imagefolderpath']).reshape(-1,1)  
           
            ped_data = np.hstack((frames, ids, x, y, w, h, imagefolderpath, crossing,age,action_s,communication))
            data = np.vstack((data, ped_data))

            data_to_write = pd.DataFrame({'frame': data[:,0].reshape(-1), 
                                        'ID': data[:,1].reshape(-1), 
                                        'x': data[:,2].reshape(-1), 
                                        'y': data[:,3].reshape(-1), 
                                        'w': data[:,4].reshape(-1), 
                                        'h': data[:,5].reshape(-1), 
                                        'imagefolderpath': data[:,6].reshape(-1), 
                                        'crossing_true': data[:,7].reshape(-1),
                                        'age': data[:,8].reshape(-1),
                                        'action_s': data[:,9].reshape(-1),
                                        'communication': data[:,10].reshape(-1),
                                        })
            
            data_to_write['img_file'] =data_to_write['imagefolderpath'] + '/' + data_to_write['frame']
        
            
            data_to_write.to_csv(os.path.join(data_path,  'test','%05d'%int(ped_count)+'.csv'), index=False)
            ped_count=ped_count+1
        anns_new = {}
        ids = list(anns.keys())
        for k in ids:
            if len(anns[k]['crossing']) > 0:
                anns_new[k] = {}
                anns_new[k]['frames'] = anns[k]['frames']
                anns_new[k]['bbox'] = anns[k]['bbox']
                anns_new[k]['crossing'] = anns[k]['crossing']
                anns_new[k]['imagefolderpath'] = anns[k]['imagefolderpath']
                anns_new[k]['age'] = anns[k]['age']
                anns_new[k]['action_s'] = anns[k]['action_s']
                anns_new[k]['communication'] = anns[k]['communication']
                # anns_new[k]['video_number'] = anns[k]['video_number']
                anns_new[k]['ID'] = anns[k]['ID']
                # anns_new[k]['attributes'] = [-1,-1,-1,-1,-1,-1]
                
    print(ped_count)
    return anns_new


# def add_trans_label_titan(anns, verbose=False) -> None:
#     """
#     Add labels to show the time (number of frames)
#     away from next action transition
#     """
#     all_wts = 0  # walking to standing
#     all_stw = 0  # standing to walking
#     pids = list(anns.keys())
#     for idx in pids:
#         action = anns[idx]['action']
#         frames = anns[idx]['frames']
#         n_frames = len(frames)
#         anns[idx]['next_transition'] = []
#         stw_time = []
#         wts_time = []
#         for j in range(len(action) - 1):
#             if action[j] == 0 and action[j + 1] == 1:
#                 all_stw += 1
#                 stw_time.append(frames[j + 1])
#             elif action[j] == 1 and action[j + 1] == 0:
#                 all_wts += 1
#                 wts_time.append(frames[j + 1])
#         # merge
#         trans_time_ped = np.array(sorted(stw_time + wts_time))
#         # set transition tag
#         for i in range(n_frames):
#             t = frames[i]
#             future_trans_ped = trans_time_ped[trans_time_ped >= t]
#             if future_trans_ped.size > 0:
#                 next_trans_ped = future_trans_ped[0]
#                 anns[idx]['next_transition'].append(next_trans_ped - t)
#             else:
#                 anns[idx]['next_transition'].append(None)
#     if verbose:
#         print('----------------------------------------------------------------')
#         print("TITAN:")
#         print(f'Total number of standing to walking transitions (raw): {all_stw}')
#         print(f'Total number of walking to standing transitions  (raw): {all_wts}')

#     return None


def build_ped_dataset_titan(anns_dir, split_vids_path, image_set="all", verbose=False) -> dict:
    """
    Build pedestrian dataset from TITAN annotations
    """
    assert image_set in ['train', 'test', 'val', "all"], "Image set should be train, test or val"
    vids = get_split_vids_titan(split_vids_path, image_set)
    ped_dataset = convert_anns_titan(anns_dir, vids)
    # add_trans_label_titan(ped_dataset, verbose=verbose)

    return ped_dataset


class TitanTransDataset:
    """
     dataset class for transition-related pedestrian samples in TITAN
    """

    def __init__(self, anns_dir, split_vids_path, image_set="all", verbose=False):
        assert image_set in ['train', 'test', 'val', "all"], " Name should be train, test, val or all"
        self.dataset = build_ped_dataset_titan(anns_dir, split_vids_path, image_set, verbose)
        self.name = image_set

    def __repr__(self):
        return f"TitanTransDataset(image_set={self.name})"

    # def extract_trans_frame(self, mode="GO", frame_ahead=0, fps=10, verbose=False) -> dict:
    #     dataset = self.dataset
    #     assert mode in ["GO", "STOP"], "Transition type should be STOP or GO"
    #     ids = list(dataset.keys())
    #     samples = {}
    #     step = 10 // fps
    #     t_ahead = step * frame_ahead
    #     j = 0
    #     for idx in ids:
    #         vid_id = copy.deepcopy(dataset[idx]['video_number'])
    #         frames = copy.deepcopy(dataset[idx]['frames'])
    #         bbox = copy.deepcopy(dataset[idx]['bbox'])
    #         action = copy.deepcopy(dataset[idx]['action'])
    #         next_transition = copy.deepcopy(dataset[idx]["next_transition"])
    #         behavior = copy.deepcopy(dataset[idx]['behavior'])
    #         attributes = copy.deepcopy(dataset[idx]['attributes'])
    #         for i in range(len(frames)):
    #             key = None
    #             if mode == "GO":
    #                 if next_transition[i] == 0 and action[i] == 1:
    #                     j += 1
    #                     new_id = "{:04d}".format(j) + "_" + self.name
    #                     key = "TG_" + new_id
    #                     old_id = copy.deepcopy(dataset[idx]['old_id'])
    #             if mode == "STOP":
    #                 if next_transition[i] == 0 and action[i] == 0:
    #                     j += 1
    #                     new_id = "{:04d}".format(j) + "_" + self.name
    #                     key = "TS_" + new_id
    #                     old_id = copy.deepcopy(dataset[idx]['old_id'])
    #             if key is not None and i - t_ahead >= 0:
    #                 samples[key] = {}
    #                 samples[key]["source"] = "TITAN"
    #                 samples[key]["old_id"] = old_id
    #                 samples[key]['video_number'] = vid_id
    #                 samples[key]['frame'] = frames[i - t_ahead]
    #                 samples[key]['bbox'] = bbox[i - t_ahead]
    #                 samples[key]['action'] = action[i - t_ahead]
    #                 samples[key]['behavior'] = [-1,-1,-1,-1]
    #                 samples[key]['attributes'] = [-1,-1,-1,-1,-1,-1]
    #                 samples[key]['frame_ahead'] = frame_ahead
    #                 samples[key]['type'] = mode
    #                 samples[key]['fps'] = fps
    #     if verbose:
    #         print(f"Extract {len(samples.keys())} {mode} sample frames from TITAN {self.name} set")

    #     return samples

    # def extract_trans_history(self, mode="GO", fps=10, max_frames=None, post_frames=0, verbose=False) -> dict:
    #     """
    #     Extract the whole history of pedestrian up to the frame when transition happens
    #     :params: mode: target transition type, "GO" or "STOP"
    #              fps: frame-per-second, sampling rate of extracted sequences, default 30
    #              verbose: optional printing of sample statistics
    #     """
    #     dataset = self.dataset
    #     assert mode in ["GO", "STOP"], "Transition type should be STOP or GO"
    #     ids = list(dataset.keys())
    #     samples = {}
    #     j = 0
    #     step = 10 // fps
    #     assert isinstance(step, int)
    #     for idx in ids:
    #         vid_id = copy.deepcopy(dataset[idx]['video_number'])
    #         frames = copy.deepcopy(dataset[idx]['frames'])
    #         bbox = copy.deepcopy(dataset[idx]['bbox'])
    #         action = copy.deepcopy(dataset[idx]['action'])
    #         behavior = copy.deepcopy(dataset[idx]['behavior'])
    #         # old_id = copy.deepcopy(dataset[idx]['old_id'])
    #         # cross = copy.deepcopy(dataset[idx]['cross'])
    #         next_transition = copy.deepcopy(dataset[idx]["next_transition"])
    #         for i in range(len(frames)):
    #             key = None
    #             if mode == "GO":
    #                 if next_transition[i] == 0 and action[i] == 1:
    #                     j += 1
    #                     new_id = "{:04d}".format(j) + "_" + self.name
    #                     key = "TG_" + new_id
    #                     old_id = copy.deepcopy(dataset[idx]['old_id'])
    #                     ae = np.array(action[i::-step])
    #                     ce = np.array(np.nonzero(ae == 1))
    #                     d_pre = ce[0][1] - 1 if ce.size > 1 else len(ae) - 1
    #                     ap = np.array(action[i::step])
    #                     cp = np.array(np.nonzero(ap == 0))
    #                     d_pos = cp[0][0] if cp.size > 0 else len(ap)
    #             if mode == "STOP":
    #                 if next_transition[i] == 0 and action[i] == 0:
    #                     j += 1
    #                     new_id = "{:04d}".format(j) + "_" + self.name
    #                     key = "TS_" + new_id
    #                     old_id = copy.deepcopy(dataset[idx]['old_id'])
    #                     ae = np.array(action[i::-step])
    #                     ce = np.array(np.nonzero(ae == 0))
    #                     d_pre = ce[0][1] - 1 if ce.size > 1 else len(ae) - 1
    #                     ap = np.array(action[i::step])
    #                     cp = np.array(np.nonzero(ap == 1))
    #                     d_pos = cp[0][0] if cp.size > 0 else len(ap)
    #             if key is not None:
    #                 if max_frames is None:
    #                     t = None
    #                 else:
    #                     t = i - max_frames * step if (i - max_frames * step >= 0) else None
    #                 i = i + min(post_frames, d_pos) * step
    #                 samples[key] = {}
    #                 samples[key]["source"] = "TITAN"
    #                 samples[key]["old_id"] = old_id
    #                 samples[key]['video_number'] = vid_id
    #                 samples[key]['frame'] = frames[i:t:-step]
    #                 samples[key]['frame'].reverse()
    #                 samples[key]['bbox'] = bbox[i:t:-step]
    #                 samples[key]['bbox'].reverse()
    #                 samples[key]['action'] = action[i:t:-step]
    #                 samples[key]['action'].reverse()
    #                 samples[key]['behavior'] = behavior[i:t:-step]
    #                 samples[key]['attributes'] = [-1,-1,-1,-1,-1,-1]
    #                 samples[key]['pre_state'] = d_pre
    #                 samples[key]['post_state'] = d_pos
    #                 samples[key]['type'] = mode
    #                 samples[key]['fps'] = fps

    #     if verbose:
    #         keys = list(samples.keys())
    #         pids = []
    #         num_frames = 0
    #         for k in keys:
    #             pids.append(samples[k]['old_id'])
    #             num_frames += len(samples[k]['frame'])
    #         print(f"Extract {len(pids)} {mode} history samples from {self.name} dataset in TITAN,")
    #         print(f"samples contain {len(set(pids))} unique pedestrians and {num_frames} frames.")

    #     return samples

    # def extract_non_trans(self, fps=10, max_frames=None, max_samples=None, verbose=False):
    #     dataset = self.dataset
    #     ids = list(dataset.keys())
    #     samples = {'walking': {}, 'standing': {}}
    #     step = 10 // fps
    #     h = 5
    #     assert isinstance(step, int)
    #     jw = 0
    #     js = 0
    #     t = max_frames * step if max_frames is not None else None
    #     for idx in ids:
    #         vid_id = copy.deepcopy(dataset[idx]['video_number'])
    #         frames = copy.deepcopy(dataset[idx]['frames'])
    #         bbox = copy.deepcopy(dataset[idx]['bbox'])
    #         action = copy.deepcopy(dataset[idx]['action'])
    #         behavior = copy.deepcopy(dataset[idx]['behavior'])
    #         a = np.array(action)  # action array
    #         key = None
    #         action_type = None
    #         old_id = None
    #         if a[a < 0.5].size == 0:  # all walking
    #             jw += 1
    #             new_id = "{:04d}".format(jw) + "_" + self.name
    #             key = "TW_" + new_id
    #             old_id = idx
    #             action_type = 'walking'
    #         elif a[a > 0.5].size == 0 and  a[a < -0.5].size==0 :  # all standing
    #             js += 1
    #             new_id = "{:04d}".format(js) + "_" + self.name
    #             key = "TN_" + new_id
    #             old_id = idx
    #             action_type = 'standing'
    #         if max_frames is None:
    #             t = None
    #         else:
    #             t = len(frames) - max_frames * step if (len(frames) - max_frames * step >= 0) else None
    #         if key is not None:
    #             samples[action_type][key] = {}
    #             samples[action_type][key]["source"] = "TITAN"
    #             samples[action_type][key]["old_id"] = old_id
    #             samples[action_type][key]['video_number'] = vid_id
    #             samples[action_type][key]['frame'] = frames[-1:t:-step]
    #             samples[action_type][key]['frame'].reverse()
    #             samples[action_type][key]['bbox'] = bbox[-1:t:-step]
    #             samples[action_type][key]['bbox'].reverse()
    #             samples[action_type][key]['action'] = action[-1:t:-step]
    #             samples[action_type][key]['action'].reverse()
    #             samples[action_type][key]['behavior'] = behavior[-1:t:-step]
    #             samples[action_type][key]['attributes'] = [-1,-1,-1,-1,-1,-1]
    #             samples[action_type][key]['action_type'] = action_type
    #             samples[action_type][key]['fps'] = fps
    #     samples_new = {'walking': {}, 'standing': {}}
    #     if max_samples is not None:
    #         keys_w = list(samples['walking'].keys())[:max_samples * h: h]
    #         keys_s = list(samples['standing'].keys())[:max_samples * 2 : 2 ]
    #         for kw in keys_w:
    #             samples_new['walking'][kw] = samples['walking'][kw]
    #         for ks in keys_s:
    #             samples_new['standing'][ks] = samples['standing'][ks]
    #     else:
    #         samples_new = samples

    #     if verbose:
    #         keys_w = list(samples_new['walking'].keys())
    #         keys_s = list(samples_new['standing'].keys())
    #         pid_w = []
    #         pid_s = []
    #         n_w = 0
    #         n_s = 0
    #         for kw in keys_w:
    #             pid_w.append(samples_new['walking'][kw]['old_id'])
    #             n_w += len(samples_new['walking'][kw]['frame'])
    #         for ks in keys_s:
    #             pid_s.append(samples_new['standing'][ks]['old_id'])
    #             n_s += len(samples_new['standing'][ks]['frame'])

    #         print(f"Extract Non-transition samples from {self.name} dataset in TITAN :")
    #         print(f"Walking: {len(pid_w)} samples,  {len(set(pid_w))} unique pedestrians and {n_w} frames.")
    #         print(f"Standing: {len(pid_s)} samples,  {len(set(pid_s))} unique pedestrians and {n_s} frames.")

    #     return samples_new

if __name__ == "__main__":
     
    data_paths = {'TITAN': {'anns': '/media/farzeen/sam/sam/titan_data/dataset/titan_0_4/',
                   'split':'/media/farzeen/sam/sam/titan_data/dataset/splits/' }}
    dataset= TitanTransDataset(anns_dir=data_paths['TITAN']['anns'], split_vids_path=data_paths['TITAN']['split'],image_set='test', verbose=True)
  