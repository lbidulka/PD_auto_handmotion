import argparse
import os
import os.path as op
from tqdm import tqdm

from utils import pose_extraction
from data import CAMERA_expert_labels

def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--dataset', default='CAMERA', help='Dataset to process')   # CAMERA, PD4T
    parser.add_argument('--data_root', default='./data/', help='Path to video data')
    parser.add_argument('--UPDRS_task', default='Hand movement', help='Task to process')
    # parser.add_argument('--outputFolder', default='./data/PD4T/pose_series', help='output folder for extracted pose data')

    args = parser.parse_args() 
    return args

def extract_CAMERA(args):    
    PD_action = 'hand_movement'
    # PD_action_task = 'right_open_close' #'right_open_close', left_open_close
    debug_imgs = False      # print out pose frames (will make a lot of files)
    # handedness = 'L' if PD_action_task == 'left_open_close' else 'R'
    
    data = CAMERA_expert_labels.UPDRS_med_data_SA

    successes = []
    # for id, date in tqdm(data[PD_action][PD_action_task].items()):
    for id in tqdm(data):
        for PD_action_task in data[id][PD_action]:
            handedness = 'L' if PD_action_task == 'left_open_close' else 'R'
            date = list(data[id][PD_action][PD_action_task].keys())[0]
            file = op.join(id, date, PD_action, f'{PD_action_task}.mp4')
            input = '/mnt/teamshare-camera/CAMERA Booth Data/Booth/'

            # f_num = f'{PD_action}/{id}_{date}'

            # # make dir for output if needed
            # output = op.join('features/mat_booth', f'{PD_action}') 
            # if not op.exists(output):
            #     os.makedirs(output)

            # # assert that input file exists
            # in_file = op.join(input, file)
            # if op.exists(in_file):
            #     backbone.get_traj(in_file, f_num, debug_pose=debug_imgs, save_side=hand)
            # else:
            #     print(f'ERR: input file {in_file} does not exist')

            vid_path = op.join(input, file)
            save_path = os.path.join(args.data_root, args.dataset, 'pose_series', ('_').join([id, date, PD_action_task,]))

            print(f'Processing {vid_path} to {save_path}')

            if op.exists(vid_path):
                extract_result = pose_extraction.get_mp_handpose(vid_path, save_path=save_path, save_side=handedness)
                successes.append(extract_result)
            else:
                print(f'ERR: input file {vid_path} does not exist')
    num_successes = len([s for s in successes if s])
    print(f'Kept {num_successes} videos, rejected {len(successes) - num_successes} videos')

def extract_PD4T(args):
    # vids = {
        # '': [''],
        # '003': ['13-005931_l', '13-005931_r'], # UPDRS 0, 1 (~perp)
        # '004': ['12-105184_l', '13-001273_l', '13-003778_l'], # (~25deg)
        
        # '015': ['13-007582_l', '13-007582_r'],  # ~35deg, ~35deg

        # '005': ['12-105480_l', '12-105480_r', '12-105481_l', '12-105481_r'],   # ~45deg, ~45deg, ~25deg, ~25deg

        # '008': ['14-003944_l'], # UPDRS 3, severe slowing and amp dec (far angled view)

        # '019': ['13-007680_l', '13-007680_r', # perp, perp
        #         '14-004782_l', '14-004782_r', # ~75deg, ~70deg
        #         '14-005778_l', '14-005778_r', # ~90deg, ~85deg
                # ], 
        # '024': ['15-004876_l', '15-004876_r',], # ~90deg, ~80deg
        # '029': ['14-003787_l', '14-003787_r',],  # ~85deg, ~80deg
        
        # '022': ['14-003489_l', '14-003489_r'],  # ~75deg, ~75deg
        # '038': ['14-005767_l', '14-005767_r',   # ~80deg, ~80deg
        #         '14-005772_l', '14-005772_r',   # ~65deg, ~65deg
        #         ],
        # '': [],
        # '': [],
        

        # '011': ['13-006817_l', '13-006817_r'],  # (l: angled view, r: good view, slight angles)

        # '001': ['13-002507_r'], # UPDRS 0, fast (decent view)
        
        # '036': ['14-005721_l'], # UPDRS 2, slow (good view)
        # }
    
    # get all video names from the data root
    subj_ids = os.listdir(os.path.join(args.data_root, 'Videos', args.UPDRS_task))
    subj_ids = [subj_id for subj_id in subj_ids if '.' not in subj_id]
    vids = {}
    for subj_id in subj_ids:
        vid_path = os.listdir(os.path.join(args.data_root, 'Videos', args.UPDRS_task, subj_id))
        vids[subj_id] = [vid[:-4] for vid in vid_path if '.mp4' in vid]
    
    # trim vids to first 10 keys
    # vids = {k: vids[k] for k in list(vids.keys())[:1]}

    successes = []
    for subj_id in tqdm(vids.keys()):
        for vid in vids[subj_id]:
            vid_path = os.path.join(args.data_root, args.dataset, 'Videos', args.UPDRS_task, subj_id, vid+'.mp4')
            handedness = vid.split('_')[-1].upper()
            save_path = os.path.join(args.data_root, args.dataset, 'pose_series', ('_').join([vid, subj_id]))
            print(f'Processing {vid_path} to {save_path}')

            if op.exists(vid_path):
                extract_result = pose_extraction.get_mp_handpose(vid_path, save_path=save_path, save_side=handedness)
                successes.append(extract_result)
            else:
                print(f'ERR: input file {vid_path} does not exist')
    num_successes = len([s for s in successes if s])
    print(f'Kept {num_successes} videos, rejected {len(successes) - num_successes} videos')

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'CAMERA':
        extract_CAMERA(args)
    elif args.dataset == 'PD4T':
        extract_PD4T(args)
    else:
        print(f'ERR: dataset {args.dataset} not recognized')
    
    