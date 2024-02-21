import argparse
import os
import os.path as op

from utils import pose_extraction

def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--data_root', default='./data/PD4T', help='Path to video data')
    parser.add_argument('--UPDRS_task', default='Hand movement', help='Task to process')
    parser.add_argument('--outputFolder', default='./data/PD4T/pose_series', help='output folder for extracted pose data')

    args = parser.parse_args() 
    return args


if __name__ == '__main__':
    args = parse_args()

    vids = {
        # '': [''],
        '003': ['13-005931_l', '13-005931_r'], # UPDRS 0, 1 (angled view)

        '008': ['14-003944_l'], # UPDRS 3, severe slowing and amp dec (fairly angled view)

        '001': ['13-002507_r'], # UPDRS 0, fast (decent view)
        
        '036': ['14-005721_l'], # UPDRS 2, slow (good view)
        }
    
    for subj_id in vids.keys():
        for vid in vids[subj_id]:
            vid_path = os.path.join(args.data_root, 'Videos', args.UPDRS_task, subj_id, vid+'.mp4')
            handedness = vid.split('_')[-1].upper()
            save_path = os.path.join(args.outputFolder, ('_').join([vid, subj_id]))
            print(f'Processing {vid_path} to {save_path}')

            if op.exists(vid_path):
                pose_extraction.get_mp_handpose(vid_path, save_path=save_path, save_side=handedness)
            else:
                print(f'ERR: input file {vid_path} does not exist')