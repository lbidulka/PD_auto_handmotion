import io
import pandas as pd


UPDRS_labels_PD4T_path = './data/PD4T/Annotations/Hand movement/'
PD4T_train_df = pd.read_csv(UPDRS_labels_PD4T_path + 'train.csv', header=None)
PD4T_test_df = pd.read_csv(UPDRS_labels_PD4T_path + 'test.csv', header=None)

# rename and split annots
PD4T_train_df['patient_id'] = PD4T_train_df[0].str[-3:]
PD4T_train_df['handedness'] = PD4T_train_df[0].str[-5:-4]
PD4T_train_df['visit'] = PD4T_train_df[0].str[:9]
PD4T_train_df = PD4T_train_df.rename(columns={1: 'frame_count', 2: 'UPDRS_score'})
# add column of 1s for train
PD4T_train_df['train'] = 1

PD4T_test_df['patient_id'] = PD4T_test_df[0].str[-3:]
PD4T_test_df['handedness'] = PD4T_test_df[0].str[-5:-4]
PD4T_test_df['visit'] = PD4T_test_df[0].str[:9]
PD4T_test_df = PD4T_test_df.rename(columns={1: 'frame_count', 2: 'UPDRS_score'})
# add column of 0s for test
PD4T_test_df['train'] = 0

# merge train and test
PD4T_handmotion_df = pd.concat([PD4T_train_df, PD4T_test_df])

