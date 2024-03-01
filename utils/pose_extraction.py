import mediapipe as mp
import cv2
import numpy as np
import scipy.io as sio

mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


def get_mp_handpose(input,
                    save_path,
                    save_side='L',
                    filter_angle=5,#1.9,   #np.pi/2,
                    model_complexity=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.8,
                    max_num_hands=1,
                    split_threshold=0.5,) -> None:
    '''
    Extract hand pose from video using mediapipe
    '''

    cap = cv2.VideoCapture(input)
    with mp_hands.Hands(model_complexity=model_complexity, 
                        min_detection_confidence=min_detection_confidence, 
                        min_tracking_confidence=min_tracking_confidence, 
                        static_image_mode=False, 
                        max_num_hands=max_num_hands) as hands:
        hand_left_save = np.empty((1, 21, 3))
        hand_right_save = np.empty((1, 21, 3))
        i = 0
        angles = []
        while cap.isOpened():
            success, image = cap.read() 
            if not success:
                print("File {} finished.".format(input))
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    _hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
                    # compute angle of hand to camera, if its tilted too forward, we wont use it
                    palm_vector = (_hand[0,:] + _hand[5,:] + _hand[17,:]) / 3
                    palm_vector -= _hand[0,:]
                    palm_vector = palm_vector / np.linalg.norm(palm_vector, axis=0).reshape(-1,1)
                    palm_vector = palm_vector[0]
                    angle = np.arccos(palm_vector[2])
                    angles.append(angle)
                    
                    if (_hand.shape[0] == 21):
                        # thumb avg x val
                        thumb_x = (_hand[4,0] + _hand[3,0] + _hand[2,0] + _hand[1,0]) / 4
                        root_x = _hand[0,0]
                        _hand = np.expand_dims(_hand, axis=0)
                        if thumb_x < root_x:
                            hand_left_save = np.append(hand_left_save, _hand, 0)
                        elif thumb_x > root_x:
                            hand_right_save = np.append(hand_right_save, _hand, 0)
        
            i += 1
        hand_left_save = np.delete(hand_left_save, 0, 0)
        hand_right_save = np.delete(hand_right_save, 0, 0)
        data_dict_left = {'left': hand_left_save}
        data_dict_right = {'right': hand_right_save}

        out_path = save_path

    cap.release()

    # check if we want to save
    bad_angles_thresh = 0.25
    bad_angle_flags = np.array(angles) > filter_angle
    print("avg angle: ", np.array(angles).mean())
    if bad_angle_flags.mean() > bad_angles_thresh:
        print(f'--> WARN: Too many bad angles, not saving {out_path}')
        return 0
    else:
        if 'L' in save_side:
            sio.savemat(out_path + '_left_enhance.mat', data_dict_left)
        if 'R' in save_side:
            sio.savemat(out_path + '_right_enhance.mat', data_dict_right)
        print(f'INFO: Saved hand pose to {out_path}')
        return 1
    