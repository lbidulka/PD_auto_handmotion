import mediapipe as mp
import cv2
import numpy as np
import scipy.io as sio

mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


def get_mp_handpose(input,
                    save_path,
                    save_side='L',
                    model_complexity=0,
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
                    # _hand = draw_landmarks(image, hand_landmarks, None, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                    _hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
                    # for idx, landmark in enumerate(hand_landmarks.landmark):
                    #     _hand = np.append(_hand, [[landmark.x, landmark.y, landmark.z]], 0)
                    # _hand = np.delete(_hand, 0, 0)

                    if _hand.shape[0] == 21:
                        pos_x = _hand[0][0]
                        _hand = np.expand_dims(_hand, axis=0)
                        if pos_x > split_threshold:
                            hand_left_save = np.append(hand_left_save, _hand, 0)
                        elif pos_x < split_threshold:
                            hand_right_save = np.append(hand_right_save, _hand, 0)
        
            i += 1
        hand_left_save = np.delete(hand_left_save, 0, 0)
        hand_right_save = np.delete(hand_right_save, 0, 0)
        data_dict_left = {'left': hand_left_save}
        data_dict_right = {'right': hand_right_save}

        out_path = save_path
        if 'L' in save_side:
            sio.savemat(out_path + '_left_enhance.mat', data_dict_left)
        if 'R' in save_side:
            sio.savemat(out_path + '_right_enhance.mat', data_dict_right)
        print(f'INFO: Saved hand pose to {out_path}')
    cap.release()