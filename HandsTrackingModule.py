import cv2
import mediapipe as mp


class HandsDetection:
    def __init__(self, mode=False, max_hands=2, model_comp=1, min_detect=0.5, min_track=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_comp = model_comp
        self.min_detect = min_detect
        self.min_track = min_track
        self.my_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.my_hands.Hands(mode, max_hands, model_comp, min_detect, min_track)
        self.fingerTips = [4, 8, 12, 16, 20]
        self.img_with_hands = []
        self.landmark_list = []

    '''detect hands with mediapipe hand's model, draw the hand's landmarks and return the image '''
    def find_hands(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_with_hands = self.hands.process(image)
        if self.img_with_hands.multi_hand_landmarks:
            for hand_landmarks in self.img_with_hands.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.my_hands.HAND_CONNECTIONS)
        return img

    '''1. find landmarks position on hands
       2. convert the position's decimal value to pixels, because the image only understand the pixels value.
       3. draw circle on the landmarks position
       4. return the landmarks list.
    '''
    def find_hands_position(self, img, no=0, draw=True):
        self.landmark_list = []
        if self.img_with_hands.multi_hand_landmarks:
            my_hand = self.img_with_hands.multi_hand_landmarks[no]
            for num, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append([num, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return self.landmark_list

    '''
    find which fingers are up or down, and store the value in fingers list, as the value of 1 (up) or 0 (down),
    then return the list.
    '''
    def fingers_up(self):
        fingers = []
        # thump
        if self.landmark_list[self.fingerTips[0]][1] < self.landmark_list[self.fingerTips[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # other fingers
        for num in range(1, 5):
            if self.landmark_list[self.fingerTips[num]][2] < self.landmark_list[self.fingerTips[num] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
