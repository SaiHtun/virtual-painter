import cv2
import time
import numpy as np
import HandsTrackingModule as htm


def main():
    c_time = 0
    p_time = 0
    video_cap = cv2.VideoCapture(0)
    video_cap.set(3, 1280)
    video_cap.set(4, 720)
    detector = htm.HandsDetection(max_hands=2, min_detect=0.5, min_track=0.5)
    xp, yp = 0, 0
    canvas = np.zeros((720, 1280, 3), np.uint8)
    paint_color = (128, 128, 0)
    user_manual = "Modes \n- up index finger to draw \n- up index, and middle finger to erase \n- up all fingers to erase everything \n- press 'q' to exit."
    while True:
        success, img = video_cap.read()
        img = cv2.flip(img, 1)
        my_img = detector.find_hands(img)
        lm_list = detector.find_hands_position(my_img)
        # display fps
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(my_img, "fps -" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        initial_y = 50
        for i, text in enumerate(user_manual.split("\n")):
            cv2.putText(my_img, text, (900, initial_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            initial_y += 30

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]
            fingers = detector.fingers_up()
            # 1 finger up ( draw mode )
            if fingers[1] and not fingers[0] and not any(fingers[2:]):
                cv2.circle(my_img, (x1, y1), 15, (135, 206, 235), cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(canvas, (xp, yp), (x1, y1), paint_color, 10)
            # 2 fingers up ( eraser mode )
            elif fingers[1] and fingers[2] and not any(fingers[3:]):
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 20)
            # all fingers up ( remove all the paints )
            elif all(fingers):
                canvas = np.zeros((720, 1280, 3), np.uint8)
            xp, yp = x1, y1
        # extract BGR to gray scale image, so we know the background, and the actual paint.
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        # set the background to white, so our video's image can replace on that white pixel when merging.
        # will output video's image + the actual paint
        # but the paint will still be black whatever colors you choose, thus we need to mask another layer again.
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        # merge the actual color paint on the black paint.
        img = cv2.bitwise_or(img, canvas)
        # show paint board window, destroy the window when press "q" key.
        cv2.imshow("Paint board", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()