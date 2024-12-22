import copy
import math
import time

import cv2 as cv
import numpy as np

lower_orange = np.array([0, 150, 100])
upper_orange = np.array([20, 300, 350])
orange_threshhold = 0.1

lower_yellow = np.array([30, 180, 104])
upper_yellow = np.array([50, 255, 220])
yellow_threshhold = 0.1

cam = cv.VideoCapture(0, cv.CAP_DSHOW)

FOV_X = 0
FOV_Y = 0
HEIGHT = 480
WIDTH = 640
FOCAL_X = 687.97583877#(WIDTH / (math.tan(0.3841161543769188) * 2))
FOCAL_Y = 723.82528471#(HEIGHT / (math.tan(0.24892437646661184) * 2))
NOTE_RADIUS = 0.3556 / 2.0
BALL_RADIUS = 0.1778 / 2.0

cam.set(cv.CAP_PROP_FRAME_HEIGHT,HEIGHT)
cam.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)


def pipeline(binary_frame, erode_iterations, dilate_iterations):
    # apply erosion on frame
    binary_frame = cv.erode(binary_frame, np.ones((5, 5), np.uint8), erode_iterations)
    # apply dilation on frame
    binary_frame = cv.dilate(binary_frame, np.ones((5, 5), np.uint8), dilate_iterations)
    # apply distance transform
    transformed_frame = cv.distanceTransform(binary_frame, cv.DIST_L2, 5)
    return transformed_frame

def main():

    cam.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cam.set(cv.CAP_PROP_FPS, 60)
    cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
    is_still, _ = cam.read()
    note = 0

    while True:
        is_still, originalFrame = cam.read()
        frame = copy.deepcopy(originalFrame)


        frame_in_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)

        in_range_frame = cv.inRange(frame_in_hsv, lower_orange, upper_orange)

        piplined_hsv_frame = pipeline(in_range_frame, 0, 0)

        final_frame = cv.inRange(piplined_hsv_frame, orange_threshhold, 1)

        cnt, _ = cv.findContours(final_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if len(cnt) != 0:
            note = cnt[0]
            for i in range(0, len(cnt)):
                if cv.contourArea(note) < cv.contourArea(cnt[i]):
                    note = cnt[i]

            center, radius = cv.minEnclosingCircle(note)
            cv.circle(final_frame, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 5)
            cv.circle(originalFrame, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 5)

            center, radius = cv.minEnclosingCircle(note)
            z = (NOTE_RADIUS * FOCAL_X) / radius
            x = (center[0] * z) / FOCAL_X
            y = (center[1] * z) / FOCAL_Y

            distanceToNote = (x**2 + y**2 + z**2)**0.5
            two_d_angle_to_note = math.degrees(math.atan(x/z))
            print("Distance to note: " + str(distanceToNote))
            print(two_d_angle_to_note)

        cv.imshow('original video', originalFrame)
        cv.imshow('proccesed video', final_frame)
        key = chr(cv.waitKey(1) % 256)
        if key == 'q':
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
