'''
    This project was done by the team Levitate(#31)! Its purpose was to detect lane lines from the video provided!
    To hande this task we used OpenCV and numpy libraries.
    The program is divided into N steps

    1. We detected region of interest
    2. Convert Image to Grayscale
    3. Apply Canny Edge Detection
    4. Apply Hough Lines Transform
    5. Draw lines on the image and merge them

    The result is shown in the output video!

'''

import numpy as np
import cv2
import matplotlib.pylab as plt

# Crop image according to the ROI vertices
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Draw lines on the blank image and merge them with original image
def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2,y2), (0,255,0), 3)
        # Merge images
        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img



# Process video frames and filter each one
# TODO: If video is not supperted check for the correctnes of video path

cap = cv2.VideoCapture('assets/test_video.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (960, 540))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize the ROI vertices
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (250, height),
        (width / 2.2, height / 1.4),
        (width / 1.7, height / 1.4),
        (width / 1.2, height)
    ]
    #
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 80, 120)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32), )
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180,
                            threshold = 50,
                            lines = np.array([]),
                            minLineLength= 40,
                            maxLineGap =100)

    image_with_lines = draw_the_lines(frame, lines)
    cv2.imshow('With lane lines',image_with_lines)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
