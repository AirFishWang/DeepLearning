# -*- coding: utf-8 -*-
import cv2

def clip(box, size):
    (xmin, ymin, xmax, ymax) = box
    (height, width) = size
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax >= width: xmax = width - 1
    if ymax >= height: ymax = height - 1
    return (xmin, ymin, xmax, ymax)


def crop_v2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = min(img[-1, -1] + 5, 10)
    ret, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    image, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    box = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > max_area:
            max_area = area
            box = x, y, w, h
    if box is None:
        box = (0, 0, img.shape[1] - 1, img.shape[0] - 1)
    else:
        x, y, w, h = box
        box = (x, y, x + w - 1, y + h - 1)
    return clip(box, img.shape)

if __name__ == "__main__":
    img_path = ur"C:/Users/viruser.v-desktop/Desktop/dupeng/src.jpg"
    image = cv2.imread(img_path)
    crop_box = crop_v2(image)
    roi_image = image[crop_box[1]:crop_box[3] + 1, crop_box[0]:crop_box[2] + 1, :]
    cv2.imwrite(ur"C:/Users/viruser.v-desktop/Desktop/dupeng/roi.jpg", roi_image)