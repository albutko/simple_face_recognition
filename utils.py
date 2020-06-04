import cv2

def draw_bbox_on_img(img, bbox_list):
    for (top, right, bottom, left) in bbox_list:
        img = cv2.rectangle(img, (left, top), (right, bottom), (0,255,0))
    return img
