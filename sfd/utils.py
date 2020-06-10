import os
import glob
import PIL.Image

import cv2
import face_recognition.api as fr
import sfd.constants as const
from sfd.constants import cv2_colors

def draw_bbox_on_image(img, bbox, color="blue"):
    new_image = cv2.cvtColor(img.copy().astype('float32'), cv2.COLOR_RGB2BGR)
    (top, right, bottom, left) = bbox
    cv2.rectangle(new_image, (left, top), (right, bottom), cv2_colors[color])
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image

def add_label_to_image(img, label, bbox, color="blue"):
    new_image = cv2.cvtColor(img.copy().astype('float32'), cv2.COLOR_RGB2BGR)
    (_, _, bottom, left) = bbox
    cv2.putText(new_image, label, (left,bottom), fontFace=const.FONT_FACE, fontScale=const.FONT_SCALE, color=cv2_colors[color])
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image

def load_known_faces(directory):
    if not os.path.exists(directory):
        raise NotADirectoryError(f"{directory} is not a directory")

    img_files = glob.glob(f'{directory}/*')
    face_imgs = [_load_image_file(f) for f in img_files]
    if len(face_imgs) == 0:
        raise ValueError("No images in directory")

    names = [f.split("/")[-1] for f in img_files]
    encodings = [fr.face_encodings(img)[0] for img in face_imgs]
    if len(encodings) == 0:
        raise ValueError("No faces found in images from directory")

    return (names, encodings)

def _load_image_file(image_path):
    img = PIL.Image.open(image_path)
    img = img.convert('RGB')
    return img
