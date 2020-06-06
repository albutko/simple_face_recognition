import cv2

import sfd.constants as const

def draw_bbox_on_img(img, bbox_list, labels=None):
    img_to_process = img.copy()
    if labels:
        assert len(bbox_list) == len(labels)
        for (top, right, bottom, left), label in zip(bbox_list, labels):
            color = const.KNOWN_COLOR if label else const.UNKNOWN_COLOR
            text = label if label else 'UNKNOWN'
            cv2.rectangle(img_to_process, (left, top), (right, bottom), color)
            cv2.putText(img_to_process, text, (left, bottom), fontFace=const.FONT_FACE, fontScale=const.FONT_SCALE, color=color)
    else:
        for (top, right, bottom, left) in bbox_list:
            cv2.rectangle(img_to_process, (left, top), (right, bottom), const.NOLABELS_COLOR)

    return img_to_process

def load_known_faces(directory):
    img_files = glob.glob(f'{directory}/*')
    face_imgs = [fr.load_image_file(f) for f in img_files]
    names = [f.split("/")[-1] for f in img_files]
    encodings = [fr.face_encodings(img)[0] for img in face_imgs]
    return (names, encodings)
