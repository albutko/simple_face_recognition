import cv2

def draw_bbox_on_img(img, bbox_list):
    for (top, right, bottom, left) in bbox_list:
        img = cv2.rectangle(img, (left, top), (right, bottom), (0,255,0))
    return img

def load_known_faces(directory):
    img_files = glob.glob(f'{directory}/*')
    face_imgs = [fr.load_image_file(f) for f in img_files]
    names = [f.split("/")[-1] for f in img_files]
    encodings = [fr.face_encodings(img)[0] for img in face_imgs]
    return (names, encodings)
