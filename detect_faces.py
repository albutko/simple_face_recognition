import cv2
import face_recognition.api as fr

from utils import draw_bbox_on_img
def main():
    vc = cv2.VideoCapture(0)

    while vc.isOpened():
        retval, image = vc.read()
        if not retval:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_locations = fr.face_locations(gray, model='hog')
        img_with_faces = draw_bbox_on_img(image, face_locations)

        cv2.imshow('frame', img_with_faces)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
