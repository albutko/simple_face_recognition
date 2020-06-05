import cv2
import face_recognition.api as fr

from video_stream import VideoStream
from utils import draw_bbox_on_img, load_known_faces

def main():
    vs = VideoStream()
    vs.start()
    names, known_encodings = load_known_faces('./faces/known_faces')
    print(len(known_encodings))
    while vs.isOpened():
        image = vs.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_locations = fr.face_locations(image, model='hog')
        img_face_encodings = fr.face_encodings(image, face_locations)
        match_matrix = [fr.compare_faces(known_encodings, f, tolerance=0.6) for f in img_face_encodings]
        print(match_matrix)
        img_with_faces = draw_bbox_on_img(image, face_locations)

        cv2.imshow('frame', img_with_faces)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
