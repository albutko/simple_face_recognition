import numpy as np
import cv2

from sfd import utils
import sfd.constants as const

def test_draw_bbox_on_img_unalters_image_if_no_bbox():
    img = np.random.randint(0, 255, (300,300))
    processed_image = utils.draw_bbox_on_img(img, [])
    assert np.all(np.isclose(img, processed_image))

def test_draw_bbox_on_img_correctly_draws_bboxs():
    correct_img = np.zeros((300,300,3))
    processed_img = correct_img.copy()

    (known_top, known_right, known_bottom, known_left) = (20, 30, 100, 15)
    (unknown_top, unknown_right, unknown_bottom, unknown_left) = (200, 270, 250, 220)
    labels = ['known', None]

    cv2.rectangle(correct_img, (known_left, known_top), (known_right, known_bottom), const.KNOWN_COLOR)
    cv2.rectangle(correct_img, (unknown_left, unknown_top), (unknown_right, unknown_bottom), const.UNKNOWN_COLOR)
    cv2.putText(correct_img, labels[0], (known_left, known_bottom),
                              fontFace=const.FONT_FACE, fontScale=const.FONT_SCALE, color=const.KNOWN_COLOR)
    cv2.putText(correct_img, 'UNKNOWN', (unknown_left, unknown_bottom),
                              fontFace=const.FONT_FACE, fontScale=const.FONT_SCALE, color=const.UNKNOWN_COLOR)

    bbox_list = [
        (known_top, known_right, known_bottom, known_left),
        (unknown_top, unknown_right, unknown_bottom, unknown_left)
    ]

    processed_img = utils.draw_bbox_on_img(processed_img, bbox_list, labels=labels)
    assert np.all(np.isclose(processed_img, correct_img))

def test_draw_bbox_on_img_draws_bbox_no_labels():

    correct_img = np.zeros((300,300,3))
    processed_img = correct_img.copy()
    (first_top, first_right, first_bottom, first_left) = (20, 30, 100, 15)
    (second_top, second_right, second_bottom, second_left) = (200, 270, 250, 220)

    cv2.rectangle(correct_img, (first_left, first_top), (first_right, first_bottom), const.NOLABELS_COLOR)
    cv2.rectangle(correct_img, (second_left, second_top), (second_right, second_bottom), const.NOLABELS_COLOR)

    bbox_list = [
        (first_top, first_right, first_bottom, first_left),
        (second_top, second_right, second_bottom, second_left)
    ]

    processed_img = utils.draw_bbox_on_img(processed_img, bbox_list)
    assert np.all(np.isclose(processed_img, correct_img))
