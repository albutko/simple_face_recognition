import pytest
import numpy as np
import cv2
from sfd import utils
import sfd.constants as const
from sfd.constants import cv2_colors

color_dict = {"red":0,"green":1,"blue":2}

@pytest.fixture
def color_image():
    return np.zeros((300,300,3))

@pytest.fixture
def make_target_image_bbox(color_image, bbox):
    def _make_target_image(color):
        color_level = color_dict[color]
        final_image = color_image.copy()
        (top, right, bottom, left) = bbox
        final_image[top:bottom+1, left, color_level] = 255
        final_image[top:bottom+1, right, color_level] = 255
        final_image[top, left:right+1, color_level] = 255
        final_image[bottom, left:right+1, color_level] = 255

        return final_image
    return _make_target_image

@pytest.fixture
def make_target_image_label(color_image, bbox):
    def _make_target_image_label(label, color):
        (top, right, bottom, left) = bbox
        processed_img = color_image.copy()
        cv2.putText(processed_img, label, (left, bottom),
                    fontFace=const.FONT_FACE, fontScale=const.FONT_SCALE, color=cv2_colors[color])
        processed_img = cv2.cvtColor(processed_img.astype('float32'), cv2.COLOR_BGR2RGB)
        return processed_img
    return _make_target_image_label

@pytest.fixture
def bbox():
    return (200, 270, 250, 220)

@pytest.mark.parametrize(
    'color',
    [
    "red",
    "green",
    "blue"
    ]
)
def test_draw_bbox(color, color_image, make_target_image_bbox, bbox):
    target_image = make_target_image_bbox(color)
    processed_image = utils.draw_bbox_on_image(color_image, bbox, color)
    np.testing.assert_array_equal(target_image, processed_image)

@pytest.mark.parametrize(
    ['color','label'],
    [
        ("red", 'known'),
        ("green", 'SORTA_KNOWN'),
        ("blue", 'unknown')
    ]
)
def test_add_label_to_image(color, label, color_image, make_target_image_label, bbox):
    target_image = make_target_image_label(label, color)
    processed_image = utils.add_label_to_image(color_image, label, bbox, color=color)
    cv2.imshow("t", target_image)
    cv2.imshow("p",processed_image)
    np.testing.assert_array_equal(target_image, processed_image)

def test_load_image(shared_datadir):
    obama = utils.load_image(shared_datadir / "Barack_Obama.jpg")
    assert type(obama) == np.ndarray

def test_load_known_faces_throws_no_directory():
    with pytest.raises(NotADirectoryError):
        utils.load_known_faces('DIR_DONT_EXIST')

def test_load_known_faces_err_empty_dir(tmp_path):
    with pytest.raises(ValueError):
        utils.load_known_faces(tmp_path)
