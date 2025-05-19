from BoardDetector_YOLO import BoardDetector_YOLO
from BoundedCornerExtractor import BoundedCornerExtractor

bd = BoardDetector_YOLO()
CE = BoundedCornerExtractor()


def detect_board_img(img):
    bd.set_img(img)
    mask, conf = bd.predict()

    img_gray = img.numpy()

    CE.setImg(img_gray, mask)
    CE.apply()

    M, corners, mask_fit, error = CE.fitBoard()

    return M, corners, mask_fit, error
