
import cv2 as cv
import numpy as np

def image_process(image: np.ndarray) -> np.ndarray:

    raw_img = image.copy()
    rows, cols = image.shape
    blur_ksize = (4, 4) if cols > 2000 else (2, 2)

    # step 1: reduce background noise
    morph_size = 1
    image = cv.blur(image, ksize=blur_ksize)
    image = cv.threshold(image, thresh=140, maxval=255, type=cv.THRESH_BINARY)[1]

    kernel = cv.getStructuringElement(
        shape=cv.MORPH_RECT,
        ksize=(2*morph_size + 1, 2*morph_size + 1),
        anchor=(morph_size, morph_size)
        )
    ## to black background and white foreground
    image = cv.bitwise_not(image)
    image = cv.morphologyEx(image, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)

    # step 2: remove vertical and horizontal lines
    dil_mask_ksize = (6, 2) if cols > 2000 else (3, 2)
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=dil_mask_ksize)
    dilate_mask = cv.morphologyEx(image.copy(), op=cv.MORPH_DILATE, kernel=kernel)

    vert_kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(1, int(rows * 0.03)))
    hori_kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(int(cols * 0.03), 1))

    vert_mask = cv.morphologyEx(dilate_mask.copy(), op=cv.MORPH_OPEN, kernel=vert_kernel, iterations=2)
    vert_mask = cv.dilate(
        vert_mask,
        kernel=cv.getStructuringElement(cv.MORPH_RECT, (3, 1)),
        iterations=2
        )

    hori_mask = cv.morphologyEx(dilate_mask.copy(), op=cv.MORPH_OPEN, kernel=hori_kernel, iterations=2)
    hori_mask = cv.dilate(
        hori_mask,
        kernel=cv.getStructuringElement(cv.MORPH_RECT, (1, 3)),
        iterations=2
        )

    lines_mask = cv.bitwise_or(vert_mask, hori_mask)
    ## to white background and black foreground
    final_mask = cv.bitwise_not(image)
    ## apply mask to remove lines
    final_mask[lines_mask == 255] = 255

    # step 3: recover original characters
    # to black background again
    final_mask = cv.bitwise_not(final_mask)
    final_mask = cv.dilate(
        final_mask,
        kernel=cv.getStructuringElement(cv.MORPH_RECT, (2, 2)), anchor=(0, 0),
        iterations=2
        )

    result_img = cv.bitwise_and(raw_img, final_mask)
    result_img[final_mask == 0] = 255

    return result_img