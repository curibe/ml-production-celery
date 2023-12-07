from io import BytesIO
from PIL import Image


def from_image_to_bytes(img):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="png")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def from_bytes_to_image(img_bytes):
    return Image.open(BytesIO(img_bytes))
