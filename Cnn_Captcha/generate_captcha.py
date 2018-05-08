from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import random
from string import digits, ascii_letters, punctuation

TEMPLATE = digits + ascii_letters
TOTAL_NUM = len(TEMPLATE)


def generate_random_text(num=4):
    return ''.join((random.choice(TEMPLATE) for _ in range(num)))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def generate_captcha(num=4, size=(160, 60)):
    width, height = size
    captcha = ImageCaptcha(width=width, height=height)  # (160, 60, 3)
    text = generate_random_text(num)
    captcha_image = captcha.generate(text)
    img = Image.open(captcha_image)  # (160, 60, 1)
    img_array = np.array(img)
    return img_array, text


def generate_gray_captcha(num=4, size=(160, 60)):
    img_array, text = generate_captcha(num, size)
    img_array = rgb2gray(img_array) / 255.
    return img_array, text


if __name__ == "__main__":
    img, text = generate_captcha()
    print(text)
    print(img.shape)
