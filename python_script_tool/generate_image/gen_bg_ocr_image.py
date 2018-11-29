# -*- coding: utf-8 -*-
import os
import cv2
import io
import random
from math import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imgaug
from imgaug import augmenters as iaa
from imgaug import parameters as iap

image_file = "../../data/1.png"
corpus_path = "./data/corpus.txt"
synthetic_train_label = "./data/data_train.txt"
synthetic_test_label = "./data/data_test.txt"
dict_path = "./data/char_std_5990.txt"
fonts_dir = "./fonts/chinese_fonts"
bg_dir = "./data/bg_dir"

WIDTH = 280
HEIGHT = 32

"""
supported chinese fonts:
FZSTK.TTF
FZYTK.TTF
huawensongti.ttf
huawenxihei.ttf
MSYH.TTC
msyh.ttf
MSYHBD.TTC
msyhbd.ttf
simfang.ttf
simhei.ttf
simkai.ttf
SIMLI.TTF
simsun.ttc
SIMYOU.TTF
STFANGSO.TTF
STKAITI.TTF
STLITI.TTF
STSONG.TTF
STXIHEI.TTF
STXINWEI.TTF
STZHONGS.TTF
"""

class Generator():
    def __init__(self, dict_path, fonts_dir, bg_dir):
        self.dict_path, self.fonts_dir, self.bg_dir = dict_path, fonts_dir, bg_dir
        self.char_set, self.char_to_label, self.label_to_char = self.load_dict(dict_path)
        self.fonts_list = self.load_fonts(fonts_dir)
        self.bg_image_list = self.load_background_image(bg_dir)

        self.augment = False
        self.seq = iaa.Sequential([
            # iaa.PerspectiveTransform(scale=0.02, keep_size=True),
            iaa.SomeOf((0, 3), [
                iaa.GaussianBlur(sigma=iap.Uniform(0, 0.01)),
                iaa.Add((-20, 20)),
                iaa.OneOf([
                    iaa.AverageBlur(k=(0, 1)),
                    iaa.MedianBlur(k=(1, 3))
                ])
            ])
        ], random_order=True)

    def load_dict(self, dict_path):
        char_set = io.open(dict_path, 'r', encoding='utf-8').readlines()
        char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + [u'卍'])  # 字 == char_set[label-1]
        char_to_label = {}
        label_to_char = {}
        for index, c in enumerate(char_set):
            char_to_label[c] = index + 1
            label_to_char[index + 1] = c
        # for x in [124, 262, 649, 176, 80, 19, 46, 6, 187, 2]:
        #     print label_to_char[x]

        return char_set, char_to_label, label_to_char

    def load_fonts(self, fonts_dir):
        lists = os.listdir(fonts_dir)
        return [os.path.join(fonts_dir, x) for x in lists]

    def load_background_image(self, bg_dir):
        # gamma tranform, enhanced brightness
        gamma = 0.3
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

        res = []
        image_names = os.listdir(bg_dir)
        for x in image_names:
            img = cv2.imread(os.path.join(bg_dir, x), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            if w > WIDTH and h > HEIGHT:
                img = cv2.LUT(img, gamma_table)
                res.append(img)
        return res

    def gen_word_from_corpus(self, corpus):
        while True:
            tmp = []
            with open(corpus, 'r') as f:
                for line in f:  # for line in f文件对象f视为一个迭代器，会自动的采用缓冲IO和内存管理，可处理大文件。
                    line = line.decode("utf-8")
                    length = len(line)
                    i = 0
                    while i < length:
                        if len(tmp) == 10:
                            word = ''.join(tmp)
                            label = [self.char_to_label[x] for x in tmp]
                            tmp = []
                            i += random.randint(0, 9)
                            yield word, label
                        else:
                            c = line[i]
                            if c in self.char_set:
                                tmp.append(c)
                            i += 1

    def gen_word_from_synthetic_label(self, label_file):
        print "load {}...".format(label_file)
        res = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                res.append(line.strip().split(' ')[1:])
        print "load {} finished".format(label_file)
        index_list = range(len(res))
        while True:
            random.shuffle(index_list)
            for i in index_list:
                label = res[i]
                word = ''.join([self.label_to_char[int(x)] for x in label])
                yield word, label

    def word_to_image(self, word, font=None):
        font_size = 28
        if font is None:
            font = random.choice(self.fonts_list)
        font = ImageFont.truetype(font, font_size)

        bg_image = random.choice(self.bg_image_list)
        h, w = bg_image.shape[:2]
        bg_canvas_image = Image.fromarray(bg_image)
        white_canvas_img = Image.new("L", (w, h), 255)
        draw_point_x = random.randint(10, w - WIDTH)
        draw_point_y = random.randint(10, h - HEIGHT)

        draw_handler = ImageDraw.Draw(white_canvas_img)
        draw_handler.text((draw_point_x, draw_point_y), word, fill=0, font=font)
        draw_handler = ImageDraw.Draw(bg_canvas_image)
        draw_handler.text((draw_point_x, draw_point_y), word, fill=random.randint(0, 50), font=font)

        white_canvas_img = np.asarray(white_canvas_img)
        bg_canvas_image = np.asarray(bg_canvas_image)

        # cv2.imshow("white_canvas_img", white_canvas_img)
        # cv2.imshow("bg_canvas_image", bg_canvas_image)
        # cv2.waitKey(0)

        synthetic_image = self.affine_and_crop(white_canvas_img, bg_canvas_image)
        synthetic_image = self.seq.augment_image(synthetic_image)
        return synthetic_image

    def find_text_box(self, img):
        height, width = img.shape[:2]
        img = 255 - img
        v_sum = np.sum(img, axis=0)  # 垂直投影，竖直方向上累加
        h_sum = np.sum(img, axis=1)  # 水平投影，水平方向上累加
        left, right, top, bottom = (0, width - 1, 0, height - 1)
        for i in range(width):
            if v_sum[i] > 0:
                left = i
                break
        for i in range(width - 1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break
        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break
        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                bottom = i
                break

        return left, right, top, bottom

    def affine_and_crop(self, white_canvas_img, bg_canvas_image):
        """
        this function would tilt and rotate word randomly
        :param src_image:
        :return:
        """
        left, right, top, bottom = self.find_text_box(white_canvas_img)
        if random.randint(0, 3) == 0:
            h, w = white_canvas_img.shape[:2]
            delta = random.randint(3, 8)
            points1 = np.float32([[left, top], [right, top], [left, bottom]])
            points2 = np.float32([[left+delta, top], [right+delta, top], [left, bottom]])
            M = cv2.getAffineTransform(points1, points2)
            bg_canvas_image = cv2.warpAffine(bg_canvas_image, M, (w, h), borderValue=(255))
            crop_img = bg_canvas_image[top:bottom, left:max(left + WIDTH, right+delta)]
        else:   # if have affine transform, would not rotate image
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            if random.randint(0, 3) == 0:
                h, w = white_canvas_img.shape[:2]
                angle = random.randint(-3, 3)
                M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])

                # compute the new bounding dimensions of the image
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))

                # adjust the rotation matrix to take into account translation
                M[0, 2] += (nW / 2) - center_x
                M[1, 2] += (nH / 2) - center_y
                white_canvas_img = cv2.warpAffine(white_canvas_img, M, (nW, nH), borderValue=(255))
                bg_canvas_image = cv2.warpAffine(bg_canvas_image, M, (nW, nH), borderValue=(255))
                left, right, top, bottom = self.find_text_box(white_canvas_img)
                crop_img = bg_canvas_image[top:bottom, left:max(left+WIDTH, right)]
            else:
                crop_img = bg_canvas_image[top:bottom, left:max(left + WIDTH, right)]
        crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
        return crop_img


def gen_synthetic_image(generator, word_generator, output_dir, count, batch_szie, label_file):
    """
    :param generator: a Generator object convert word to image
    :param word_generator: a word_generator can yield (word, label)
    :param output_dir:
    :param count:
    :param batch_szie:
    :param label_file:
    :return:
    """
    with open(label_file, 'w') as f:
        for index in range(1, count):
            dir_id = str(index / batch_szie + 1)
            dir_path = os.path.join(output_dir, dir_id)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            word, label = word_generator.next()
            image = generator.word_to_image(word)
            label_string = ' '.join([str(x) for x in label])
            image_name = "{:07d}.png".format(index)
            cv2.imwrite(os.path.join(dir_path, image_name), image)
            label_line = r"{}/{} {}".format(dir_id, image_name, label_string)
            f.write("{}\n".format(label_line))
            print r"{}/{} {} {}".format(index, count, word.encode("utf-8"), label_line)
    print "All finished"


def word_to_image_test():
    output_dir = "./word_to_image"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generator = Generator(dict_path, fonts_dir, bg_dir)

    # # test one image
    # font_path = os.path.join(fonts_dir, "huawensongti.ttf")
    # img = generator.word_to_image(u"醉里挑灯看剑，梦回吹角连营。", font=font_path)
    # img = generator.word_to_image(u"醉里挑灯看剑梦回吹角", font=font_path)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # exit()

    # test batch
    for index, font_path in enumerate(generator.fonts_list):
        font_name = os.path.split(font_path)[1]
        print "font = {}".format(font_name)
        img = generator.word_to_image(u"醉里挑灯看剑梦回吹角", font=font_path)
        cv2.imwrite(os.path.join(output_dir, font_name+".png"), img)

        # cv2.imshow("img", img)
        # cv2.waitKey(0)

if __name__ == "__main__":

    generator = Generator(dict_path, fonts_dir, bg_dir)
    word_generator = generator.gen_word_from_synthetic_label(synthetic_test_label)
    gen_synthetic_image(generator, word_generator, "./1000", 1000, 100, "1000_test.txt")

    # word_generator = generator.gen_word_from_corpus(corpus_path)
    # for i in range(100):
    #     word, label = word_generator.next()
    #     print word.encode("utf-8"), label

    # word_to_image_test()
    pass