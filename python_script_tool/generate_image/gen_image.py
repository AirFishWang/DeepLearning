# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imgaug
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import random

imgaug.seed(random.randint(0, 100))


class Generator():
    def __init__(self):
        self.fonts_dir = "./fonts"
        # self.fonts_list = ["arial.ttf", "huawenxihei.ttf", "huawensongti.ttf"]          # arial不能再图片上绘制中文
        self.fonts_list = os.listdir(self.fonts_dir)
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

    def gen_word(self):
        pass

    def font_to_image(self, word):
        pass

    def find_text_box(self, img):
        height, width = img.shape[:2]
        img = 255 - img
        v_sum = np.sum(img, axis=0)              # 垂直投影，竖直方向上累加
        h_sum = np.sum(img, axis=1)              # 水平投影，水平方向上累加
        left, right, top, bottom = (0, width - 1, 0, height - 1)
        for i in range(width):
            if v_sum[i] > 0:
                left = i
                break
        for i in range(width-1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break
        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break
        for i in range(height-1, -1, -1):
            if h_sum[i] > 0:
                bottom = i
                break
        # w = right - left + 1
        # h = bottom - top + 1
        return left, right, top, bottom


class GeneratorWuHan(Generator):
    def __init__(self):
        Generator.__init__(self)

    def gen_word(self):
        number = random.randint(10000, 100000)
        word = str(number) + ".00"
        return word

    def font_to_image(self, word):
        font_size = 13
        height = 17
        font = ImageFont.truetype(os.path.join(self.fonts_dir, self.fonts_list[0]), font_size)
        img = Image.new("L", (300, 100), 255)

        draw = ImageDraw.Draw(img)
        draw.text((0, 0), word, fill=0, font=font)
        # img.show()
        left, right, top, bottom = self.find_text_box(np.array(img))
        text_w = right - left + 1

        left_padding = 4
        right_padding = 7
        top_padding = 3
        bottom_padding = 4

        width = left_padding + text_w + right_padding
        img = Image.new('L', (width, height), random.randint(240, 255))
        draw = ImageDraw.Draw(img)
        draw.text((left_padding - left, top_padding - top), word, fill=0, font=font)

        # convert to opencv image format
        img = np.array(img)

        if self.augment:
            img = self.seq.augment_image(img)

        return img


class GeneratorXiaMen(Generator):
    def __init__(self):
        Generator.__init__(self)

    def gen_word(self):
        random_type = random.randint(0, 6)
        if random_type in [0, 1]:
            number = random.randint(100, 1000)
        elif random_type in [2, 3]:
            number = random.randint(10000, 100000)
        elif random_type in [4]:
            number = random.randint(100000, 1000000)
        elif random_type in [5]:
            number = random.randint(1000000, 10000000)
        else:
            number = random.randint(1000000, 100000000)

        # number = random.randint(0, 35000000)
        word = str(number)
        length = len(word)
        stack_list = []
        count = 0
        for i in range(length - 1, -1, -1):
            # stack_list.append(word[i].decode("utf-8"))
            stack_list.append(word[i])
            count += 1
            if count == 3 and i != 0:
                stack_list.append(',')
                count = 0

        word = ''.join(stack_list)[::-1]
        return word

    def font_to_image(self, word):
        xiamen_char = "./xiamen_char"
        image_dicts = {}
        for x in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',']:
            image_dicts[x] = cv2.imread(os.path.join(xiamen_char, x + ".png"))

        length = len(word)

        if length == 0:
            raise Exception("the length of word must not be zero")
        result = image_dicts[word[0]]
        for i in range(1, length):
            result = np.concatenate((result, image_dicts[word[i]]), axis=1)

        h, w = result.shape[:2]
        result = cv2.resize(result, (32*w/h, 32))

        return result


class GeneratorUniversal(Generator):
    def __init__(self):
        Generator.__init__(self)
        self.augment = False
        self.seq = iaa.Sequential([
            # iaa.PerspectiveTransform(scale=0.02, keep_size=True),
            iaa.SomeOf((2, 4), [                  # 每次使用0到3个方式增强
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.Add((-5, 5)),
                iaa.Sometimes(0.5, [
                    iaa.Scale(2.0),
                    iaa.Scale(0.5)
                ]),
                iaa.OneOf([
                    iaa.AverageBlur(k=(1, 3)),
                    iaa.MedianBlur(k=(1, 3)),
                    iaa.GaussianBlur(sigma=iap.Uniform(0.01, 0.05)),
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)
            ])
        ], random_order=True)

    def gen_word(self):
        power = random.randint(0, 10)   # [x, y]
        number = random.randint(10**power, 10**(power+1))

        random_type = random.randint(0, 1)
        if random_type in [0]:
            return str(number)
        elif random_type in [1]:
            word = str(number)
            length = len(word)
            stack_list = []
            count = 0
            for i in range(length - 1, -1, -1):
                stack_list.append(word[i])
                count += 1
                if count == 3 and i != 0:
                    stack_list.append(',')
                    count = 0

            word = ''.join(stack_list)[::-1]
            if random.randint(0, 2) == 0:
                return word
            else:
                return word + "." + str(random.randint(0, 1000))

    def font_to_image(self, word):
        font_size = 28
        height = 32
        font = ImageFont.truetype(os.path.join(self.fonts_dir, random.choice(self.fonts_list)), font_size)
        img = Image.new("L", (300, 100), 255)

        draw = ImageDraw.Draw(img)
        try:
            draw.text((0, 0), word, fill=0, font=font)
        except:
            print word
            exit()
        # img.show()
        left, right, top, bottom = self.find_text_box(np.array(img))
        text_w = right - left + 1

        left_padding = 2
        right_padding = 2
        top_padding = 3
        bottom_padding = 4

        width = left_padding + text_w + right_padding
        background_gray = random.randint(200, 255)
        img = Image.new('L', (width, height), background_gray)
        draw = ImageDraw.Draw(img)
        draw.text((left_padding - left, top_padding - top), word, fill=0, font=font)

        # convert to opencv image format
        img = np.array(img)

        img = self.random_pad_image(img, background_gray)

        if self.augment:
            img = self.seq.augment_image(img)

        return img

    def random_pad_image(self, image, background_gray):
        h, w = image.shape[:2]

        left_pad = random.randint(-3, 10)
        if left_pad < 0:
            image = image[:, (-left_pad):]
        elif left_pad > 0:
            image = np.concatenate((np.ones([h, left_pad], dtype=np.uint8)*background_gray, image), axis=1)
        else:
            pass

        right_pad = random.randint(-3, 10)
        if right_pad < 0:
            image = image[:, :right_pad]
        elif right_pad > 0:
            image = np.concatenate((image, np.ones([h, right_pad], dtype=np.uint8)*background_gray), axis=1)
        else:
            pass

        return image


def gen_sample(num, type, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if type == "wuhan":
        generator = GeneratorWuHan()
    elif type == "xiamen":
        generator = GeneratorXiaMen()
    elif type == "universal":
        generator = GeneratorUniversal()
    else:
        raise Exception("error type")

    label_name = output_dir + ".txt"
    with open(label_name, "w") as fwiter:
        for i in range(num):
            word = generator.gen_word()
            image = generator.font_to_image(word)
            image_name = "{}.jpeg".format(i+1+100000)
            cv2.imwrite(os.path.join(output_dir, image_name), image)
            fwiter.write("{}/{} {}\n".format(output_dir, image_name, word))
            print "generator {} word = {} type = {}".format(image_name, word, type)


if __name__ == "__main__":
    # gen_sample(20, "wuhan", "./wuhan_sample")
    # gen_sample(20, "xiamen", "./xiamen_sample")
    # gen_sample(100000, "universal", "./universal_sample_train")
    # gen_sample(20000, "universal", "./universal_sample_test")

    gen_sample(100000, "universal", "./universal_sample_train2")