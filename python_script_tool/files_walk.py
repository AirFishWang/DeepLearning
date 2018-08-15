# -*- coding: utf-8 -*-
import os


def get_image_list(dir):
    image_list = []
    support_files = [".bmp", ".png", ".jprg", ".jpg"]
    for root, dirs, files in os.walk(dir):
        for f in files:
            fname = os.path.splitext(f)
            if fname[-1].lower() in support_files:
                image_list.append(os.path.join(root, f))
    return image_list


if __name__ == "__main__":
    dir = "impurity"
    image_list = get_image_list(dir)
    for x in image_list:
        print x