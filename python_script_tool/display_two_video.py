# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np

def concatenate_image():
    image = cv2.imread("yi.jpeg")
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    image = cv2.putText(image, '1', (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    gray_img = cv2.putText(gray_img, '2', (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    result = np.concatenate((image, gray_img), axis=0)

    result = cv2.resize(result, (int(result.shape[1] / 3), int(result.shape[0] / 3)))
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.imwrite("result.jpg", result)


def display_two_video(video1, video2, output = None):
    fourcc, fps, writer_handle = None, None, None
    if output is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 25
        writer_handle = cv2.VideoWriter(output, fourcc, fps, (int(1920 / 2.5), int(1080*2 / 2.5)))

    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == False or ret2 == False:
            break

        frame1 = cv2.putText(frame1, 'video1', (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        frame2 = cv2.putText(frame2, 'video2', (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        frame = np.concatenate((frame1, frame2), axis=0)
        frame = cv2.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
        if output is not None:
            writer_handle.write(frame)

        cv2.imshow("video", frame)

        k = cv2.waitKey(20)
        if k == ord('q'):
            break

    cap1.release()
    cap2.release()
    writer_handle.release()


def display_two_video_batch(dir1, dir2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, 11):
        video1 = os.path.join(dir1, "{}.mp4".format(i))
        video2 = os.path.join(dir2, "{}.mp4".format(i))
        output = os.path.join(output_dir, "{}.mp4".format(i))
        display_two_video(video1, video2, output)
        print "combined {}.mp4".format(i)


def read_write_video():
    src_video = "10.mp4"
    dst_video = "./10_copy.mp4"
    cap = cv2.VideoCapture(src_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 25
    writer_handle = cv2.VideoWriter(dst_video, fourcc, fps, (1920, 1080))
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("video", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        writer_handle.write(frame)

    cap.release()
    writer_handle.release()


if __name__ == "__main__":
    # concatenate_image()
    # video1 = r"/home/wangchun/Desktop/retinanet_video_output_infer/1.mp4"
    # video2 = r"/media/wangchun/新加卷/wangchun/retinanet_video_output/1.mp4"

    video1 = r"./10.mp4"
    video2 = r"./10.mp4"
    output = r"./10_combine.mp4"
    display_two_video(video1, video2, output)

    # read_write_video()