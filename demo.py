import torch
from face_detector import *
from face_landmark import *


def camera_run():
    face_detector_handle = FaceDetector()
    face_landmark_handle = FaceLandmark()

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if image is None:
            continue
        detections, _ = face_detector_handle.run(image)

        if len(detections) == 0:
            continue
        for detection in detections:
            landmarks, states = face_landmark_handle.run(image, detection)
            if landmarks is None:
                continue
            face_landmark_handle.show_result(image, landmarks)

# import argparse
# import cv2
# import face_detector, face_landmark  # 请替换为实际的模块名
# # def image_run():
# #     face_detector_handle = FaceDetector()
# #     face_landmark_handle = FaceLandmark()

# #     image = cv2.imread('data/20230719091829.jpg')
# #     detections, _ = face_detector_handle.run(image)

# #     #face_detector_handle.show_result(image, detections)

# #     if len(detections) == 0:
# #         return

# #     for detection in detections:
# #         landmarks, states = face_landmark_handle.run(image, detection)
# #         face_landmark_handle.show_result(image, landmarks)
# def image_run(image_path):  
#     face_detector_handle = FaceDetector()
#     face_landmark_handle = FaceLandmark()
#     image = cv2.imread(image_path)  
#     detections, _ = face_detector_handle.run(image)
    
#     if len(detections) == 0:
#         return

#     for detection in detections:
#         landmarks, states = face_landmark_handle.run(image, detection)
#         Result =face_landmark_handle.show_result(image, landmarks)
#         # cv2.namedWindow('Result', cv2.WINDOW_NORMAL)  
#         # cv2.imshow('Result', image)

#     # try:
#     #     cv2.waitKey(0)  # 等待用户按下一个键
#     # except KeyboardInterrupt:
#     #     print("Interrupted by user")  # 用户按下ctrl+c时的处理
#     #     cv2.destroyAllWindows()  # 关闭所有窗口

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run face landmark detection on an image.')
#     parser.add_argument('--img_path', help='Path to the image file.')
#     args = parser.parse_args()
#     image_run(args.img_path)
# # if __name__ == '__main__':
# #     image_run()
# #     #ctrl + c 终止程序
# #     #如果不终止，就一直显示图像
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


import argparse
import cv2
import face_detector, face_landmark # 请替换为实际的模块名
# import threading

# def show_image(image):
#     cv2.namedWindow('Result', cv2.WINDOW_NORMAL)  
#     cv2.imshow('Result', image)
#     cv2.waitKey(0)

# def image_run(image_path):  
#     face_detector_handle = FaceDetector()
#     face_landmark_handle = FaceLandmark()

#     image = cv2.imread(image_path)  
#     detections, _ = face_detector_handle.run(image)

#     if len(detections) == 0:
#         return

#     for detection in detections:
#         landmarks, states = face_landmark_handle.run(image, detection)
    
#     # Start a new thread to display the image
#     t = threading.Thread(target=show_image, args=(image,))
#     t.start()

#     try:
#         while True:
#             pass
#     except KeyboardInterrupt:
#         print("Interrupted by user")  # 用户按下ctrl+c时的处理
#         cv2.destroyAllWindows()  # 关闭所有窗口
#         t.join()  # 等待显示图片的线程结束

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run face landmark detection on an image.')
#     parser.add_argument('--img_path', help='Path to the image file.')
#     args = parser.parse_args()
#     image_run(args.img_path)
# def image_run(image_path):  
#     face_detector_handle = FaceDetector()
#     face_landmark_handle = FaceLandmark()

#     image = cv2.imread(image_path)  
#     detections, _ = face_detector_handle.run(image)

#     if len(detections) == 0:
#         return

#     for detection in detections:
#         landmarks, states = face_landmark_handle.run(image, detection)
#         #landmarks, states = face_landmark_handle.run(image, detection)
#         face_landmark_handle.show_result(image, landmarks)

#     cv2.namedWindow('Result', cv2.WINDOW_NORMAL)  

#     while True:
#         cv2.imshow('Result', image)  
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # If 'q' key is pressed then break
#             break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run face landmark detection on an image.')
#     parser.add_argument('--img_path', help='Path to the image file.')
#     args = parser.parse_args()
#     image_run(args.img_path)
import threading




def image_run(image_path):  
    face_detector_handle = FaceDetector()
    face_landmark_handle = FaceLandmark()

    image = cv2.imread(image_path)  
    detections, _ = face_detector_handle.run(image)

    if len(detections) == 0:
        return

    for detection in detections:
        landmarks, states = face_landmark_handle.run(image, detection)
        face_landmark_handle.show_result(image, landmarks)

    

    while True:
        cv2.imshow('Result', image)  
        if cv2.waitKey(1) & 0xFF == ord('q'):  # If 'q' key is pressed then break
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run face landmark detection on an image.')
    parser.add_argument('--img_path', help='Path to the image file.')
    args = parser.parse_args()

    thread = threading.Thread(target=image_run, args=(args.img_path,))  # 创建守护线程
    thread.daemon = True  # 设置为守护线程
    thread.start()  # 开始线程

    # Your main program continues to run in parallel with the thread
    while True:
        # 模拟主线程运行，此处可添加主线程的任务
        pass