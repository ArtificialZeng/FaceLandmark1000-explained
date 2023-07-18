import torch
import numpy as np
import cv2
import onnx
import onnxruntime
import math


class FaceLandmark(object):
    def __init__(self):
        self.model_path = r'model/FaceLandmark.onnx'
        self.onnx_model = onnx.load(self.model_path)
        onnx.checker.check_model(self.onnx_model)
        self.ort_session = onnxruntime.InferenceSession(self.model_path)
        self.image_size = 128
        self.min_face = 100
        self.iou_thres = 0.5
        self.thres = 1
        self.filter = OneEuroFilter()
        self.previous_landmarks_set = None

    def run(self, image, bbox):
        processed_image, details = self.preprocess(image, bbox)  # 调用预处理函数，处理输入的图像和边界框
        ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(processed_image)}  # 将处理过的图像转为numpy数组，用于传入ONNX Runtime会话
        result = self.ort_session.run(None, ort_inputs)  # 运行ONNX Runtime会话
        landmarks = result[0][0, :1946].reshape(-1, 2)  # 提取特征点信息
        states = result[(1946 + 3):]  # 提取状态信息
        landmarks = self.postprocess(landmarks, details)  # 对特征点进行后处理
        return np.array(landmarks), np.array(states)  # 返回特征点和状态信息
    
    def show_result(self, image, landmark):
        for point in landmark:  # 遍历每一个特征点
            cv2.circle(image, center=(int(point[0]), int(point[1])),  # 在图像上画圆标记特征点
                       color=(255, 122, 122), radius=1, thickness=1)
        cv2.imshow('', image)  # 展示处理过的图像
        cv2.waitKey(1)  # 等待1毫秒，以便看到图像
'''
这个代码中的 1946 是一个固定的值，可能是因为这个模型的设计者知道预期的结果输出的具体结构。特别的，它可能是在某种情况下模型输出结果的特定形状或布局。这是针对特定模型的硬编码值，并且可能对其他模型并不适用。

在这个例子中，result[0][0, :1946].reshape(-1, 2) 这行代码可能意味着模型预计会产生至少 1946 个值作为特征点(landmarks)，并且这些特征点预期以二维坐标的形式表示，因此它们被reshape为一个二维数组，每行包含两个元素。所以1946应该能够被2整除，也就是说，这个模型预计会产生 973 个特征点。

对于 states = result[(1946 + 3):]，这里的 1946 + 3 可能表示在 1946 个特征点数据后面，还有3个数据是用于其他目的，例如分隔符或者其他信息。然后剩余的数据被视为状态信息。具体的意义和用途取决于模型的设计者。

以上的所有解释都基于对这个代码片段的猜测，只有编写这段代码的原作者才能给出确切的解释。这些数字可能在模型训练过程中产生，或者是根据实验结果选择的，每种模型都可能有不同的输出结构。
'''


    def preprocess(self, image, bbox):
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        if bbox_width <= self.min_face or bbox_height <= self.min_face:
            return None, None
        add = int(max(bbox_width, bbox_height))
        bimg = cv2.copyMakeBorder(image, add, add, add, add,
                                  borderType=cv2.BORDER_CONSTANT,
                                  value=np.array([127., 127., 127.]))
        bbox += add

        face_width = (1 + 2 * 0.1) * bbox_width
        face_height = (1 + 2 * 0.2) * bbox_height
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

        bbox[0] = center[0] - face_width // 2
        bbox[1] = center[1] - face_height // 2
        bbox[2] = center[0] + face_width // 2
        bbox[3] = center[1] + face_height // 2

        # crop
        bbox = bbox.astype(np.int)
        crop_image = bimg[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        h, w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, (self.image_size, self.image_size))
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        crop_image = np.expand_dims(crop_image, axis=0)
        crop_image = np.expand_dims(crop_image, axis=0)
        crop_image = torch.from_numpy(crop_image).detach().float()
        return crop_image, [h, w, bbox[1], bbox[0], add]


    def postprocess(self, landmark, detail):
        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]  # 对landmark的x坐标进行处理
        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]  # 对landmark的y坐标进行处理
        return landmark  # 返回处理后的landmark
    
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()  # 将Tensor对象转为Numpy数组
    
    def calculate(self, now_landmarks_set):  # 计算特征点
        if self.previous_landmarks_set is None or self.previous_landmarks_set.shape[0]==0:  # 如果previous_landmarks_set为空或长度为0
            self.previous_landmarks_set = now_landmarks_set  # 将当前的landmarks赋值给previous_landmarks_set
            result = now_landmarks_set  # 将当前的landmarks赋值给result
        else:  # 如果previous_landmarks_set不为空
            if self.previous_landmarks_set.shape[0] == 0:  # 如果previous_landmarks_set的长度为0
                return now_landmarks_set  # 直接返回当前的landmarks
            else:  # 如果previous_landmarks_set的长度不为0
                result = []  # 创建一个空列表result
                for i in range(now_landmarks_set.shape[0]):  # 遍历当前的landmarks
                    not_in_flag = True  # 设置一个标志
                    for j in range(self.previous_landmarks_set.shape[0]):  # 遍历previous_landmarks_set
                        if self.iou(now_landmarks_set[i], self.previous_landmarks_set[j]) > self.iou_thres:  # 如果当前landmark与previous_landmark的iou大于阈值
                            result.append(self.smooth(now_landmarks_set[i], self.previous_landmarks_set[j]))  # 对当前landmark进行平滑处理并添加到result中
                            not_in_flag = False  # 将标志设为False
                            break  # 跳出内层循环
                    if not_in_flag:  # 如果标志为True
                        result.append(now_landmarks_set[i])  # 直接将当前landmark添加到result中
        result = np.array(result)  # 将result转为Numpy数组
        self.previous_landmarks_set=result  # 将result赋值给previous_landmarks_set
        return result  # 返回result


    def iou(self, p_set0, p_set1):
        rec1=[np.min(p_set0[:, 0]), np.min(p_set0[:, 1]), np.max(p_set0[:, 0]), np.max(p_set0[:, 1])]
        rec2 = [np.min(p_set1[:, 0]), np.min(p_set1[:, 1]), np.max(p_set1[:, 0]), np.max(p_set1[:, 1])]

        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        x1 = max(rec1[0], rec2[0])
        y1 = max(rec1[1], rec2[1])
        x2 = min(rec1[2], rec2[2])
        y2 = min(rec1[3], rec2[3])

        # judge if there is an intersect
        intersect = max(0, x2 - x1) * max(0, y2 - y1)

        return intersect / (sum_area - intersect)

    def smooth(self, now_landmarks, previous_landmarks):

        result=[]
        for i in range(now_landmarks.shape[0]):

            dis = np.sqrt(np.square(now_landmarks[i][0] - previous_landmarks[i][0]) + np.square(now_landmarks[i][1] - previous_landmarks[i][1]))

            if dis < self.thres:
                result.append(previous_landmarks[i])
            else:
                result.append(self.filter(now_landmarks[i], previous_landmarks[i]))

        return np.array(result)


class OneEuroFilter:
    def __init__(self, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.

        self.dx_prev = float(dx0)
        #self.t_prev = float(t0)

    def __call__(self, x,x_prev):

        if x_prev is None:

            return x
        """Compute the filtered signal."""
        t_e = 1

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, x_prev)

        # Memorize the previous values.

        self.dx_prev = dx_hat
        return x_hat

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev


if __name__ == '__main__':
    image = cv2.imread('data/1.jpg')
    bbox = np.array([117.58737, 58.62614, 354.0737, 401.39395])
    handle = FaceLandmark()
    handle.run(image, bbox)
