''' :Author: "QingTai Jiang"
    :keyword:"实现了YoloV3的封装"
    :time:"2018/11/30"
'''
from yolo import YOLO
from PIL import Image
import cv2
import numpy
class Yolo_NN(object):
    def __init__(self):
        self.yolov3=YOLO()
    def predict(self,image,show_picture=False):

        r_image = self.yolov3.detect_image(image)
        if show_picture:
            self.yolov3.detect_image(image)

        print("r_image:",r_image)
        img = cv2.cvtColor(numpy.asarray(r_image), cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return r_image
if __name__ == '__main__':

    YL=Yolo_NN()
    IMAGE_PATH = './222.jpg'
    image = Image.open(IMAGE_PATH)
    temp = YL.predict(image)
    print("temp", temp)



