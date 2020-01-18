import cv2
import matplotlib.pyplot as plt
import numpy as np


class image(object):
    def __init__(self, path, show_size=(3, 3)):
        self.img = cv2.imread(path)
        self.show_size = show_size

    def get_image_base_info(self):
        img = self.img
        self.shape = img.shape  # shape返回的是图像的高像素，宽像素，通道数
        self.size = img.size  # 总通道数=高* 宽* 通道数
        self.dtype = img.dtype  # 查看图片的数据类型 uint8  3个通道每个通道占的位数（8位，一个字节)
        self.B_parm_np, self.G_parm_np, self.R_parm_np = np.array(img)  # 每个像素点的参数（ B , G , R )

    def show_image(self, new_img=''):
        if len(new_img):
            img = new_img
        else:
            img = self.img
        plt.figure(figsize=self.show_size)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def show_image_for_opencv(self):
        cv2.imshow('img', self.img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

    def change_channel_value(self, channel_dict):
        img = self.img
        for key, value in channel_dict.items():
            key = key.lower()
            if key == 'b':
                img[:, :, 0] = value
            elif key == 'g':
                img[:, :, 1] = value
            else:
                img[:, :, 2] = value  # 改变单个通道（0,1,2 => B,G,R) 注： 这里的：表示全部，这里是将全部的像素位置的R变成定值
        B, G, R = cv2.split(img)  # 通道分离,  可以用于通道图的单独显示，如:cv2.imshow('B', self.B)
        img = cv2.merge([B, G, R])  # 合并通道
        self.show_image(new_img=img)

    def channel_num_transformation(self, channel_dict):
        img = self.img
        B, G, R = cv2.split(img)
        for key, value in channel_dict.items():
            key = key.lower()
            value = int(value)
            if key == 'b':
                b_lim = 255 - value
                B[B > b_lim] = 255
                B[B <= b_lim] = (value + B[B <= b_lim]).astype(img.dtype)

        img = cv2.merge([B, G, R])  # 合并通道
        self.show_image(new_img=img)

    def image_clipping(self, lx, ly, rx, ry):
        img = self.img[ly:ry, lx:rx]
        self.show_image(new_img=img)

    def image_rotate(self, x_center, y_center, angle, scaling_factor):
        """
            按角度旋转，并裁剪
        :param x_center: 中心点X的坐标
        :param y_center: 中心点Y的坐标
        :param angle: 旋转角度
        :param scaling_factor: 缩放因子
        :return:
        """
        height, width = self.img.shape[:2]
        CX, CY = width // x_center, height // y_center
        M = cv2.getRotationMatrix2D((CX, CY), angle, scaling_factor)  # 第一个参数是旋转中心，第二个参数是旋转角度，第三个因子是旋转后的缩放因子
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算图像的新边界维数
        NW = int((height * sin) + (width * cos))
        NH = int((height * cos) + (width * sin))

        # 调整旋转矩阵以考虑平移
        M[0, 2] += (NW / 2) - CX
        M[1, 2] += (NH / 2) - CY

        img = cv2.warpAffine(self.img, M, (NW, NH))  # 第三个参数是输出图像的尺寸中心，图像的宽和高
        self.show_image(new_img=img)

    def adjust_gamma(self, gamma):
        """
        Gamma变换就是用来图像增强，其提升了暗部细节，

        简单来说就是通过非线性变换，让图像从暴光强度的线性响应变得更接近人眼感受的响应，
        即将漂白（相机曝光）或过暗（曝光不足）的图片，进行矫正
        :param gamma:
        :return:
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        img = cv2.LUT(self.img, table)
        self.show_image(new_img=img)

    def image_affine(self):
        """
        图片的仿射，就是在图像的旋转加上拉升就是图像仿射变换 需要三个点
        :return:
        """
        rows, cols = self.img.shape[:2]
        pts1 = np.float32([[1, 0], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 20], [200, 20], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(self.img, M, (rows, cols))  # 第三个参数：变换后的图像大小
        self.show_image(new_img=img)

    def image_projective(self):
        """
        透视  就是视角变换， 在变换前后要保证直线还是直线。 需要四点，且四个点中的任意三个点不能共线。
        :return:
        """
        rows, cols = self.img.shape[:2]
        pts1 = np.float32([[0, 0], [368, 52], [28, 387], [589, 390]])
        pts2 = np.float32([[1, 1], [300, 0], [0, 300], [500, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(self.img, M, (rows, cols))
        self.show_image(new_img=img)

    def change_color_with_YUV(self):
        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        self.show_image(new_img=img)

    def change_color_with_HSV(self):
        rows, cols = self.img.shape[:2]
        img = cv2.resize(self.img, None, fx=1.0, fy=1.0)

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([0, 0, 255])
        upper_blue = np.array([155, 255, 255])
        make = cv2.inRange(hsv, lower_blue, upper_blue)
        erode = cv2.erode(make, None, iterations=1)
        dilate = cv2.dilate(erode, None, iterations=1)
        for i in range(rows):
            for j in range(cols):
                if dilate[i, j] == 255:
                    img[i, j] = (70, 240, 40)  # 此处替换颜色，为BGR通道

        self.show_image(new_img=img)


if __name__ == "__main__":
    path = "scarlett_johansson.jpg"
    img = image(path=path, show_size=(5, 5))
    img.show_image()
    img.image_clipping(230, 0, 500, 400)
    img.channel_num_transformation({"B": "50"})
    img.image_rotate(2, 2, -90, 1.0)
    img.adjust_gamma(1.5)
    img.image_affine()
    img.image_projective()
    img.change_color_with_YUV()
    img.change_channel_value({"R": 100})

    path = "scarlett_johansson_one.jpg"
    img_one = image(path=path, show_size=(5, 5))
    img_one.change_color_with_HSV()


class image(object):
    def __init__(self, path, show_size=(3, 3)):
        self.img = cv2.imread(path)
        self.show_size = show_size

    def show(self, image_new=""):
        if len(image_new):
            image = image_new
        else:
            image = self.img

        #         plt.figure(figsize=self.show_size)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为灰度
        plt.show()

    def median_blur(self, ksize=5):
        img = self.img
        img_new = cv2.medianBlur(img, ksize)
        self.show(image_new=img_new)

    def median_gray(self, ksize=5):  # image为传入灰度图像，ksize为滤波窗口大小
        image = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        high, wide = image.shape[:2]
        img = image.copy()
        mid = (ksize - 1) // 2

        med_arry = []
        for i in range(high - ksize):
            for j in range(wide - ksize):
                for m1 in range(ksize):
                    for m2 in range(ksize):
                        med_arry.append(int(image[i + m1, j + m2]))

                med_arry.sort()  # 对窗口像素点排序
                img[i + mid, j + mid] = med_arry[(len(med_arry) + 1) // 2]  # 将滤波窗口的中值赋给滤波窗口中间的像素点
                del med_arry[:]
        img_new = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.show(image_new=img_new)

    def mid_sort(data):
        for i in range(len(data) - 1):
            for j in range(len(data) - 2):
                if data[j + 1] < data[j]:
                    tmp = data[j]
                    data[j] = data[j + 1]
                    data[j + 1] = tmp
        index = int(len(data) / 2)
        return data[index]

    def flatten(matrix):
        w = len(matrix)
        h = len(matrix[0])
        tmp = []
        for i in range(w):
            for j in range(h):
                tmp.append(matrix[i][j])
        return tmp

    def mid_filter(data, width, height, n):
        tmp = data
        for i in range(0, height - n + 1):  # 先h后w :(
            for j in range(0, width - n + 1):
                modle = data[i:i + n]
                modle = modle[:, j:j + n]
                modle = self.flatten(modle)
                mid = self.mid_sort(modle)
                tmp[i + int((n - 1) / 2), j + int((n - 1) / 2)] = mid
        return tmp

    def median_new(self):
        im = self.img
        data = []
        width, height = im.shape[:2][::-1]

        # 读取图像像素值，并计算灰度值
        for h in range(height):
            row = []
            for w in range(width):
                value = im.getpixel((w, h))
                row.append((value[0] + value[1] + value[2]) / 3)  # pixel是RGBA  :(，所以需要计算灰度值
            data.append(row)

        # 二维中值滤波
        data = np.float32(data)
        # data = signal.medfilt2d(data, (3,3))
        data = self.mid_filter(data, width, height, 3)

        # 创建并保存结果图像
        for h in range(height):
            for w in range(width):
                tmp = (int(data[h][w]), int(data[h][w]), int(data[h][w]))
                # im.putpixel((w,h), int(data[h][w]))  放回去还要改维数 :(
                im.putpixel((w, h), tmp)

        self.show(image_new=im)
