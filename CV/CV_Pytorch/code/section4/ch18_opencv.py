# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch18_opencv.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/27 10:00
# https://chat.deepseek.com/a/chat/s/871afbd1-3cde-460c-956c-a8dd1bdb05da

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
from typing import List, Tuple, Optional


def download_file(url: str, filename: str) -> None:
    """下载文件并保存到当前目录"""
    if not os.path.exists(filename):
        print(f"正在下载 {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"下载完成: {filename}")
    else:
        print(f"文件已存在: {filename}")


class PanoramaStitcher:
    """全景图像拼接类"""

    def __init__(self):
        self.feature_extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def download_images(self):
        """下载全景拼接所需图像"""
        download_file("https://www.dropbox.com/s/mfg1codtc2rue84/g1.png", "g1.png")
        download_file("https://www.dropbox.com/s/4yhui8s1xjndavm/g2.png", "g2.png")

    def load_images(self):
        """加载图像"""
        queryImg = cv2.imread('g1.png', 1)
        queryImg_gray = cv2.imread('g1.png', 0)

        trainImg = cv2.imread('g2.png', 1)
        trainImg_gray = cv2.imread('g2.png', 0)

        return queryImg, queryImg_gray, trainImg, trainImg_gray

    def display_images(self, images, titles, nc=2, figsize=(10, 5)):
        """显示图像"""
        plt.figure(figsize=figsize)
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, nc, i + 1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def extract_features(self, img_gray):
        """提取图像特征"""
        kps, features = self.feature_extractor.detectAndCompute(img_gray, None)
        return kps, features

    def draw_keypoints(self, img_gray, kps):
        """绘制关键点"""
        return cv2.drawKeypoints(img_gray, kps, None, color=(0, 255, 0))

    def match_features(self, featuresA, featuresB):
        """特征匹配"""
        matches = self.matcher.match(featuresA, featuresB)
        return sorted(matches, key=lambda x: x.distance)

    def draw_matches(self, imgA, kpsA, imgB, kpsB, matches, num_matches=100):
        """绘制匹配结果"""
        return cv2.drawMatches(imgA, kpsA, imgB, kpsB, matches[:num_matches],
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def find_homography(self, kpsA, kpsB, matches):
        """计算单应性矩阵"""
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4)
        return H, status

    def stitch_images(self, imgA, imgB, H):
        """图像拼接"""
        height = imgA.shape[0] + imgB.shape[0]
        width = imgA.shape[1] + imgB.shape[1]

        result = cv2.warpPerspective(imgB, H, (width, height))
        result[0:imgA.shape[0], 0:imgA.shape[1]] = imgA

        # 裁剪黑色边框
        y_nonzero, x_nonzero = np.nonzero(result.sum(axis=2))
        if len(y_nonzero) > 0 and len(x_nonzero) > 0:
            result = result[min(y_nonzero):max(y_nonzero), min(x_nonzero):max(x_nonzero)]

        return result

    def run(self):
        """运行全景拼接流程"""
        print("开始全景图像拼接...")
        self.download_images()
        queryImg, queryImg_gray, trainImg, trainImg_gray = self.load_images()

        # 显示原始图像
        self.display_images(
            [trainImg, queryImg],
            ['Query image', 'Training image (Image to be stitched to Query image)']
        )

        # 提取特征
        kpsA, featuresA = self.extract_features(trainImg_gray)
        kpsB, featuresB = self.extract_features(queryImg_gray)

        # 显示关键点
        img_kpsA = self.draw_keypoints(trainImg_gray, kpsA)
        img_kpsB = self.draw_keypoints(queryImg_gray, kpsB)
        self.display_images(
            [img_kpsB, img_kpsA],
            ['Query image with keypoints', 'Training image with keypoints']
        )

        # 特征匹配
        matches = self.match_features(featuresA, featuresB)

        # 显示匹配结果
        img_matches = self.draw_matches(trainImg, kpsA, queryImg, kpsB, matches)
        plt.figure(figsize=(15, 5))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title('Feature matches')
        plt.axis('off')
        plt.show()

        # 计算单应性矩阵并拼接图像
        kpsA_pts = np.float32([kp.pt for kp in kpsA])
        kpsB_pts = np.float32([kp.pt for kp in kpsB])
        H, status = self.find_homography(kpsA_pts, kpsB_pts, matches)

        result = self.stitch_images(queryImg, trainImg, H)

        # 显示拼接结果
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Stitched image')
        plt.axis('off')
        plt.show()


class LaneDetector:
    """车道线检测类"""

    def __init__(self):
        pass

    def download_image(self):
        """下载道路图像"""
        download_file("https://www.dropbox.com/s/vgd22go8a6k721t/road_image.png", "road_image.png")

    def load_image(self):
        """加载图像"""
        img = cv2.imread('road_image.png')
        return img

    def canny_edge_detection(self, img, low_threshold=50, high_threshold=150):
        """Canny边缘检测"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges

    def hough_line_detection(self, edges, rho=1, theta=np.pi / 180, threshold=150):
        """霍夫直线检测"""
        lines = cv2.HoughLines(edges, rho, theta, threshold)
        return lines

    def draw_lines(self, img, lines):
        """在图像上绘制检测到的直线"""
        img_with_lines = img.copy()
        if lines is not None:
            lines = lines[:, 0, :]
            for rho, theta in lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))
                cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img_with_lines

    def run(self):
        """运行车道线检测流程"""
        print("开始车道线检测...")
        self.download_image()
        img = self.load_image()

        # 显示原始图像
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original road image')
        plt.axis('off')
        plt.show()

        # 边缘检测
        edges = self.canny_edge_detection(img)

        # 显示边缘检测结果
        plt.figure(figsize=(10, 6))
        plt.imshow(edges, cmap='gray')
        plt.title('Canny edges')
        plt.axis('off')
        plt.show()

        # 霍夫直线检测
        lines = self.hough_line_detection(edges)

        # 绘制检测到的直线
        img_with_lines = self.draw_lines(img, lines)

        # 显示结果
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
        plt.title('Detected lanes')
        plt.axis('off')
        plt.show()


class ColorBasedDetector:
    """基于颜色的物体检测类"""

    def __init__(self):
        pass

    def download_image(self):
        """下载UNO卡片图像"""
        download_file("https://www.dropbox.com/s/utrkdooh08y9mvm/uno_card.png", "uno_card.png")

    def load_image(self):
        """加载图像"""
        img = cv2.imread('uno_card.png')
        return img

    def create_color_mask(self, img, lower_bound, upper_bound):
        """创建颜色掩膜"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return mask

    def apply_mask(self, img, mask):
        """应用掩膜"""
        return cv2.bitwise_and(img, img, mask=mask)

    def display_results(self, images, titles, nc=3, figsize=(15, 5)):
        """显示结果"""
        plt.figure(figsize=figsize)
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, nc, i + 1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def run(self):
        """运行基于颜色的物体检测流程"""
        print("开始基于颜色的物体检测...")
        self.download_image()
        img = self.load_image()

        # 显示原始图像
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original image')
        plt.axis('off')
        plt.show()

        # 定义绿色范围 (HSV颜色空间)
        lower_green = np.array([45, 100, 100])
        upper_green = np.array([80, 255, 255])

        # 创建掩膜
        mask = self.create_color_mask(img, lower_green, upper_green)

        # 应用掩膜
        result = self.apply_mask(img, mask)

        # 显示结果
        self.display_results(
            [img, mask, result],
            ['Original image', 'Mask on image', 'Resulting image']
        )


class LicensePlateDetector:
    """车牌检测类"""

    def __init__(self):
        self.plate_cascade = None

    def download_files(self):
        """下载所需文件"""
        download_file("https://raw.githubusercontent.com/zeusees/HyperLPR/master/model/cascade.xml", "cascade.xml")
        download_file("https://www.dropbox.com/s/4hbem2kxzqcwo0y/car1.jpg", "car1.jpg")

    def load_cascade(self):
        """加载级联分类器"""
        if self.plate_cascade is None:
            self.plate_cascade = cv2.CascadeClassifier('cascade.xml')

    def load_image(self):
        """加载图像"""
        img = cv2.imread('car1.jpg')
        return img

    def detect_plates(self, img):
        """检测车牌"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=2,
            minSize=(40, 40),
            maxSize=(1000, 100)
        )
        return plates, gray

    def draw_plates(self, img, plates):
        """在图像上绘制检测到的车牌"""
        img_with_boxes = img.copy().astype('uint8')
        for (x, y, w, h) in plates:
            # 调整边界框大小
            x -= w * 0.14
            w += w * 0.75
            y -= h * 0.15
            h += h * 0.3

            # 绘制矩形
            cv2.rectangle(
                img_with_boxes,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (0, 255, 0),
                10
            )
        return img_with_boxes

    def run(self):
        """运行车牌检测流程"""
        print("开始车牌检测...")
        self.download_files()
        self.load_cascade()
        img = self.load_image()

        # 检测车牌
        plates, gray = self.detect_plates(img)
        print(f"检测到 {len(plates)} 个车牌")

        # 绘制检测结果
        img_with_boxes = self.draw_plates(img, plates)

        # 显示结果
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title('Detected license plates')
        plt.axis('off')
        plt.show()


class TextBoundingBoxDetector:
    """文本边界框检测类"""

    def __init__(self):
        pass

    def download_image(self):
        """下载文本图像"""
        download_file("https://www.dropbox.com/s/3jkwy16m6xdlktb/18_5.JPG", "18_5.JPG")

    def load_image(self):
        """加载图像"""
        img = cv2.imread('18_5.JPG')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img_rgb

    def preprocess_image(self, img):
        """图像预处理"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 二值化
        binary = np.uint8(gray < 200) * 255
        return gray, binary

    def find_contours(self, binary_img):
        """查找轮廓"""
        contours, hierarchy = cv2.findContours(
            binary_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def draw_bounding_boxes(self, img, contours, min_area=5, min_height=5, max_height=100):
        """绘制边界框"""
        img_with_boxes = img.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if min_height < h < max_height:
                    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img_with_boxes

    def dilate_image(self, img, kernel_size=(1, 2), iterations=1):
        """图像膨胀"""
        kernel = np.ones(kernel_size, np.uint8)
        dilated = cv2.dilate(img, kernel, iterations=iterations)
        return dilated

    def run(self):
        """运行文本边界框检测流程"""
        print("开始文本边界框检测...")
        self.download_image()
        img, img_rgb = self.load_image()

        # 显示原始图像
        plt.figure(figsize=(15, 10))
        plt.imshow(img_rgb)
        plt.title('Original image')
        plt.axis('off')
        plt.show()

        # 预处理
        gray, binary = self.preprocess_image(img_rgb)

        # 显示灰度图像和二值图像
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary image')
        plt.axis('off')
        plt.show()

        # 查找轮廓并绘制边界框
        contours = self.find_contours(binary)
        img_with_boxes = self.draw_bounding_boxes(img_rgb, contours)

        # 显示初步结果
        plt.figure(figsize=(15, 10))
        plt.imshow(img_with_boxes)
        plt.title('Initial bounding boxes')
        plt.axis('off')
        plt.show()

        # 创建三通道的二值图像用于膨胀
        binary_3ch = np.stack([binary] * 3, axis=2)

        # 膨胀处理
        dilated = self.dilate_image(binary)

        # 查找膨胀后的轮廓
        dilated_contours = self.find_contours(np.uint8(dilated))

        # 绘制最终的边界框
        final_img = img_rgb.copy()
        for cnt in dilated_contours:
            if cv2.contourArea(cnt) > 5:
                x, y, w, h = cv2.boundingRect(cnt)
                if 5 < h < 100:
                    cv2.rectangle(final_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示最终结果
        plt.figure(figsize=(15, 10))
        plt.imshow(final_img)
        plt.title('Final bounding boxes after dilation')
        plt.axis('off')
        plt.show()


def main():
    """主函数"""
    print("计算机视觉工具集")
    print("=" * 50)
    print("1. 全景图像拼接")
    print("2. 车道线检测")
    print("3. 基于颜色的物体检测")
    print("4. 车牌检测")
    print("5. 文本边界框检测")
    print("6. 运行所有")
    print("=" * 50)

    choice = input("请选择要运行的功能 (1-6): ").strip()

    if choice == "1":
        stitcher = PanoramaStitcher()
        stitcher.run()
    elif choice == "2":
        detector = LaneDetector()
        detector.run()
    elif choice == "3":
        detector = ColorBasedDetector()
        detector.run()
    elif choice == "4":
        detector = LicensePlateDetector()
        detector.run()
    elif choice == "5":
        detector = TextBoundingBoxDetector()
        detector.run()
    elif choice == "6":
        # 运行所有功能
        stitcher = PanoramaStitcher()
        stitcher.run()

        detector = LaneDetector()
        detector.run()

        detector = ColorBasedDetector()
        detector.run()

        detector = LicensePlateDetector()
        detector.run()

        detector = TextBoundingBoxDetector()
        detector.run()
    else:
        print("无效选择，请重新运行程序并选择1-6之间的数字。")


if __name__ == "__main__":
    main()