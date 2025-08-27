# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch17_deployment.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/27 10:00
# https://chat.deepseek.com/a/chat/s/8df0d1ea-7cdb-4fcb-aded-1312ea42ec83
import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


class FMNISTModel(nn.Module):
    """FMNIST分类模型"""

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def __init__(self, model_path='fmnist.weights.pth'):
        """初始化模型并加载权重"""
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print('Loaded FMNIST Model')

    def _build_model(self):
        """构建模型架构"""
        return nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        ).to(self.device)

    def forward(self, x):
        """前向传播"""
        x = x.view(1, -1).to(self.device)
        pred = self.model(x)
        pred = F.softmax(pred, -1)[0]
        conf, clss = pred.max(-1)
        clss = self.classes[clss.cpu().item()]
        return conf.item(), clss

    def preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, str):  # 文件路径
            x = cv2.imread(image, 0)
        elif isinstance(image, Image.Image):  # PIL图像
            x = np.array(image.convert('L'))
        else:  # numpy数组
            x = np.array(image)

        x = cv2.resize(x, (28, 28))
        x = torch.Tensor(255 - x) / 255.
        return x

    def predict_from_path(self, path):
        """从文件路径预测"""
        x = self.preprocess_image(path)
        conf, clss = self(x)
        return {'class': clss, 'confidence': f'{conf:.4f}'}

    def predict_from_image(self, image):
        """从图像对象预测"""
        x = self.preprocess_image(image)
        conf, clss = self(x)
        return {'class': clss, 'confidence': f'{conf:.4f}'}


class FMNISTApp:
    """FMNIST Web应用"""

    def __init__(self, model_path='fmnist.weights.pth'):
        """初始化应用"""
        self.model = FMNISTModel(model_path)
        self.app = FastAPI()
        self._setup_routes()
        self._setup_mounts()

    def _setup_mounts(self):
        """设置静态文件挂载"""
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.app.mount("/files", StaticFiles(directory="files"), name="files")

    def _setup_routes(self):
        """设置路由"""
        templates = Jinja2Templates(directory="templates")

        @self.app.get("/")
        async def read_item(request: Request):
            return templates.TemplateResponse("home.html", {"request": request})

        @self.app.post('/uploaddata/')
        async def upload_file(request: Request, file: UploadFile = File(...)):
            content = file.file.read()
            saved_filepath = f'files/{file.filename}'

            # 确保files目录存在
            os.makedirs(os.path.dirname(saved_filepath), exist_ok=True)

            with open(saved_filepath, 'wb') as f:
                f.write(content)

            output = self.model.predict_from_path(saved_filepath)
            payload = {
                'request': request,
                "filename": file.filename,
                'output': output
            }
            return templates.TemplateResponse("home.html", payload)

        @self.app.post("/predict")
        def predict(request: Request, file: UploadFile = File(...)):
            content = file.file.read()
            image = Image.open(io.BytesIO(content)).convert('L')
            output = self.model.predict_from_image(image)
            return output


def run_web_app(model_path='fmnist.weights.pth', host="0.0.0.0", port=8000):
    """运行Web应用"""
    import uvicorn
    app = FMNISTApp(model_path).app
    uvicorn.run(app, host=host, port=port)


def predict_single_image(image_path, model_path='fmnist.weights.pth'):
    """预测单张图像"""
    model = FMNISTModel(model_path)
    result = model.predict_from_path(image_path)
    print(f"预测结果: {result['class']}, 置信度: {result['confidence']}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FMNIST分类应用')
    parser.add_argument('--mode', type=str, default='web',
                        choices=['web', 'predict'],
                        help='运行模式: web服务或单张图像预测')
    parser.add_argument('--image', type=str,
                        help='预测模式下的图像路径')
    parser.add_argument('--model', type=str, default='fmnist.weights.pth',
                        help='模型权重文件路径')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Web服务主机地址')
    parser.add_argument('--port', type=int, default=8000,
                        help='Web服务端口')

    args = parser.parse_args()

    if args.mode == 'web':
        run_web_app(args.model, args.host, args.port)
    elif args.mode == 'predict':
        if not args.image:
            print("预测模式需要指定图像路径 (--image)")
            exit(1)
        predict_single_image(args.image, args.model)
