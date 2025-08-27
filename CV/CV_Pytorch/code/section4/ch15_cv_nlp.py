# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch15_cv_nlp.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/27 9:59
# https://chat.deepseek.com/a/chat/s/f0c91096-3a12-439a-bd87-4a80ec4352c5
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import editdistance
from torch_snippets import *
from torchsummary import summary

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== 手写文字识别 (Handwriting Transcription) ====================

def download_handwriting_data():
    """下载手写文字识别数据"""
    if not os.path.exists('synthetic-data'):
        os.system('wget -q https://www.dropbox.com/s/l2ul3upj7dkv4ou/synthetic-data.zip')
        os.system('unzip -qq synthetic-data.zip')
        os.system('rm synthetic-data.zip')


class OCRDataset(Dataset):
    def __init__(self, items, vocab='QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm',
                 preprocess_shape=(32, 128), timesteps=32):
        super().__init__()
        self.items = items
        self.charList = {ix + 1: ch for ix, ch in enumerate(vocab)}
        self.charList.update({0: '`'})
        self.invCharList = {v: k for k, v in self.charList.items()}
        self.ts = timesteps
        self.H, self.W = preprocess_shape

    def __len__(self):
        return len(self.items)

    def sample(self):
        return self[randint(len(self))]

    def __getitem__(self, ix):
        item = self.items[ix]
        image = cv2.imread(item, 0)
        fname2label = lambda fname: stem(fname).split('@')[0]
        label = fname2label(item)
        return image, label

    def collate_fn(self, batch):
        images, labels, label_lengths, label_vectors, input_lengths = [], [], [], [], []
        for image, label in batch:
            images.append(torch.Tensor(self.preprocess(image))[None, None])
            label_lengths.append(len(label))
            labels.append(label)
            label_vectors.append(self.str2vec(label))
            input_lengths.append(self.ts)
        images = torch.cat(images).float().to(device)
        label_lengths = torch.Tensor(label_lengths).long().to(device)
        label_vectors = torch.Tensor(label_vectors).long().to(device)
        input_lengths = torch.Tensor(input_lengths).long().to(device)
        return images, label_vectors, label_lengths, input_lengths, labels

    def str2vec(self, string, pad=True):
        string = ''.join([s for s in string if s in self.invCharList])
        val = list(map(lambda x: self.invCharList[x], string))
        if pad:
            while len(val) < self.ts:
                val.append(0)
        return val

    def preprocess(self, img, shape=None):
        if shape is None:
            shape = (self.H, self.W)
        target = np.ones(shape) * 255
        try:
            H, W = shape
            h, w = img.shape
            fx = H / h
            fy = W / w
            f = min(fx, fy)
            _h = int(h * f)
            _w = int(w * f)
            _img = cv2.resize(img, (_w, _h))
            target[:_h, :_w] = _img
        except:
            pass
        return (255 - target) / 255

    def decoder_chars(self, pred):
        decoded = ""
        last = ""
        pred = pred.cpu().detach().numpy()
        for i in range(len(pred)):
            k = np.argmax(pred[i])
            if k > 0 and self.charList[k] != last:
                last = self.charList[k]
                decoded = decoded + last
            elif k > 0 and self.charList[k] == last:
                continue
            else:
                last = ""
        return decoded.replace(" ", " ")

    def wer(self, preds, labels):
        c = 0
        for p, l in zip(preds, labels):
            c += p.lower().strip() != l.lower().strip()
        return round(c / len(preds), 4)

    def cer(self, preds, labels):
        c, d = [], []
        for p, l in zip(preds, labels):
            c.append(editdistance.eval(p, l) / len(l))
        return round(np.mean(c), 4)

    def evaluate(self, model, ims, labels, lower=False):
        model.eval()
        preds = model(ims).permute(1, 0, 2)  # B, T, V+1
        preds = [self.decoder_chars(pred) for pred in preds]
        return {'char-error-rate': self.cer(preds, labels),
                'word-error-rate': self.wer(preds, labels),
                'char-accuracy': 1 - self.cer(preds, labels),
                'word-accuracy': 1 - self.wer(preds, labels)}


class BasicBlock(nn.Module):
    def __init__(self, ni, no, ks=3, st=1, padding=1, pool=2, drop=0.2):
        super().__init__()
        self.ks = ks
        self.block = nn.Sequential(
            nn.Conv2d(ni, no, kernel_size=ks, stride=st, padding=padding),
            nn.BatchNorm2d(no, momentum=0.3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout2d(drop)
        )

    def forward(self, x):
        return self.block(x)


class Ocr(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.model = nn.Sequential(
            BasicBlock(1, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 256, pool=(4, 2)),
            Reshape(-1, 256, 32),
            Permute(2, 0, 1)  # T, B, D
        )
        self.rnn = nn.Sequential(
            nn.LSTM(256, 256, num_layers=2, dropout=0.2, bidirectional=True),
        )
        self.classification = nn.Sequential(
            nn.Linear(512, vocab + 1),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.model(x)
        x, lstm_states = self.rnn(x)
        y = self.classification(x)
        return y


def ctc(log_probs, target, input_lengths, target_lengths, blank=0):
    loss = nn.CTCLoss(blank=blank, zero_infinity=True)
    ctc_loss = loss(log_probs, target, input_lengths, target_lengths)
    return ctc_loss


def train_ocr_batch(data, model, optimizer, criterion):
    model.train()
    imgs, targets, label_lens, input_lens, labels = data
    optimizer.zero_grad()
    preds = model(imgs)
    loss = criterion(preds, targets, input_lens, label_lens)
    loss.backward()
    optimizer.step()
    results = trn_ds.evaluate(model, imgs.to(device), labels)
    return loss, results


@torch.no_grad()
def validate_ocr_batch(data, model, criterion):
    model.eval()
    imgs, targets, label_lens, input_lens, labels = data
    preds = model(imgs)
    loss = criterion(preds, targets, input_lens, label_lens)
    return loss, val_ds.evaluate(model, imgs.to(device), labels)


def run_handwriting_transcription():
    """运行手写文字识别训练和评估"""
    print("开始手写文字识别任务...")

    # 下载数据
    download_handwriting_data()

    # 准备数据
    vocab = 'QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm'
    B, T, V = 64, 32, len(vocab)

    trn_items, val_items = train_test_split(Glob('synthetic-data'), test_size=0.2, random_state=22)
    trn_ds = OCRDataset(trn_items, vocab=vocab)
    val_ds = OCRDataset(val_items, vocab=vocab)

    trn_dl = DataLoader(trn_ds, batch_size=B, collate_fn=trn_ds.collate_fn, drop_last=True, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=B, collate_fn=val_ds.collate_fn, drop_last=True)

    # 创建模型
    model = Ocr(len(vocab)).to(device)
    criterion = ctc
    optimizer = optim.AdamW(model.parameters(), lr=3e-3)

    # 训练模型
    n_epochs = 5  # 减少epochs以便快速演示
    log = Report(n_epochs)

    for ep in range(n_epochs):
        N = len(trn_dl)
        for ix, data in enumerate(trn_dl):
            pos = ep + (ix + 1) / N
            loss, results = train_ocr_batch(data, model, optimizer, criterion)
            ca, wa = results['char-accuracy'], results['word-accuracy']
            log.record(pos=pos, trn_loss=loss, trn_char_acc=ca, trn_word_acc=wa, end='\r')

        val_results = []
        N = len(val_dl)
        for ix, data in enumerate(val_dl):
            pos = ep + (ix + 1) / N
            loss, results = validate_ocr_batch(data, model, criterion)
            ca, wa = results['char-accuracy'], results['word-accuracy']
            log.record(pos=pos, val_loss=loss, val_char_acc=ca, val_word_acc=wa, end='\r')

        log.report_avgs(ep + 1)
        print(f"Epoch {ep + 1}:")

        # 展示一些预测结果
        for jx in range(3):
            img, label = val_ds.sample()
            _img = torch.Tensor(val_ds.preprocess(img)[None, None]).to(device)
            pred = model(_img)[:, 0, :]
            pred = trn_ds.decoder_chars(pred)
            print(f'Pred: `{pred}` :: Truth: `{label}`')
        print()

    # 绘制训练曲线
    log.plot_epochs(['trn_word_acc', 'val_word_acc'], title='Training and validation word accuracy')

    print("手写文字识别任务完成!")


# ==================== 图像描述生成 (Image Captioning) ====================

def download_captioning_data():
    """下载图像描述数据"""
    if not os.path.exists('open_images_train_captions.jsonl'):
        os.system(
            'wget -q -O open_images_train_captions.jsonl https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl')

    if not os.path.exists('data.csv'):
        with open('open_images_train_captions.jsonl', 'r') as json_file:
            json_list = json_file.read().split('\n')
        np.random.shuffle(json_list)
        data = []
        N = 10000  # 减少数据量以便快速演示
        for ix, json_str in Tqdm(enumerate(json_list), N):
            if ix == N: break
            try:
                result = json.loads(json_str)
                x = pd.DataFrame.from_dict(result, orient='index').T
                data.append(x)
            except:
                pass

        np.random.seed(10)
        data = pd.concat(data)
        data['train'] = np.random.choice([True, False], size=len(data), p=[0.95, 0.05])
        data.to_csv('data.csv', index=False)


def download_captioning_images():
    """下载图像描述图像"""
    from openimages.download import _download_images_by_id

    if not os.path.exists('train-images'):
        os.makedirs('train-images', exist_ok=True)
        data = pd.read_csv('data.csv')
        subset_imageIds = data[data['train']].image_id.tolist()[:100]  # 减少图像数量
        _download_images_by_id(subset_imageIds, 'train', './train-images/')

    if not os.path.exists('val-images'):
        os.makedirs('val-images', exist_ok=True)
        data = pd.read_csv('data.csv')
        subset_imageIds = data[~data['train']].image_id.tolist()[:20]  # 减少图像数量
        _download_images_by_id(subset_imageIds, 'train', './val-images/')


class CaptioningData(Dataset):
    def __init__(self, root, df, vocab):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.vocab = vocab
        self.transform = T.Compose([
            T.Resize(224),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        row = self.df.iloc[index].squeeze()
        id = row.image_id
        image_path = f'{self.root}/{id}.jpg'
        image = Image.open(os.path.join(image_path)).convert('RGB')

        caption = row.caption
        tokens = str(caption).lower().split()
        target = []
        target.append(vocab.stoi['<start>'])
        target.extend([vocab.stoi[token] for token in tokens])
        target.append(vocab.stoi['<end>'])
        target = torch.Tensor(target).long()
        return image, target, caption

    def choose(self):
        return self[np.random.randint(len(self))]

    def __len__(self):
        return len(self.df)

    def collate_fn(self, data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, targets, captions = zip(*data)
        images = torch.stack([self.transform(image) for image in images], 0)
        lengths = [len(tar) for tar in targets]
        _targets = torch.zeros(len(captions), max(lengths)).long()
        for i, tar in enumerate(targets):
            end = lengths[i]
            _targets[i, :end] = tar[:end]
        return images.to(device), _targets.to(device), torch.tensor(lengths).long().to(device)


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(ffeatures))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=80):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True)
        outputs, _ = self.lstm(packed)
        outputs = self.linear(outputs[0])
        return outputs

    def predict(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs: (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)

        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        # convert predicted tokens to strings
        sentences = []
        for sampled_id in sampled_ids:
            sampled_id = sampled_id.cpu().numpy()
            sampled_caption = []
            for word_id in sampled_id:
                word = vocab.itos[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            sentences.append(sentence)
        return sentences


def train_captioning_batch(data, encoder, decoder, optimizer, criterion):
    encoder.train()
    decoder.train()
    images, captions, lengths = data
    images = images.to(device)
    captions = captions.to(device)
    targets = pack_padded_sequence(captions, lengths.cpu(), batch_first=True)[0]
    features = encoder(images)
    outputs = decoder(features, captions, lengths)
    loss = criterion(outputs, targets)
    decoder.zero_grad()
    encoder.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def validate_captioning_batch(data, encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()
    images, captions, lengths = data
    images = images.to(device)
    captions = captions.to(device)
    targets = pack_padded_sequence(captions, lengths.cpu(), batch_first=True)[0]
    features = encoder(images)
    outputs = decoder(features, captions, lengths)
    loss = criterion(outputs, targets)
    return loss


def load_image_and_predict(image_path, encoder, decoder, vocab):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))
    ])

    def load_image(image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)
        if transform is not None:
            tfm_image = transform(image)[None]
        return image, tfm_image

    org_image, tfm_image = load_image(image_path, transform)
    image_tensor = tfm_image.to(device)
    encoder.eval()
    decoder.eval()
    feature = encoder(image_tensor)
    sentence = decoder.predict(feature)[0]
    show(org_image, title=sentence)
    return sentence


def run_image_captioning():
    """运行图像描述生成训练和评估"""
    print("开始图像描述生成任务...")

    # 下载数据
    download_captioning_data()
    download_captioning_images()

    # 准备数据
    data = pd.read_csv('data.csv')

    # 构建词汇表
    from torchtext.data import Field
    captions = Field(sequential=False, init_token='<start>', eos_token='<end>')
    all_captions = data[data['train']]['caption'].tolist()
    all_tokens = [[w.lower() for w in c.split()] for c in all_captions]
    all_tokens = [w for sublist in all_tokens for w in sublist]
    captions.build_vocab(all_tokens)

    class Vocab:
        pass

    vocab = Vocab()
    captions.vocab.itos.insert(0, '<pad>')
    vocab.itos = captions.vocab.itos

    vocab.stoi = defaultdict(lambda: captions.vocab.itos.index('<unk>'))
    vocab.stoi['<pad>'] = 0
    for s, i in captions.vocab.stoi.items():
        vocab.stoi[s] = i + 1

    # 创建数据集
    trn_ds = CaptioningData('train-images', data[data['train']], vocab)
    val_ds = CaptioningData('val-images', data[~data['train']], vocab)

    trn_dl = DataLoader(trn_ds, 32, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, 32, collate_fn=val_ds.collate_fn)

    # 创建模型
    encoder = EncoderCNN(256).to(device)
    decoder = DecoderRNN(256, 512, len(vocab.itos), 1).to(device)
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    # 训练模型
    n_epochs = 3  # 减少epochs以便快速演示
    log = Report(n_epochs)

    for epoch in range(n_epochs):
        if epoch == 2:  # 学习率调整
            optimizer = torch.optim.AdamW(params, lr=1e-4)

        N = len(trn_dl)
        for i, data in enumerate(trn_dl):
            trn_loss = train_captioning_batch(data, encoder, decoder, optimizer, criterion)
            pos = epoch + (1 + i) / N
            log.record(pos=pos, trn_loss=trn_loss, end='\r')

        N = len(val_dl)
        for i, data in enumerate(val_dl):
            val_loss = validate_captioning_batch(data, encoder, decoder, criterion)
            pos = epoch + (1 + i) / N
            log.record(pos=pos, val_loss=val_loss, end='\r')

        log.report_avgs(epoch + 1)

    # 绘制训练曲线
    log.plot_epochs(log=True)

    # 测试一些图像
    files = Glob('val-images')
    for _ in range(3):
        load_image_and_predict(choose(files), encoder, decoder, vocab)

    print("图像描述生成任务完成!")


# ==================== DETR目标检测 ====================

def download_detr_data():
    """下载DETR目标检测数据"""
    if not os.path.exists('open-images-bus-trucks'):
        os.system('wget -q https://www.dropbox.com/s/agmzwk95v96ihic/open-images-bus-trucks.tar.xz')
        os.system('tar -xf open-images-bus-trucks.tar.xz')
        os.system('rm open-images-bus-trucks.tar.xz')
        os.system('git clone https://github.com/sizhky/detr/')

    os.chdir('detr')

    if not os.path.exists('../open-images-bus-trucks/annotations/mini_open_images_train_coco_format.json'):
        os.chdir('../open-images-bus-trucks/annotations')
        os.system('cp mini_open_images_train_coco_format.json instances_train2017.json')
        os.system('cp mini_open_images_val_coco_format.json instances_val2017.json')
        os.chdir('..')
        os.system('ln -s images/ train2017')
        os.system('ln -s images/ val2017')
        os.chdir('../detr')

    if not os.path.exists('detr-r50-e632da11.pth'):
        os.system('wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth')
        checkpoint = torch.load("detr-r50-e632da11.pth", map_location='cpu')
        del checkpoint["model"]["class_embed.weight"]
        del checkpoint["model"]["class_embed.bias"]
        torch.save(checkpoint, "detr-r50_no-class-head.pth")


def train_detr():
    """训练DETR模型"""
    os.system('python main.py --coco_path ../open-images-bus-trucks/ \
              --epochs 5 --lr=1e-4 --batch_size=2 --num_workers=4 \
              --output_dir="outputs" --resume="detr-r50_no-class-head.pth"')


# DETR工具函数
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, transform):
    img = transform(im).unsqueeze(0)
    assert img.shape[-2] <= 1600 and img.shape[
        -1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
    outputs = model(img)
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def plot_results(pil_img, prob, boxes, CLASSES):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def run_object_detection():
    """运行DETR目标检测"""
    print("开始DETR目标检测任务...")

    # 下载数据
    download_detr_data()

    # 训练模型
    train_detr()

    # 加载训练好的模型
    from main import get_args_parser, argparse, build_model
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args, _ = parser.parse_known_args()

    model, _, _ = build_model(args)
    model.load_state_dict(torch.load("outputs/checkpoint.pth")['model'])

    # 定义类别
    CLASSES = ['', 'BUS', 'TRUCK']

    # 定义图像变换
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 测试一些图像
    for _ in range(2):
        image = Image.open(choose(Glob('../open-images-bus-trucks/images/*'))).resize((800, 800)).convert('RGB')
        scores, boxes = detect(image, model, transform)
        plot_results(image, scores, boxes, CLASSES)

    print("DETR目标检测任务完成!")


# ==================== 主程序 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='运行计算机视觉任务')
    parser.add_argument('--task', type=str, choices=['handwriting', 'captioning', 'detection', 'all'],
                        default='all', help='选择要运行的任务')

    args = parser.parse_args()

    if args.task == 'handwriting' or args.task == 'all':
        run_handwriting_transcription()

    if args.task == 'captioning' or args.task == 'all':
        run_image_captioning()

    if args.task == 'detection' or args.task == 'all':
        run_object_detection()
