import pickle
import os
import torch
import argparse
import time
from scrl.models.backbones.visual.resnet import encoder_resnet50
import json
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
from scrl.SceneSeg import run_seg

class MovieNet_SingleShot_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_path, shot_info_path, transform,
        frame_per_shot = 3, _Type='train'):
        self.img_path = img_path
        with open(shot_info_path, 'rb') as f:
            self.shot_info = json.load(f)
        self.img_path = img_path
        self.frame_per_shot = frame_per_shot
        self.transform = transform
        self._Type = _Type.lower()
        assert self._Type in ['train','val','test']
        self.idx_imdb_map = {}
        data_length = 0
        for info in self.shot_info[_Type]:
            imdb = info['name']
            for shot in info['label']:
                self.idx_imdb_map[data_length] = (imdb, shot[0], shot[1])
                data_length += 1

    def __len__(self):
        return len(self.idx_imdb_map.keys())

    def _process(self, idx):
        imdb, _id, label = self.idx_imdb_map[idx]  # _id表示第几个镜头， label表示是否到达场景边界
        img_path_0 =  f'{self.img_path}/{imdb}/shot_{_id}_img_0.jpg'
        img_path_1 =  f'{self.img_path}/{imdb}/shot_{_id}_img_1.jpg'
        img_path_2 =  f'{self.img_path}/{imdb}/shot_{_id}_img_2.jpg'
        img_0 = cv2.cvtColor(cv2.imread(img_path_0), cv2.COLOR_BGR2RGB)
        img_1 = cv2.cvtColor(cv2.imread(img_path_1), cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(cv2.imread(img_path_2), cv2.COLOR_BGR2RGB)
        data_0 = self.transform(img_0)
        data_1 = self.transform(img_1)
        data_2 = self.transform(img_2)
        data = torch.cat([data_0, data_1, data_2], axis=0)
        label = int(label)
        # According to LGSS[1]
        # [1] https://arxiv.org/abs/2004.02678
        if label == -1:
            label = 1
        return data, label, (imdb, _id)


    def __getitem__(self, idx):
        return self._process(idx)

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, video_list, transform, frame_per_shot = 3):
        # video_list = [os.path.join(video_dir, name) for name in os.listdir(video_dir)]
        self.video_list = sorted(video_list)
        self.frame_per_shot = frame_per_shot
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def get_fast_video_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        
        # 提速关键：关闭缓冲，强制精准跳转
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = total // 2
        if total < 7:
            start_id = mid
            mid_id = mid
            last_id = mid
        elif total < 30:
            start_id = 2
            mid_id = mid
            last_id = mid+1
        else:
            start_id = 2
            mid_id = mid
            last_id = total - 10
        print("last_id: ", total, last_id)
        # 第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_id)
        _, first = cap.read()

        # 中间帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_id)
        _, middle = cap.read()

        # 最后一帧（安全跳转）
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_id)
        _, last = cap.read()

        cap.release()
        return first, middle, last  # 全部是 numpy 数组

    def _process(self, idx):
        # image_dir = self.image_dir_list[idx]
        # img_path_0 =  os.path.join(image_dir, "first.jpg")
        # img_path_1 =  os.path.join(image_dir, "middle.jpg")
        # img_path_2 =  os.path.join(image_dir, "last.jpg")
        # print("img_path_0: ", img_path_0)
        video_path = self.video_list[idx]
        first_img, middle_img, last_img = self.get_fast_video_frames(video_path)
        img_0 = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
        img_1 = cv2.cvtColor(middle_img, cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB)
        data_0 = self.transform(img_0)
        data_1 = self.transform(img_1)
        data_2 = self.transform(img_2)
        data = torch.cat([data_0, data_1, data_2], axis=0)
        return data

    def __getitem__(self, idx):
        return self._process(idx)

def get_loader(video_list):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    dataset = Custom_Dataset(
        video_list = video_list,
        transform = _transform,
        frame_per_shot = 3)

    loader = DataLoader(
        dataset, batch_size=64,  drop_last=False,
        shuffle=False, num_workers=0, pin_memory=True
    )
    return loader

def get_encoder(model_name='resnet50', weight_path='', input_channel=9):
    encoder = None
    model_name = model_name.lower()
    if model_name == 'resnet50':
        encoder = encoder_resnet50(weight_path='',input_channel=input_channel)
        model_weight = torch.load(weight_path,map_location=torch.device('cpu'))['state_dict']
        pretrained_dict = {}
        for k, v in model_weight.items():
            # moco loading 
            if k.startswith('module.encoder_k'):
                continue
            if k == 'module.queue' or k == 'module.queue_ptr':
                continue
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                k = k[17:]

            pretrained_dict[k] = v
        encoder.load_state_dict(pretrained_dict, strict = False)
        print(f'loaded from {weight_path}')
    return encoder


@torch.no_grad()
def get_save_embeddings(model, loader, shot_num, filename, log_interval=100):
    # dict
    # key: index, value: [(embeddings, label), ...]
    embeddings = {} 
    model.eval()
    
    print(f'total length of dataset: {len(loader.dataset)}')
    print(f'total length of loader: {len(loader)}')
    
    embeddings["data"] = []
    for batch_idx, data in enumerate(loader):
        if batch_idx % log_interval == 0:
            print(f'processed: {batch_idx}')

        data = data.cuda(non_blocking=True) # ([bs, shot_num, 9, 224, 224])
        data = data.view(-1, 9, 224, 224)
        # print("data_name: ", data_name)
        # target = target.view(-1).cuda()
        output = model(data, False)   # ([bs * shot_num, 2048])
        for i in range(output.shape[0]):
            embeddings["data"].append((output[i:i+1].cpu().numpy(), 0))

        # embeddings["data"] = []
        # t_emb = output[0:shot_num].cpu().numpy()
        # t_label =  target[i].cpu().numpy()
        # embeddings[key].append((t_emb.copy() ,t_label.copy()))
    pickle.dump(embeddings, open(filename, 'wb'))


def run_scrl(scrl_encoder, scrl_BiLSTM, video_list, label_dir):
    encoder = scrl_encoder
    loader = get_loader(video_list)
    filename = os.path.join(label_dir, 'tmp_embedding.pkl')
    get_save_embeddings(
        encoder, 
        loader, 
        1, 
        filename, 
        log_interval=100
    )
    seg_result = run_seg.inference_only(16, 2048, scrl_BiLSTM, filename)
    print(seg_result)
    os.remove(filename)
    return seg_result


def to_log(cfg, content, echo=True):
    with open(cfg.log_file, 'a') as f:
        f.writelines(content+'\n')
    if echo: print(content)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/checkpoint_0099.pth.tar')
    parser.add_argument('--image_dir', type=str, default='./data/zhouchuchusanhai/images/')
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--frame_per_shot', type=int, default=3)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--worker_num', type=int, default=16)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='./embeddings/')
    parser.add_argument('--gpu-id', type=str, default='0')
    cfg = parser.parse_args()

    # select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id

    return cfg


if __name__ == '__main__':
    cfg = get_config()
    # extract_features(cfg)