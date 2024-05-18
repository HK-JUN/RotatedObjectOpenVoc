#DataParallel
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import os
import glob
import gc
import matplotlib.pyplot as plt
import numpy as np

from transformers import CLIPProcessor, CLIPModel
import torchvision.ops as ops


torch.cuda.empty_cache()
gc.collect()
# 데이터셋 경로 설정
train_img_dir = '/home/jhpark/my_dota/DOTA_devkit/splitted_data/train/images'
train_ann_dir = '/home/jhpark/my_dota/DOTA_devkit/splitted_data/train/labelTxt'
eval_img_dir = '/home/jhpark/my_dota/DOTA_devkit/splitted_data/val/images'
eval_ann_dir = '/home/jhpark/my_dota/DOTA_devkit/splitted_data/val/labelTxt'

# 클래스 이름을 인덱스로 매핑
class_names_to_index = {
    'plane': 0, 'ship': 1, 'storage-tank': 2, 'baseball-diamond': 3,
    'tennis-court': 4, 'basketball-court': 5, 'ground-track-field': 6,
    'harbor': 7, 'bridge': 8, 'large-vehicle': 9, 'small-vehicle': 10,
    'helicopter': 11, 'roundabout': 12, 'soccer-ball-field': 13, 'swimming-pool': 14
}

# 클래스 인덱스를 이름으로 매핑
index_to_class_names = {v: k for k, v in class_names_to_index.items()}

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    obbs = [item[1] for item in batch]
    class_indexes = [item[2] for item in batch]
    text_prompts = [item[3] for item in batch]

    # 이미지는 스택을 사용하여 배치를 생성합니다.
    images = torch.stack(images, dim=0)

    # obbs와 class_indexes에 대해서는 패딩을 적용합니다.
    # pad_sequence를 사용하여 각 리스트의 텐서들을 패딩합니다. 배치 내의 가장 긴 텐서에 맞춰집니다.
    obbs = pad_sequence(obbs, batch_first=True, padding_value=0)
    class_indexes = pad_sequence(class_indexes, batch_first=True, padding_value=-1)  # -1은 무시될 클래스 인덱스로 가정

    return images, obbs, class_indexes, text_prompts

# 어노테이션 파일 읽기
def read_annotations(ann_path):
    annotations = []
    with open(ann_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            if len(parts) >= 9:  # OBB + category
                obb = list(map(float, parts[:8]))
                category = parts[8]
                if category in class_names_to_index:
                    annotations.append((obb, class_names_to_index[category]))
    return annotations

# DOTA 데이터셋 클래스
class DOTADataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))  # Adjust as needed
        self.ann_files = [os.path.join(ann_dir, os.path.splitext(os.path.basename(x))[0] + '.txt') for x in self.img_files]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]
        image = Image.open(img_path).convert('RGB')
        print("img_path",img_path)
        print("anno_path",ann_path)
        annotations = read_annotations(ann_path)
        text_prompts = []
        if not annotations:  # 어노테이션이 없는 경우 처리
            obbs = torch.zeros((0, 8), dtype=torch.float)
            class_indexes = torch.zeros((0,), dtype=torch.long)
        else:
            obbs, class_indexes = zip(*annotations)
            obbs = torch.tensor(obbs, dtype=torch.float)
            class_indexes = torch.tensor(class_indexes, dtype=torch.long)
            text_prompts = [generate_prompts(index_to_class_names[index.item()]) for index in class_indexes]
        
        #for index in class_indexes:
        #    class_name = index_to_class_names[index]
        #    prompts_for_class = generate_prompts(class_name)
        #    text_prompts.extend(prompts_for_class)
        #text_prompts = [index_to_class_names[idx] for idx in class_indexes]  # Convert class indexes to class names
        #text_prompts = [generate_prompts(class_name) for class_name in text_prompts]  # Generate text prompts
        #text_prompts = [item for sublist in text_prompts for item in sublist]  # Flatten list of lists

        if self.transform:
            image = self.transform(image)
            
        print(f"[debug]dataset_getitem:Image shape after transform: {image.size()}")
        return image, obbs, class_indexes, text_prompts

# 텍스트 프롬프트 생성 함수
def generate_prompts(category):
    templates = [f'There is {category} in the scene.',
   f'There is the {category} in the scene.',
   f'a photo of {category} in the scene.',
   f'a photo of the {category} in the scene.',
   f'a photo of one {category} in the scene.',
   f'itap of {category}.',
   f'itap of my {category}.',
   f'itap of the {category}.',
   f'a photo of {category}.',
   f'a photo of my {category}.',
   f'a photo of the {category}.',
   f'a photo of one {category}.',
   f'a photo of many {category}.',
   f'a good photo of {category}.',
   f'a good photo of the {category}.',
   f'a bad photo of {category}.',
   f'a bad photo of the {category}.',
   f'a photo of a nice {category}.',
   f'a photo of the nice {category}.',
   f'a photo of a cool {category}.',
   f'a photo of the cool {category}.',
   f'a photo of a weird {category}.',
   f'a photo of the weird {category}.',
   f'a photo of a small {category}.',
   f'a photo of the small {category}.',
   f'a photo of a large {category}.',
   f'a photo of the large {category}.',
   f'a photo of a clean {category}.',
   f'a photo of the clean {category}.',
   f'a photo of a dirty {category}.',
   f'a photo of the dirty {category}.',
   f'a bright photo of {category}.',
   f'a bright photo of the {category}.',
   f'a dark photo of {category}.',
   f'a dark photo of the {category}.',
   f'a photo of a hard to see {category}.',
   f'a photo of the hard to see {category}.',
   f'a low resolution photo of {category}.',
   f'a low resolution photo of the {category}.'
    ]
    return templates


# RoI Transformer Components
class RRoILearner(nn.Module): #5차원 벡터는 [x_center, y_center, width, height, angle] 형태로, RRoI의 중심 위치, 너비, 높이, 회전 각도
    def __init__(self, input_dim=2048, output_dim=5,num_rois=100):
        super(RRoILearner, self).__init__()
        self.num_rois = num_rois
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(input_dim, output_dim * num_rois)

    def forward(self, x):
         # x는 특성 맵: [배치 크기, 채널 수, 높이, 너비]
        batch_size = x.size(0)
        print(f"batch:{batch_size}")
        x = self.global_avg_pool(x)  # 전역 평균 풀링: [배치 크기, 채널 수, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # 결과를 [batch_size, num_rois, output_dim]으로 재구성
        return x.view(batch_size, self.num_rois, -1)

class RotatedPSRoIAlign(nn.Module):
    def __init__(self, output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2):
        super(RotatedPSRoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        # self.ps_roi_align = ops.RoIAlign(output_size=output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)  # 기존 RoIAlign은 사용하지 않음.

    def forward(self, feature_maps, rois):
        # 'rois'는 [batch_idx, x_center, y_center, width, height, angle] 형식으로 가정합니다.feature_maps, roi
        aligned_features = []
        for roi in rois:
            batch_idx, x_center, y_center, width, height, angle = roi
            # 각도를 라디안으로 변환
            theta = torch.tensor([
                [torch.cos(angle), -torch.sin(angle), x_center],
                [torch.sin(angle), torch.cos(angle), y_center]
            ], dtype=torch.float, device=feature_maps.device)
            
            # Affine 그리드 생성 및 샘플링
            grid = F.affine_grid(theta.unsqueeze(0), [1, 1, *self.output_size], align_corners=True)
            sampled = F.grid_sample(feature_maps[batch_idx.long()].unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            aligned_features.append(sampled.squeeze(0))

        return torch.stack(aligned_features)
    
    def align_rotated_roi(self, feature_maps, roi): #사용 안하면 추후에 지워야함
        batch_idx, x_center, y_center, width, height, angle = roi
        feature_map = feature_maps[batch_idx.long()]

        # 각도를 라디안으로 변환 및 Affine 변환 매트릭스 구성
        angle_rad = -angle * (np.pi / 180)
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
        affine_matrix = torch.tensor([[[cos_a, -sin_a, 0], 
                                        [sin_a, cos_a, 0]]], 
                                      dtype=torch.float, device=feature_map.device)
        # size 인자를 올바르게 수정합니다. 목표 출력 크기는 (N, C, H, W) 여야 합니다.
        size = torch.Size([1, feature_map.size(0), *self.output_size])
        # Grid 생성 및 샘플링
        grid = F.affine_grid(affine_matrix, size=size, align_corners=False)
        sampled = F.grid_sample(feature_map.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return sampled.squeeze(0)

# ResNet50 Backbone 정의 (마지막 fc 제거)
class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet50 모델 로드
        original_model = models.resnet50(pretrained=True)
        # 마지막 layer 제거
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x

# Distillation Integrated Model
class DistillationIntegratedModel(nn.Module):
    def __init__(self):
        super(DistillationIntegratedModel, self).__init__()
        self.backbone = ResNet50Backbone()  # ResNet50 백본 사용
        #self.backbone.fc = nn.Identity()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        self.rroi_learner = RRoILearner(input_dim=2048, output_dim=5,num_rois=100)
        self.rotated_ps_roi_align = RotatedPSRoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)
        #self.classifier의 입력 차원 수정
        self.text_feature_dim = self.clip_model.config.text_config.hidden_size
        combined_feature_dim = 2048 + self.text_feature_dim
        #combined_feature_dim = 41984 #foward의 combined_feature_dim 이 바뀌면 몇인지 설정해줘야함
        self.classifier = nn.Linear(combined_feature_dim, len(class_names_to_index))
        self.visual_to_text_dim = nn.Linear(2048, 512) 
        self.distillation_temperature = 5.0
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, images, obbs, text_prompts_list,class_indexes):
        print("[DEBUG]model forward enter")
        print(f"image shape:{images.shape}")
        visual_features = self.backbone(images)
        
        #rois = obb_to_roi(obbs)
        print("After Backbone:", visual_features.shape) #torch.Size([1, 2048, 7, 7])
        # RRoILearner에서 회전된 RoI 파라미터를 예측합니다.
        transformed_rois = self.rroi_learner(visual_features)
        print("Transformed RoIs:", transformed_rois.shape) #torch.Size([1, 100, 5])
        roi_counts = transformed_rois.size(1)  # 각 이미지당 ROI 수
        batch_size = images.size(0)
        class_indexes_expanded = []
        print(f"class_indexes example:{class_indexes}")
        for i in range(batch_size):
            repeat_count = roi_counts  # 이 예제에서는 모든 이미지에 대해 동일한 ROI 수를 가정
            expanded_indexes = class_indexes[i].repeat(repeat_count)
            class_indexes_expanded.append(expanded_indexes)
        class_indexes_expanded = torch.cat(class_indexes_expanded)
        print("shape of class_indexes:", class_indexes.shape)  #torch.Size([1, 19]
        print("shape of class_indexes_expanded:", class_indexes_expanded.shape) #torch.Size([1900]
        # 회전된 RoI 파라미터를 RotatedPSRoIAlign에 전달합니다.
        # 여기서는 obbs (회전된 RoI 정보)와 함께 transformed_rois (회전된 RoI 파라미터)를 전달해야 합니다.
        # 주의: transformed_rois와 obbs의 형태를 확인하고 필요에 따라 조정해야 합니다.
        batch_indices = torch.arange(images.size(0), device=images.device) #torch.Size([1])
        print("Batch indices1:", batch_indices.shape)
        batch_indices = batch_indices.repeat_interleave(transformed_rois.size(1))#torch.Size([100])
        batch_indices = batch_indices.unsqueeze(1) #[100,1]
        
        transformed_rois = transformed_rois.squeeze(0) #[100,5]
        print("Batch indices2:", batch_indices.shape)
        print(f"batch_indices:{batch_indices.shape},transformed_rois:{transformed_rois.shape}")
        rois_with_batch_indices = torch.cat((batch_indices, transformed_rois), dim=1)
        print("rois_with_batch_indices shape:", rois_with_batch_indices.shape)
        print("[DEBUG]roi indices finish")
        pooled_visual_features = self.rotated_ps_roi_align(visual_features, rois_with_batch_indices) #feature_maps, roi
        print("After RotatedPSRoIAlign:", pooled_visual_features.shape)
        # [batch_size * num_rois, 2048, 7, 7] -> [batch_size * num_rois, 2048]
        pooled_visual_features_avg = pooled_visual_features.mean([2, 3])
        # pooled_visual_features의 평균 계산 (가정: 모든 ROI가 동일한 이미지에 속함)
        #pooled_visual_features_avg = pooled_visual_features.view(-1, 100, 2048).mean(dim=1)
        print("pooled_visual_features_avg shape:", pooled_visual_features_avg.shape) #torch.Size([100, 2048])
        # pooled_visual_features_flat이 최종적으로 [batch_size * num_rois_per_image, 2048] 크기를 가지도록 조정
        pooled_visual_features_flat = pooled_visual_features_avg.view(batch_size * roi_counts, -1) #torch.Size([100, 2048])
        
        all_text_features = []
        for text_prompts in text_prompts_list:
            text_features_list = []
            for prompt in text_prompts:
                text_inputs = self.processor(text=prompt, return_tensors="pt", padding=True, truncation=True).to(images.device)
                text_features = self.clip_model.get_text_features(**text_inputs)
                # 각 텍스트 특성의 차원을 확인하고, 필요한 경우 수정
                if text_features.dim() == 1:
                    text_features = text_features.unsqueeze(0)  # 차원 확장
                if text_features.nelement() != 0:
                    text_features_list.append(text_features)
                else:
                    # 텍스트 특성이 비어있는 경우 해당 특성 차원에 맞는 0 벡터 추가
                    text_features_list.append(torch.zeros((1, self.text_feature_dim), device=images.device))

            # 텍스트 특성 리스트를 평균 내어 하나의 평균 특성 텐서 생성
            if text_features_list:
                text_features_avg = torch.mean(torch.cat(text_features_list, dim=0), dim=0).unsqueeze(0)
            else:
                # 모든 프롬프트에 대해 특성이 비어 있으면 0 벡터 추가
                text_features_avg = torch.zeros((1, self.text_feature_dim), device=images.device)
            all_text_features.append(text_features_avg)
        # 모든 이미지에 대한 텍스트 특성들을 결합
        all_text_features_flat = torch.cat(all_text_features, dim=0)

        
        for i,tmp in enumerate(all_text_features):
            print(f"{i}:{tmp.shape}")
        all_text_features_flat = torch.cat(all_text_features).view(-1, self.text_feature_dim)  # 평탄화
        #all_text_features = all_text_features.view(images.size(0), -1)  # 필요시 차원 조정
        print(f" all_text_features_flat shape:{all_text_features_flat.shape}")
        print("all_text_features_flat shape before squeeze:", all_text_features_flat.shape) #torch.Size([78, 512])
        #all_text_features = all_text_features.squeeze()
        #print("all_text_features shape after squeeze:", all_text_features.shape)
        num_images = images.size(0)  # 배치 크기
        num_features_per_image = all_text_features_flat.size(0) //num_images
        all_text_features_flat_avg = all_text_features_flat.view(num_images, num_features_per_image, -1).mean(dim=1)
        print(f" all_text_features_flat_avg shape:{all_text_features_flat_avg.shape}") #torch.Size([1, 512])

        all_text_features_flat_avg_expanded = all_text_features_flat_avg.repeat(pooled_visual_features_flat.size(0), 1) #여기서 pooled_visual_features_avg[200,2048]요구 all_text_features_flat_avg 은[1,512]
        print(f"expended text_features:{all_text_features_flat_avg_expanded.shape}")#torch.Size([49, 512])
        combined_features = torch.cat((pooled_visual_features_flat, all_text_features_flat_avg_expanded), dim=1)
        print("Combined features shape:", combined_features.shape)#torch.Size([49, 2560])
        #self.visual_to_text_dim = nn.Linear(2048, all_text_features_flat.shape[-1]).to(images.device)
        logits = self.classifier(combined_features)
        
        return logits, pooled_visual_features_flat, all_text_features_flat,class_indexes_expanded

    def calculate_distillation_loss(self, visual_features, text_features):
        visual_norm = torch.nn.functional.normalize(visual_features, p=2, dim=1) #[200, 2048]
        text_norm = torch.nn.functional.normalize(text_features, p=2, dim=1) #[156,512]
        print(f"visual_norm: {visual_norm.shape},text_norm:{text_norm.shape}")
        # Transform visual features to match text feature dimensions
        visual_transformed = self.visual_to_text_dim(visual_norm)
        sim_matrix = torch.mm(visual_transformed, text_norm.T) / self.distillation_temperature
        loss = self.distillation_loss(torch.nn.functional.log_softmax(sim_matrix, dim=1),
                                      torch.nn.functional.softmax(sim_matrix, dim=1))
        return loss


def calculate_iou(pred_box, gt_box):
    x1_inter = max(pred_box[0], gt_box[0])
    y1_inter = max(pred_box[1], gt_box[1])
    x2_inter = min(pred_box[2], gt_box[2])
    y2_inter = min(pred_box[3], gt_box[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    
    union_area = pred_area + gt_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_mAP(preds, gts, iou_thresh=0.5):
    class_preds = {}
    class_gts = {}

    # 클래스 별로 예측값과 어노테이션을 분류합니다.
    for pred in preds:
        img_id, _, pred_box = pred[0], pred[1], pred[2:]
        class_id = img_id.split('_')[0]  # 클래스 ID는 이미지 ID의 첫 부분으로 가정합니다.
        if class_id not in class_preds:
            class_preds[class_id] = []
        class_preds[class_id].append((img_id, _) + pred_box)

    for gt in gts:
        img_id, gt_box = gt[0], gt[1:]
        class_id = img_id.split('_')[0]
        if class_id not in class_gts:
            class_gts[class_id] = []
        class_gts[class_id].append((img_id,) + gt_box)

    # 각 클래스 별 AP를 계산하고 리스트에 저장합니다.
    aps = []
    for class_id in class_preds.keys():
        if class_id in class_gts:
            ap = calculate_ap(class_preds[class_id], class_gts[class_id], iou_thresh)
            aps.append(ap)

    # AP의 평균을 계산하여 mAP를 반환합니다.
    mAP = np.mean(aps)
    return mAP

def calculate_ap(preds, gts, iou_thresh):
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    for i, pred in enumerate(preds_sorted):
        img_id, _, pred_box = pred[0], pred[1], pred[2:]
        matched = False
        for gt in gts:
            iou = calculate_iou(pred_box, gt[1:])
            if iou >= iou_thresh:
                matched = True
                break
        if matched:
            tp[i] = 1
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / len(gts)

    ap = np.trapz(precisions, recalls)
    return ap

def evaluate_model(model, eval_loader, device):
    model.eval()
    preds = []
    gts = []
    for images, _, _, text_prompts in eval_loader:
        images = images.to(device)
        
        # 모델 예측
        with torch.no_grad():
            logits, _, _ = model(images)
        
        # 모델의 출력을 바운딩 박스로 변환하여 preds 리스트에 추가
        for i in range(len(images)):
            # 예측된 클래스 ID 추출
            pred_class_index = torch.argmax(logits[i]).item()
            
            # 예측된 바운딩 박스를 추출
            img_id = text_prompts[i].split('_')[0]  # 이미지 ID는 어노테이션 파일명의 첫 부분
            x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = text_prompts[i].split('_')[1:]
            pred_box = [float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]
            
            # preds 리스트에 추가
            preds.append((img_id, pred_class_index, *pred_box))
        
        # 정답 바운딩 박스 추출하여 gts 리스트에 추가
        for text_prompt in text_prompts:
            img_id = text_prompt.split('_')[0]  # 이미지 ID는 어노테이션 파일명의 첫 부분
            x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = text_prompt.split('_')[1:]
            gt_box = [float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]
            
            # gts 리스트에 추가
            gts.append((img_id, *gt_box))
    
    # mAP 계산
    mAP = calculate_mAP(preds, gts, iou_thresh=0.5)
    return mAP


def save_model(model, path='/home/jhpark/my_dota/DOTA_devkitss/saved_model'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# Main Training Loop
def main():
    BATCH_SIZE = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    # 1,2 로하면 67gpu
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = DOTADataset(train_img_dir, train_ann_dir, transform=transform)
    eval_dataset = DOTADataset(eval_img_dir, eval_ann_dir, transform=transform)
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistillationIntegratedModel().to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    losses = []
    mAPs = []

    for epoch in range(10):
        model.train()
        epoch_losses = []
        for images, obbs, class_indexes, text_prompts in train_loader:
            print(f"epoch {epoch} inner for start")
            images, obbs, class_indexes = images.to(device), obbs.to(device), class_indexes.to(device)
            print(f"epoch {epoch} [tag1]")
            optimizer.zero_grad()
            print(f"epoch {epoch} [tag2]")
            print(f"[debug]main_epoch images shape:{images.shape}")
            #logits, visual_features, text_features = model(images, obbs, text_prompts)
            
            logits, pooled_visual_features, all_text_features,class_indexes_expanded = model(images, obbs, text_prompts,class_indexes)
            print(f"class_indexes: sss :{class_indexes_expanded}")
            print("logit,classindex size: ",logits.shape,class_indexes_expanded.shape) #torch.Size([98, 15]) torch.Size([3800])
            # 가정: 각 객체에 대해 200개의 RoI를 예측하고, 각 RoI에 대한 로그이트가 필요
            repeat_factor = class_indexes_expanded.shape[0] // logits.shape[0]  # 3800 / 200 = 19
            expanded_logits = logits.repeat_interleave(repeat_factor, dim=0)
            print("expanded_logits size: ",expanded_logits.shape) 
            classification_loss = criterion(expanded_logits, class_indexes_expanded)  # logits와 class_indexes가 맞도록 조정
            print(f"epoch {epoch} [tag4]")
            #distillation_loss = model.calculate_distillation_loss(visual_features, text_features)
            distillation_loss = model.module.calculate_distillation_loss(pooled_visual_features, all_text_features)
            print(f"epoch {epoch} [tag5]")
            total_loss = classification_loss + distillation_loss
            
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())
            print(f"epoch {epoch} inner for end")
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)

        # 에폭의 mAP 계산 (여기서는 단순히 예시 값 사용)
        epoch_mAP = evaluate_model(model,eval_loader,device)
        mAPs.append(epoch_mAP)
        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss}, mAP: {epoch_mAP}")

    #recall, precision, ap = voc_eval(preds, gts, iou_thresh=0.5)
    #print(f"Recall: {recall}, Precision: {precision}, AP: {ap}")
    #mAP = calculate_mAP(preds, gts)
    #print(f"Mean Average Precision (mAP): {mAP}")
    save_model(model,'/home/jhpark/my_dota/DOTA_devkitss/saved_model')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mAPs, label='mAP')
    plt.title('Mean Average Precision (mAP)')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()

    plt.show()
if __name__ == "__main__":
    main()
    