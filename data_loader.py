import os
import glob
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ===== 하이퍼파라미터 =====
DATA_DIR    = './data/anime-faces'
IMAGE_SIZE  = 64
BATCH_SIZE  = 128
NUM_WORKERS = 4


# ===== Transform 정의 =====
# GAN은 Generator 마지막이 tanh라 [-1, 1] 정규화가 표준
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

def denormalize(tensor):
    """[-1, 1] → [0, 1] 역정규화 (시각화용)"""
    return tensor * 0.5 + 0.5


# ===== Dataset 클래스 =====
class AnimeFaceDataset(Dataset):
    """
    Kaggle Anime Face Dataset 로더
    - 라벨 없는 기본 버전 (DCGAN용)
    - 추후 Conditional GAN 확장 시 라벨 추가 예정
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform

        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths += glob.glob(
                os.path.join(root_dir, '**', ext), recursive=True
            )

        print(f'총 이미지 수: {len(self.image_paths):,}장')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'이미지 로드 실패: {img_path}, 오류: {e}')
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image


# ===== DataLoader 생성 함수 =====
def get_dataloader():
    dataset = AnimeFaceDataset(root_dir=DATA_DIR, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
        drop_last   = True
    )

    print(f'총 배치 수: {len(dataloader)}')
    return dataloader


# ===== 샘플 시각화 =====
def show_samples(dataloader, n_row=4, n_col=8, save_path='./outputs/sample_images.png'):
    batch = next(iter(dataloader))
    imgs  = denormalize(batch[:n_row * n_col])

    fig, axes = plt.subplots(n_row, n_col, figsize=(16, 8))
    fig.suptitle('Anime Face Dataset 샘플', fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        img = imgs[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'샘플 이미지 저장 완료: {save_path}')


# ===== 데이터 통계 확인 =====
def print_stats(dataloader, n_batches=50):
    print(f'채널별 통계 계산 중 ({n_batches}개 배치)...')

    channel_sum    = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        batch_denorm    = denormalize(batch)
        channel_sum    += batch_denorm.mean(dim=[0, 2, 3])
        channel_sq_sum += (batch_denorm ** 2).mean(dim=[0, 2, 3])

    mean = channel_sum    / n_batches
    std  = (channel_sq_sum / n_batches - mean ** 2).sqrt()

    print(f'채널별 평균 (R, G, B): {mean.numpy().round(3)}')
    print(f'채널별 표준편차 (R, G, B): {std.numpy().round(3)}')


# ===== 메인 실행 =====
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 디바이스: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    dataloader = get_dataloader()

    # 배치 shape 확인
    sample_batch = next(iter(dataloader))
    print(f'\n배치 shape: {sample_batch.shape}')   # [128, 3, 64, 64]
    print(f'픽셀 최솟값: {sample_batch.min():.3f}') # 약 -1.0
    print(f'픽셀 최댓값: {sample_batch.max():.3f}') # 약 +1.0

    show_samples(dataloader)
    print_stats(dataloader)