"""
Augmentation pipelines. Kept deliberately simple/standard rather than
exotic — the proposal's "Image variability" challenge (angle, lighting,
growth stage, image quality) is addressed with the usual crop/flip/
color-jitter combination, which is the right starting point before
reaching for anything fancier.
"""
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),  # plants are photographed at arbitrary angles
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
