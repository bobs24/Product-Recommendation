import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np


## ResNet50
# class FeatureExtractor:
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Load pretrained ResNet50 without the final classification layer
#         resnet = models.resnet50(pretrained=True)
#         # Remove the final fully connected layer (fc)
#         self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
#         self.model.eval().to(self.device)

#         # Standard ImageNet preprocessing
#         self.transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             ),
#         ])

#     def extract(self, image: Image.Image):
#         image = self.transform(image).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             features = self.model(image)
#         features = features.squeeze().cpu().numpy()
#         features = features.reshape(-1)  # flatten (2048,)

#         # Normalize to unit vector (important for cosine similarity)
#         norm = np.linalg.norm(features)
#         if norm > 0:
#             features = features / norm
#         return features

## ConvNext-Tiny
class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained ConvNeXt Tiny
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        convnext = models.convnext_tiny(weights=weights)
        # Remove classification head (last layer)
        self.model = torch.nn.Sequential(*list(convnext.children())[:-1])
        self.model.eval().to(self.device)

        # Use official preprocessing transform for ConvNeXt Tiny
        self.transform = weights.transforms()

    def extract(self, image: Image.Image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image)
        features = features.squeeze().cpu().numpy()
        features = features.reshape(-1)  # flatten
        
        # Normalize to unit vector (important for cosine similarity)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
