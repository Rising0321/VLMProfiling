import PIL.Image
import torch
from simclr import SimCLR
from simclr.modules import get_resnet
from torchvision import transforms

encoder = get_resnet('resnet50', pretrained=False)  # don't load a pre-trained model from PyTorch repo
n_features = encoder.fc.in_features  # get dimensions of fc layer
device = 'cuda:0'
transform = transforms.ToTensor()
# load pre-trained model from checkpoint
simclr_model = SimCLR(encoder=encoder, projection_dim=64, n_features=n_features)
simclr_model.load_state_dict(torch.load("./checkpoint_100.tar", map_location=torch.device('cuda:0')))
simclr_model = simclr_model.to(device)

img = PIL.Image.open('../ResNet/img.png')
img = transform(img).unsqueeze(0).to(device)
print(img.shape)
with torch.no_grad():
    h, _, z, _ = simclr_model(img, img)

emb = h.detach().cpu().numpy()

print(emb)

print(emb.shape)
