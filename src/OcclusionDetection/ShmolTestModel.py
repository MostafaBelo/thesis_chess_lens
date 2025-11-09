import torch, cv2
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

IMG_PATH = "7ot makan el soora hena"
MODEL_PATH = "shoof enta 7atet el best_tiny_occlusion_cnn.pt fein we7ot el path hena"


class TinyOcclusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64,2)
        )
    def forward(self,x): return self.net(x)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256


model = TinyOcclusionCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
img = img.to(DEVICE)


with torch.no_grad():
    out = model(img)
    pred = out.argmax(1).item()

classes = ["occlusion", "valid"]
confidence = F.softmax(out, dim=1)[0, pred].item()

plt.imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
plt.title(f"Prediction: {classes[pred]}, Confidence: {confidence:.4f}")
plt.axis("off")
plt.show()
