import torch
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import numpy as np
import pyautogui

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(31616, 128)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

model = MyModel()

#class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
class_names = ["palm", "l"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('handrecognition_model.pth'))
# model = torch.load('handrecognition_model.pth', map_location=device)
model.eval()

capture = cv2.VideoCapture(0)

while capture.isOpened():

    _, frame = capture.read()
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 0), 1)
    crop_image = frame[100:300, 100:300]
    frameCopy = cv2.resize(crop_image, (120, 320))
    gray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray)
    img_array = img_array.reshape(120, 320, 1)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_array)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor.to(device))
    prediction = prediction.cpu().numpy()
    prediction = prediction[0,1:3]
    value = np.argmax(prediction)
    print(class_names[value])
    if class_names[value] == "l":
        pyautogui.press('space')
        cv2.putText(frame, "GO UP!", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()