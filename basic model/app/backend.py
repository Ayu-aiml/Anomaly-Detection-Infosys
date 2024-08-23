import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Transformation for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ensure images are resized to 128x128
    transforms.ToTensor()
])

# Load Faster R-CNN model
def load_fasterrcnn_model():
    num_classes = 2  # Adjust based on your use case (1 class + background)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load("app/fasterrcnn_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Load CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjusted to match the dimensions used during training
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if x.size(2) > 1 and x.size(3) > 1:  # Check if pooling is possible
            x = self.pool(x)
        x = F.relu(self.conv2(x))
        if x.size(2) > 1 and x.size(3) > 1:  # Check if pooling is possible
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_cnn_model():
    model = CNNModel()
    model.load_state_dict(torch.load("app/cnn.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Load MLP Model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(128*128*3, 128)  # Adjust based on the correct input size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_mlp_model():
    model = SimpleMLP()
    model.load_state_dict(torch.load("app/simplemlp.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Initialize models
fasterrcnn_model = load_fasterrcnn_model()
cnn_model = load_cnn_model()
mlp_model = load_mlp_model()

# Helper function to process image from request
def preprocess_image(image_bytes, target_size=(128, 128)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_tensor = transform(image)
    return image_tensor

# Define Flask route for anomaly detection
@app.route('/detect', methods=['POST'])
def detect_anomalies():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    model_name = request.form.get('model', 'fasterrcnn')

    try:
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes).to(device)

        if model_name == 'fasterrcnn':
            outputs = fasterrcnn_model([image_tensor])
            boxes = outputs[0]['boxes'].detach().cpu().numpy().tolist()
            labels = outputs[0]['labels'].detach().cpu().numpy().tolist()
            scores = outputs[0]['scores'].detach().cpu().numpy().tolist()

            # Map COCO labels to names
            coco_labels = {1: 'person', 2: 'bicycle', 3: 'car', 8: 'truck', 10: 'traffic light', 12: 'stop sign', 16: 'dog', 18: 'cat'}
            detected_objects = [coco_labels.get(label, 'unknown') for label in labels]  # Remove .item() usage
            
            # Return JSON response with structured results
            return jsonify({
                'boxes': boxes,
                'labels': detected_objects,
                'scores': scores
            })

        elif model_name == 'cnn':
            outputs = cnn_model(image_tensor.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            result = 'Anomaly' if predicted.item() == 1 else 'Normal'
            return jsonify({'result': result})

        elif model_name == 'mlp':
            outputs = mlp_model(image_tensor.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            result = 'Anomaly' if predicted.item() == 1 else 'Normal'
            return jsonify({'result': result})

        else:
            return jsonify({'error': f'Invalid model name: {model_name}'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
