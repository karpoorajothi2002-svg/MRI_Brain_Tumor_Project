from flask import Flask, render_template, request, redirect, url_for, flash
import os
import uuid
import json
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


# =========================================================
# FLASK APP CONFIGURATION
# =========================================================
app = Flask(__name__)
app.secret_key = "brain_tumor_project_secret_key"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH = "brain_tumor_resnet18_model.pth"

UPLOAD_FOLDER = os.path.join("static", "uploads")
HEATMAP_FOLDER = os.path.join("static", "heatmaps")
HISTORY_FILE = "history.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"] = HEATMAP_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)


# =========================================================
# CBAM MODULE
# =========================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attention_input = torch.cat([avg_out, max_out], dim=1)

        return self.sigmoid(self.conv(attention_input))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# =========================================================
# RESNET18 + CBAM MODEL
# IMPORTANT: This architecture must exactly match training.
# =========================================================
class ResNet18CBAM(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        base_model = models.resnet18(weights=None)

        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

        self.cbam = CBAM(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =========================================================
# GRAD-CAM
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, class_index):
        self.model.eval()

        output = self.model(image_tensor)
        self.model.zero_grad()

        score = output[0, class_index]
        score.backward()

        gradients = self.gradients[0]      # [channels, height, width]
        activations = self.activations[0]  # [channels, height, width]

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device
        )

        for channel_index, weight in enumerate(weights):
            cam += weight * activations[channel_index]

        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))

        cam_min = np.min(cam)
        cam_max = np.max(cam)

        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def create_gradcam_image(original_image_path, cam, output_path):
    original_image = cv2.imread(original_image_path)

    if original_image is None:
        raise ValueError("Could not read uploaded MRI image for Grad-CAM.")

    original_image = cv2.resize(original_image, (224, 224))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Original MRI remains visible; colored map is only an overlay.
    overlay = cv2.addWeighted(original_image, 0.68, heatmap, 0.32, 0)

    cv2.imwrite(output_path, overlay)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================
model = ResNet18CBAM(num_classes=4)

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Last convolution layer in custom ResNet18CBAM model
    target_layer = model.features[7][-1].conv2
    grad_cam = GradCAM(model, target_layer)

    MODEL_LOADED = True
    print("MODEL LOADED SUCCESSFULLY")

except Exception as error:
    print("MODEL LOAD ERROR:", error)
    MODEL_LOADED = False
    grad_cam = None


# =========================================================
# IMAGE TRANSFORM
# Must match model training transform.
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def format_class_name(name):
    class_labels = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No Tumor", "pituitary": "Pituitary"}
    return class_labels.get(name.lower(), name.title())

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            if not isinstance(data, list):
                return []

            # Backward compatibility: old scans do not have Grad-CAM fields.
            for item in data:
                if isinstance(item, dict):
                    item.setdefault("id", uuid.uuid4().hex)
                    item.setdefault("heatmap_file", "")
                    item.setdefault("probabilities", {})
                    item.setdefault("image_file", "")
            return data
    except (json.JSONDecodeError, OSError):
        return []

def save_history_item(item):
    history = load_history()
    history.insert(0, item)
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history[:50], file, indent=4)


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["GET", "POST"])
def scan():
    if request.method == "GET":
        return render_template("scan.html")

    if not MODEL_LOADED:
        flash("Model file could not be loaded. Check .pth file path and model architecture.")
        return redirect(url_for("scan"))

    if "image" not in request.files:
        flash("Please select an MRI image.")
        return redirect(url_for("scan"))

    file = request.files["image"]

    if file.filename == "":
        flash("Please select an MRI image.")
        return redirect(url_for("scan"))

    if not allowed_file(file.filename):
        flash("Only PNG, JPG and JPEG images are allowed.")
        return redirect(url_for("scan"))

    extension = os.path.splitext(secure_filename(file.filename))[1].lower()
    unique_name = f"{uuid.uuid4().hex}{extension}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(save_path)

        image = Image.open(save_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities_tensor = torch.softmax(output, dim=1)[0]

        confidence_value, predicted_index = torch.max(probabilities_tensor, 0)

                prediction_raw = CLASS_NAMES[predicted_index.item()]
        prediction = format_class_name(prediction_raw)

        # Clinical Recommendation
        if prediction == "No Tumor":
            recommendation = (
                "No tumor detected. Continue routine clinical follow-up if symptoms persist."
            )
            warning = ""
        else:
            recommendation = (
                "Tumor detected. Please consult a neurologist or radiologist for detailed clinical evaluation."
            )
            warning = (
                "AI prediction should always be confirmed by a qualified medical professional."
            )

        confidence = round(confidence_value.item() * 100, 2)
        probabilities = {}

        for index, class_name in enumerate(CLASS_NAMES):
            probabilities[format_class_name(class_name)] = round(
                probabilities_tensor[index].item() * 100,
                2
            )

        # Grad-CAM must run WITHOUT torch.no_grad()
        cam = grad_cam.generate(image_tensor, predicted_index.item())

        heatmap_name = f"gradcam_{uuid.uuid4().hex}.jpg"
        heatmap_path = os.path.join(
            app.config["HEATMAP_FOLDER"],
            heatmap_name
        )

        create_gradcam_image(save_path, cam, heatmap_path)

        image_url = url_for(
            "static",
            filename=f"uploads/{unique_name}"
        )

        heatmap_url = url_for(
            "static",
            filename=f"heatmaps/{heatmap_name}"
        )
        history_item = {
            "id": uuid.uuid4().hex,
            "date": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence,
            "image_file": unique_name,
            "heatmap_file": heatmap_name,
            "probabilities": probabilities
        }
        save_history_item(history_item)

        return render_template(
    "result.html",
    prediction=prediction,
    confidence=confidence,
    probabilities=probabilities,
    filename=file.filename,
    image_url=image_url,
    heatmap_url=heatmap_url,
    uploaded_file=unique_name,
    recommendation=recommendation,
    warning=warning
)

    except UnidentifiedImageError:
        if os.path.exists(save_path):
            os.remove(save_path)

        flash("The uploaded file is not a valid image.")
        return redirect(url_for("scan"))

    except Exception as error:
        print("PREDICTION ERROR:", error)

        if os.path.exists(save_path):
            os.remove(save_path)

        flash("Prediction failed. Check VS Code terminal for the exact error.")
        return redirect(url_for("scan"))
    

@app.route("/download-report")
def download_report():
    flash("PDF report generation is not implemented yet.")
    return redirect(url_for("index"))


@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/performance')
def performance():
    performance_data = {
        "accuracy": "94.76%",
        "precision": "94.76%",
        "recall": "94.76%",
        "f1_score": "94.76%",
        "total_test_images": "1602"
    }

    return render_template(
        "performance.html",
        performance=performance_data
    )
@app.route("/history")
def history():
    history_items = load_history()
    return render_template("history.html", history_items=history_items)


@app.route("/history/<scan_id>")
def history_result(scan_id):
    history_items = load_history()

    selected_item = next(
        (item for item in history_items if item.get("id") == scan_id),
        None
    )

    if selected_item is None:
        flash("History result not found.")
        return redirect(url_for("history"))

    image_url = url_for(
        "static",
        filename=f"uploads/{selected_item.get('image_file', '')}"
    )

    heatmap_file = selected_item.get("heatmap_file", "")
    heatmap_url = (
        url_for("static", filename=f"heatmaps/{heatmap_file}")
        if heatmap_file else None
    )

    return render_template(
    "result.html",
    prediction=selected_item["prediction"],
    confidence=selected_item["confidence"],
    probabilities=selected_item.get("probabilities", {}),
    filename=selected_item.get("filename", "MRI Scan"),
    image_url=image_url,
    moment=datetime.now().strftime("%d-%m-%Y %I:%M %p"),
    heatmap_url=heatmap_url,
    uploaded_file=selected_item.get("image_file", ""),
    recommendation=(
        "No tumor detected. Continue routine follow-up."
        if selected_item["prediction"] == "No Tumor"
        else "Tumor detected. Please consult a neurologist."
    ),
    warning=(
        ""
        if selected_item["prediction"] == "No Tumor"
        else "AI prediction should be confirmed by a medical professional."
    )
)
@app.route("/history/delete/<scan_id>", methods=["POST"])
def delete_history(scan_id):
    history_items = load_history()

    updated_history = [
        item for item in history_items
        if item.get("id") != scan_id
    ]

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(updated_history, file, indent=4)

    flash("History item deleted.")
    return redirect(url_for("history"))
@app.errorhandler(413)
def too_large(error):
    flash("File is too large. Maximum allowed size is 10 MB.")
    return redirect(url_for("scan"))


if __name__ == "__main__":
    app.run(debug=True)
   