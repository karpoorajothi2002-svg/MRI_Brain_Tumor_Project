import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # Global average pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input image size (224x224)
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        return cam.squeeze().cpu().numpy(), class_idx


def apply_heatmap(original_img_path, cam_array, alpha=0.4):
    """
    original_img_path: path to original MRI image
    cam_array: numpy array from GradCAM.generate()
    Returns: heatmap overlaid image as numpy array
    """
    # Read original image
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))

    # Convert CAM to heatmap (COLORMAP_JET: blue→green→red)
    cam_uint8 = np.uint8(255 * cam_array)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    overlaid = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return overlaid, heatmap