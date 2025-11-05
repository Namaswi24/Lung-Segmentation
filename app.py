import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
from pathlib import Path
import time

# ============================================================
# 1. CONFIGURATION: PLEASE CHECK THESE PATHS
# ============================================================

# --- Path to your saved model ---
# Make sure this file is in the same folder as app.py
MODEL_CHECKPOINT_PATH = "best_medt_model.pth"

# --- Model Parameters (must match training) ---
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Folder for saving uploads & predictions ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


print(f"Using device: {DEVICE}")

# ============================================================
# 2. MODEL DEFINITION
# (Must be identical to your training script to load the weights)
# ============================================================

class GatedAxialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, axis='width', max_size=128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.axis = axis
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.pos_query = nn.Parameter(torch.randn(1, num_heads, max_size, self.head_dim))
        self.pos_key = nn.Parameter(torch.randn(1, num_heads, max_size, self.head_dim))
        self.pos_value = nn.Parameter(torch.randn(1, num_heads, max_size, self.head_dim))
        self.gate_query = nn.Parameter(torch.ones(1))
        self.gate_key = nn.Parameter(torch.ones(1))
        self.gate_value1 = nn.Parameter(torch.ones(1))
        self.gate_value2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape
        if self.axis == 'width':
            x = x.permute(0, 2, 3, 1)
            x_flat = x.reshape(B * H, W, C)
            dim_len = W
        else:
            x = x.permute(0, 3, 2, 1)
            x_flat = x.reshape(B * W, H, C)
            dim_len = H
        q = self.q_proj(x_flat).reshape(-1, dim_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_flat).reshape(-1, dim_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_flat).reshape(-1, dim_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        pos_q = self.pos_query[:, :, :dim_len, :]
        pos_k = self.pos_key[:, :, :dim_len, :]
        pos_v = self.pos_value[:, :, :dim_len, :]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + (q @ (self.gate_query * pos_q).transpose(-2, -1)) * self.scale
        attn = attn + ((k @ (self.gate_key * pos_k).transpose(-2, -1)).transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ (self.gate_value1 * v + self.gate_value2 * pos_v)
        out = out.permute(0, 2, 1, 3).reshape(-1, dim_len, self.dim)
        out = self.out_proj(out)
        if self.axis == 'width':
            out = out.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)
        else:
            out = out.reshape(B, W, H, self.dim).permute(0, 3, 2, 1)
        return out

class GatedAxialTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim); self.norm2 = nn.GroupNorm(1, dim); self.norm3 = nn.GroupNorm(1, dim)
        self.height_attn = GatedAxialAttention(dim, num_heads, axis='height')
        self.width_attn = GatedAxialAttention(dim, num_heads, axis='width')
        self.ffn = nn.Sequential(nn.Conv2d(dim, dim * 4, 1), nn.GELU(), nn.Conv2d(dim * 4, dim, 1))
    def forward(self, x):
        x = x + self.height_attn(self.norm1(x)); x = x + self.width_attn(self.norm2(x)); x = x + self.ffn(self.norm3(x))
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class MedT(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, img_size=128):
        super().__init__()
        base = 32
        self.g_enc1 = ConvBlock(in_channels, base); self.g_pool1 = nn.MaxPool2d(2)
        self.g_enc2 = ConvBlock(base, base * 2); self.g_pool2 = nn.MaxPool2d(2)
        self.g_transformer = nn.Sequential(*[GatedAxialTransformerBlock(base * 2, 4) for _ in range(2)])
        self.l_enc1 = ConvBlock(in_channels, base); self.l_pool1 = nn.MaxPool2d(2)
        self.l_enc2 = ConvBlock(base, base * 2); self.l_pool2 = nn.MaxPool2d(2)
        self.l_enc3 = ConvBlock(base * 2, base * 4); self.l_pool3 = nn.MaxPool2d(2)
        self.l_transformer = nn.Sequential(*[GatedAxialTransformerBlock(base * 4, 8) for _ in range(4)])
        self.dec3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec_conv3 = ConvBlock(base * 4, base * 2)
        self.dec2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec_conv2 = ConvBlock(base * 2, base)
        self.dec1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec_conv1 = ConvBlock(base, base)
        self.final = nn.Conv2d(base, num_classes, 1)
        self.g_conv = nn.Conv2d(base * 2, base * 4, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        g1 = self.g_enc1(x); g2 = self.g_enc2(self.g_pool1(g1)); g_feat = self.g_transformer(self.g_pool2(g2))
        patch_size = H // 4
        patches = []
        for i in range(4):
            for j in range(4):
                patch = x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                l_feat = self.l_transformer(self.l_pool3(self.l_enc3(self.l_pool2(self.l_enc2(self.l_pool1(self.l_enc1(patch)))))))
                patches.append(l_feat)
        rows = [torch.cat(patches[i*4:(i+1)*4], dim=3) for i in range(4)]; l_features = torch.cat(rows, dim=2)
        g_up = F.interpolate(g_feat, size=l_features.shape[2:], mode='bilinear', align_corners=False)
        g_up = self.g_conv(g_up) if g_up.shape[1] != l_features.shape[1] else g_up
        merged = l_features + g_up
        d3 = self.dec3(merged); d3 = self.dec_conv3(torch.cat([d3, F.interpolate(g2, size=d3.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        d2 = self.dec2(d3); d2 = self.dec_conv2(torch.cat([d2, F.interpolate(g1, size=d2.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        d1 = self.dec_conv1(self.dec1(d2))
        # Return logits (model was trained with BCEWithLogitsLoss)
        return self.final(d1)

# ============================================================
# 3. HELPER FUNCTIONS & MODEL LOADING
# ============================================================

def preprocess_image(img_bytes):
    """Converts image bytes to a normalized tensor."""
    # Read image from bytes
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize (must match training)
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    
    # Convert to Tensor
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    return img_tensor, img

# Load the model
try:
    model = MedT(in_channels=3, num_classes=1, img_size=IMG_SIZE).to(DEVICE)
    
    # Load the weights
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
    
    # Check if it's a state_dict or a full checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval() # Set to evaluation mode
    print(f"✅ Model loaded successfully from {MODEL_CHECKPOINT_PATH}")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    model = None

# ============================================================
# 4. FLASK WEB SERVER
# ============================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    """Serves the main HTML page."""
    # This sends the index.html file from the same directory
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and returns a prediction."""
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        
        # 1. Preprocess the image
        img_tensor, original_img = preprocess_image(img_bytes)
        
        # 2. Run prediction
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
            mask = (probs > 0.5).cpu().numpy().astype(np.uint8) * 255
            
        # 3. Post-process mask
        mask = mask.squeeze() # Remove batch and channel dims
        
        # Resize mask to original image size
        mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # 4. Save the mask
        # Create a unique filename
        filename = f"{int(time.time())}_{Path(file.filename).stem}_mask.png"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(save_path, mask_resized)
        
        # 5. Return the URL to the saved mask
        # 'static' is the flask endpoint, 'filename' is the path *within* static
        mask_url = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify({'mask_url': mask_url})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Failed to process image.'}), 500

if __name__ == "__main__":
    print("Starting Flask server... Go to http://127.0.0.1:5000")
    # Using host='0.0.0.0' makes it accessible on your network
    app.run(debug=True, host='0.0.0.0')