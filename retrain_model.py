# Fixed Training Script
import numpy as np
import cv2
import os
import glob
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Global
FIXED_SIZE = (512, 512)

def azimuthalAverage(image, center=None):
    """
    Menghitung rata-rata radial dari spektrum frekuensi 2D.
    Mengubah gambar 2D menjadi grafik garis 1D.
    """
    y, x = image.shape
    if not center:
        center = np.array([x//2, y//2])

    # Buat grid koordinat
    xx, yy = np.meshgrid(np.arange(x), np.arange(y))
    
    # Hitung jarak setiap piksel ke pusat (Radius)
    r = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    r = r.astype(int)

    # Jumlahkan energi per radius (Binning)
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    
    # Rata-rata (hindari pembagian nol)
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

def extract_features(image_path):
    """Extract FFT features from image"""
    try:
        img = cv2.imread(image_path, 0)
        if img is None: 
            return None
        img = cv2.resize(img, FIXED_SIZE)

        # FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        
        # Log Magnitude
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)

        # Azimuthal Integration
        psd1D = azimuthalAverage(magnitude_spectrum)
        
        # Ambil data tanpa titik 0
        return psd1D[1:] 
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Set paths
real_folder = "images/ffhq/00000"
fake_folder = "images/fake"

# Check if folders exist and have images
if not os.path.exists(real_folder):
    print(f"❌ Folder REAL tidak ditemukan: {real_folder}")
    exit()
if not os.path.exists(fake_folder):
    print(f"❌ Folder FAKE tidak ditemukan: {fake_folder}")
    exit()

real_count = len(glob.glob(os.path.join(real_folder, "*")))
fake_count = len(glob.glob(os.path.join(fake_folder, "*")))

print(f"📂 Found folders:")
print(f"   REAL: {real_folder} ({real_count} files)")
print(f"   FAKE: {fake_folder} ({fake_count} files)")

if real_count == 0:
    print("\n❌ ERROR: No REAL images found!")
    print("   Please add REAL images to images/ffhq/ folder")
    exit()
if fake_count == 0:
    print("\n❌ ERROR: No FAKE images found!")
    exit()

MAX_DATA = 1000  # Adjust as needed

X = []
y = []

print(f"🔄 Processing REAL images...")
real_files = glob.glob(os.path.join(real_folder, "*"))[:MAX_DATA]
for f in real_files:
    feat = extract_features(f)
    if feat is not None:
        X.append(feat)
        y.append(1)  # 1 = REAL
print(f"✅ Processed {len([yi for yi in y if yi == 1])} REAL images")

print(f"🔄 Processing FAKE images...")
fake_files = glob.glob(os.path.join(fake_folder, "*"))[:MAX_DATA]
for f in fake_files:
    feat = extract_features(f)
    if feat is not None:
        X.append(feat)
        y.append(0)  # 0 = FAKE
print(f"✅ Processed {len([yi for yi in y if yi == 0])} FAKE images")

# Convert to arrays
X = np.array(X)
y = np.array(y)
X = np.nan_to_num(X)

print(f"\n📊 Dataset Summary:")
print(f"   Total samples: {len(X)}")
print(f"   REAL: {np.sum(y == 1)} | FAKE: {np.sum(y == 0)}")
print(f"   Feature shape: {X.shape}")

# Check for class imbalance
if np.sum(y == 1) / len(y) > 0.8 or np.sum(y == 0) / len(y) > 0.8:
    print("⚠️ WARNING: Severe class imbalance detected!")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🚀 Training SVM with Grid Search...")
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.1, 0.01],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    SVC(probability=True, class_weight='balanced'),  # Added class_weight
    param_grid, 
    refit=True, 
    verbose=2, 
    cv=3
)

grid.fit(X_train, y_train)

print(f"\n✅ Best parameters: {grid.best_params_}")
print(f"🏆 Best CV score: {grid.best_score_ * 100:.2f}%")

# Test model
model = grid.best_estimator_
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 TEST SET ACCURACY: {acc * 100:.2f}%")
print("\n" + classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred FAKE', 'Pred REAL'], 
            yticklabels=['True FAKE', 'True REAL'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("📊 Confusion matrix saved as 'confusion_matrix.png'")

# Save model
filename = 'model_svm_best.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print(f"\n💾 Model saved as '{filename}'")

# Test on some samples to verify
print("\n🔍 Quick verification on 5 test samples:")
for i in range(min(5, len(X_test))):
    pred = model.predict([X_test[i]])
    prob = model.predict_proba([X_test[i]])
    print(f"Sample {i+1}: True={y_test[i]}, Pred={pred[0]}, Prob=[{prob[0][0]:.3f}, {prob[0][1]:.3f}]")
