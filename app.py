import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURASI HALAMAN (DESIGN)
# ==========================================
st.set_page_config(
    page_title="Deepfake Face Image Detector",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS biar tampilannya agak "Cyber/Dark"
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-family: 'Courier New', monospace;
        color: #00FF00;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #4B4B4B;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI MATEMATIKA ALJABAR LINEAR
# ==========================================

# Konfigurasi Global
FIXED_SIZE = (512, 512)  # Ukuran standar resize

def azimuthalAverage(image, center=None):
    """
    Fungsi ALIN: Reduksi Dimensi (2D Matrix -> 1D Vector)
    Menghitung rata-rata radial dari pusat ke pinggir.
    """
    y, x = image.shape
    if not center:
        center = np.array([x//2, y//2])
    
    # Bikin grid koordinat (x,y)
    xx, yy = np.meshgrid(np.arange(x), np.arange(y))
    
    # Hitung jarak setiap pixel dari pusat (Euclidean Distance)
    r = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    r = r.astype(int)

    # Jumlahkan energi per cincin jari-jari (Binning)
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    
    # Hindari pembagian dengan nol
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

def extract_features(image_array):
    """
    PIPELINE UTAMA:
    1. Resize Image (Matriks)
    2. FFT (Transformasi Basis)
    3. Logaritma (Scaling)
    4. Azimuthal Average (Vektorisasi)
    """
    try:
        # image_array sudah berupa numpy array, tidak perlu cv2.imread lagi
        img = image_array
        if img is None: 
            return None
        img = cv2.resize(img, FIXED_SIZE)  # Resize ke ukuran standar

        # FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        
        # KEMBALI KE LOG MAGNITUDE (Ini yang terbukti bekerja)
        # Logaritma menyelamatkan data frekuensi tinggi dari angka nol
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)

        # Azimuthal Integration
        psd1D = azimuthalAverage(magnitude_spectrum)
        
        # Ambil data tanpa titik 0 (DC component yang nilainya raksasa itu)
        return psd1D[1:] 
    
    except Exception as e:
        return None

# ==========================================
# 3. LOAD MODEL & SCALER
# ==========================================
@st.cache_resource
def load_model():
    try:
        with open('model_svm_best.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

@st.cache_resource
def load_scaler():
    """Load scaler if exists (optional - only if model was trained with scaling)"""
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        return None

model = load_model()
scaler = load_scaler()

# ==========================================
# 4. TAMPILAN UTAMA (UI)
# ==========================================

# Sidebar Informasi
with st.sidebar:
    st.title("Deepfake Detector")
    st.info("""
    **Mata Kuliah:** Aljabar Linear
    
    **NPM** 140810240054
    
    **Nama:** M. Nurrizal Zid Maulana
    """)

    st.write("---")
    
    with st.expander("Konsep di Balik Layar"):
        st.markdown("""
        **1. Gambar adalah Matriks**
        Komputer tidak melihat "wajah", melainkan **matriks angka** (piksel). Setiap angka mewakili seberapa terang titik tersebut.
        
        **2. Analisis Frekuensi (FFT)**
        Kita tidak melihat posisi mata/hidung, tapi melihat **seberapa kasar atau halus** tekstur gambar tersebut:
        * **Frekuensi Rendah:** Bagian yang mulus (pipi, jidat, pencahayaan).
        * **Frekuensi Tinggi:** Bagian yang kasar/detail (pori-pori, ujung rambut, noise kamera).
        
        **3. Meringkas Data (Vektor 1D)**
        Matriks 2D yang rumit tadi diringkas menjadi satu grafik garis (vektor). Caranya dengan menghitung rata-rata energi dari pusat (frekuensi rendah) melebar ke pinggir (frekuensi tinggi).
        
        **4. Keputusan (SVM)**
        Kita menggunakan SVM untuk mencari **Garis Batas (Hyperplane)** di ruang vektor yang memisahkan dua kelompok data:
        * **REAL:** Vektor yang jatuh di area "kaya energi frekuensi tinggi".
        * **FAKE:** Vektor yang jatuh di area "minim energi frekuensi tinggi".
        """)

    with st.expander("Cara Membaca Grafik Spektrum"):
        st.markdown("""
        Aplikasi ini mengubah gambar menjadi sinyal gelombang (1D Profile). Berikut pedoman membacanya sesuai analisis laporan:
        
        **1. Sumbu X (Kiri ke Kanan)**
        Menunjukkan frekuensi dari rendah (pusat gambar) ke tinggi (detail halus).
        
        **2. Sumbu Y (Tinggi Grafik)**
        Menunjukkan kekuatan energi (Log Magnitude) di frekuensi tersebut.
        
        **3. Perbedaan Real vs Fake:**
        * ✅ **REAL (Asli):** Grafik melandai secara alami. Di ujung kanan (frekuensi tinggi), **masih terdapat nilai energi** yang stabil (karena kamera asli menangkap tekstur/noise alami).
        * 🚨 **FAKE (Palsu):** Grafik **menurun lebih curam/tajam**. Di ujung kanan, nilainya **lebih rendah** dibandingkan citra asli (karena efek *smoothing* dari AI menghilangkan detail mikro).
        """)
    
    with st.expander("Sumber Foto Wajah"):
        st.markdown("""
        - [FFHQ Dataset Github](https://github.com/NVlabs/ffhq-dataset?tab=readme-ov-file)
        - [FFHQ Dataset GDrive](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP)
        - [This Person Does Not Exist](https://thispersondoesnotexist.com/)
        """)

    st.write("---")
    st.write("### Cara Kerja:")
    st.write("1. **Upload** Foto Wajah")
    st.write("2. **FFT** mengubah gambar jadi gelombang.")
    st.write("3. **SVM** mengecek pola frekuensi tinggi.")

# Header
st.title("Deepfake Face Image Detector")
st.write("---")

# Area Upload
uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar dengan PIL lalu convert ke Grayscale (L) -> Array
    image_pil = Image.open(uploaded_file).convert('L')
    img_array = np.array(image_pil)
    
    # Tampilkan Gambar Asli dan Tombol Analisis berdampingan
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, caption="Input Image (Matrix)", use_container_width=True)
    
    with col2:
        st.write("### Analysis Dashboard")
        
        if model is None:
            st.error("⚠️ File 'model_svm_best.pkl' tidak ditemukan! Pastikan file ada di folder yang sama.")
        else:
            if st.button("Jalankan Analisis", type="primary"):
                with st.spinner("Menghitung Transformasi Fourier (FFT)..."):
                    
                    # --- PROSES MATEMATIKA ---
                    features = extract_features(img_array)
                    
                    if features is None:
                        st.error("❌ Gagal mengekstrak fitur dari gambar!")
                        st.stop()
                    
                    # Reshape karena SVM butuh input 2D array [[f1, f2, ...]]
                    input_data = features.reshape(1, -1)
                    
                    # Apply scaler jika ada (jika model ditraining dengan scaling)
                    if scaler is not None:
                        input_data = scaler.transform(input_data)
                        st.caption("⚙️ StandardScaler applied (model trained with scaling)")
                    
                    # Prediksi
                    prediction = model.predict(input_data)
                    probabilities = model.predict_proba(input_data)
                    
                    # Ambil confidence score tertinggi
                    confidence = np.max(probabilities) * 100
                    class_label = "REAL (ASLI)" if prediction[0] == 1 else "FAKE (AI/GAN)"
                    
                    # --- TAMPILKAN HASIL ---
                    st.write("---")
                    
                    if prediction[0] == 1:
                        st.success(f"**HASIL DETEKSI: {class_label}**")
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                        st.caption("Analisis: Grafik spektrum menunjukkan pola frekuensi tinggi yang kaya (natural noise).")
                    else:
                        st.error(f"**HASIL DETEKSI: {class_label}**")
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                        st.caption("Analisis: Terdeteksi penurunan drastis pada frekuensi tinggi (ciri khas upsampling GAN).")

                    st.write("### Visualisasi Vektor Spektrum")
                    st.write("Grafik ini adalah representasi vektor 1D dari citra setelah transformasi basis:")                    
                    st.line_chart(features)
                    st.info("Sumbu X: Frekuensi (Rendah ke Tinggi) | Sumbu Y: Log Magnitude (Energi)")

else:
    st.info("Silakan upload foto wajah di sebelah kiri/atas untuk memulai.")