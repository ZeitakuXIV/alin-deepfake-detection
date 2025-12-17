# Deteksi Deepfake menggunakan Analisis Spektral

**Mata Kuliah:** Aljabar Linear
**Penulis:** M. Nurrizal Zid Maulana (140810240054)

## Ringkasan
Proyek ini adalah alat **Deteksi Wajah Deepfake** yang memanfaatkan **Analisis Domain Frekuensi** untuk membedakan antara wajah manusia asli dan gambar yang dihasilkan oleh AI (GAN). Berbeda dengan metode berbasis CNN tradisional yang mencari artefak visual, pendekatan ini menganalisis **pola spektral** gambar menggunakan **Fast Fourier Transform (FFT)**.

Gambar yang dihasilkan AI sering kali menunjukkan artefak yang berbeda dalam domain frekuensi tinggi karena operasi upsampling dalam GAN. Alat ini mendeteksi anomali tersebut.

## Fitur
- **Analisis Spektral**: Mengubah gambar ke domain frekuensi menggunakan FFT.
- **Spektrum Daya 1D**: Mengekstrak profil radial (Rata-rata Azimuthal) untuk mengurangi dimensi.
- **Machine Learning**: Menggunakan pengklasifikasi Support Vector Machine (SVM) untuk memprediksi Asli vs. Palsu.
- **UI Interaktif**: Dibangun dengan [Streamlit](https://streamlit.io/) untuk mengunggah gambar dengan mudah dan analisis waktu nyata.
- **Visualisasi**: Menampilkan grafik Spektrum Daya Log 1D untuk analisis forensik.

## Instalasi

1. **Clone repositori**
   ```bash
   git clone <repository-url>
   cd alin-detect-deepfake
   ```

2. **Instal Dependensi**
   Pastikan Anda telah menginstal Python. Kemudian jalankan:
   ```bash
   pip install streamlit opencv-python numpy matplotlib scikit-learn Pillow
   ```

## Penggunaan

Jalankan aplikasi Streamlit:
```bash
streamlit run app.py
```
Buka browser Anda di `http://localhost:8501` untuk menggunakan detektor.

## Metodologi

1. **Preprocessing**: Ubah gambar menjadi grayscale dan ubah ukuran menjadi 512x512.
2. **FFT 2D**: Terapkan Fast Fourier Transform untuk mendapatkan spektrum frekuensi.
3. **Rata-rata Azimuthal**: Hitung magnitudo rata-rata pada setiap radius dari pusat (frekuensi rendah) ke tepi (frekuensi tinggi).
4. **Ekstraksi Fitur**: Profil 1D yang dihasilkan berfungsi sebagai vektor fitur.
5. **Klasifikasi**: Model SVM (dilatih pada gambar FFHQ dan StyleGAN) mengklasifikasikan vektor tersebut.

## Dataset yang Digunakan

- **Gambar Palsu**: [thispersondoesnotexist.com](https://thispersondoesnotexist.com/) (StyleGAN2)
- **Gambar Asli**: [Flickr-Faces-HQ (FFHQ)](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)

## Referensi

- [AutoGAN: Neural Architecture Search for Generative Adversarial Networks](https://arxiv.org/pdf/2003.01826)
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.11842)
- [Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions](https://arxiv.org/pdf/2003.01826)
