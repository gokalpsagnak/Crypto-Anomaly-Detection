# Crypto Anomaly Detection Pipeline

> **A robust anomaly detection pipeline for cryptocurrency markets using Statistical Methods, Supervised/Unsupervised LSTM, and Hybrid Autoencoders with a strict focus on preventing data leakage.**

---

## 🚀 Proje Hakkında (Overview)

Bu proje, kripto para piyasalarındaki (BTC/USDT) fiyat hareketlerini analiz ederek **Anomaly Detection** gerçekleştiren uçtan uca bir pipeline sunar. Projenin temel amacı, hem geleneksel istatistiksel yöntemleri hem de modern derin öğrenme (Deep Learning) mimarilerini kullanarak piyasadaki aykırı hareketleri tespit etmektir.

En kritik özelliği, zaman serisi çalışmalarında sıkça yapılan **Data Leakage** (veri sızıntısı) hatasını; veri ölçeklendirme (scaling) ve model eğitimi süreçlerini tamamen izole ederek çözmesidir.

## 🛠 Kullanılan Metotlar (Methodology)

Proje içerisinde 4 farklı yaklaşım entegre edilmiştir:

1.  **Statistical Methods:**
    * **Z-Score:** Standart sapma üzerinden anlık fiyat sapmalarını yakalar.
    * **EWMA (Exponentially Weighted Moving Average):** Dinamik bir hareketli ortalama kullanarak trend dışı hareketleri belirler.
2.  **Unsupervised LSTM (Forecasting):**
    * Model, geçmiş veriye bakarak bir sonraki fiyatı tahmin eder. **Prediction Error** belirli bir **Threshold** (eşik) değerini aştığında bu noktaları anomali olarak tanımlar.
3.  **Supervised LSTM (Classification):**
    * İstatistiksel metotlardan elde edilen etiketleri (ground truth) kullanarak, yeni verilerin anomali olup olmadığını sınıflandırır.
4.  **Hybrid Autoencoder + OCSVM:**
    * **LSTM Autoencoder** veriyi düşük boyutlu bir temsil alanına (**Latent Space**) sıkıştırır. **One-Class SVM** ise bu alandaki aykırı noktaları yüksek hassasiyetle yakalar.

---

## 📂 Dosya Yapısı (File Structure)

* `main.py`: Projenin giriş noktası. Tüm akışı (Data fetch -> Preprocess -> Training -> Evaluation) yönetir.
* `config.py`: Tüm **Hyperparameters** (epoch, batch size, thresholds) ayarlarının yapıldığı merkez.
* `statistic.py`: Veri çekme (CCXT) ve **Feature Engineering** modülü.
* `lstm_AE.py`: Autoencoder ve OCSVM hibrit model mimarisi.
* `lstm_unsupervised.py`: Tahmin tabanlı LSTM operasyonları.
* `lstm_supervised.py`: Sınıflandırma odaklı LSTM yapısı.
* `evaluation.py`: **ROC-AUC**, **Precision-Recall** ve **F1-Score** hesaplama ve görselleştirme araçları.
* `graphs.py`: Anomali sonuçlarının fiyat grafiği üzerinde görselleştirilmesi.

---

## ⚙️ Kurulum ve Kullanım (Setup)

### 1. Gereksinimleri Yükleyin
```bash
pip install ccxt pandas numpy tensorflow scikit-learn matplotlib seaborn
