# Crypto Anomaly Detection Pipeline

> **A robust anomaly detection pipeline for cryptocurrency markets using
> statistical methods, supervised/unsupervised LSTM models, and hybrid
> autoencoders --- with a strict focus on preventing data leakage.**

------------------------------------------------------------------------

## 🚀 Proje Hakkında (Overview)

Bu proje, kripto para piyasalarındaki (özellikle **BTC/USDT**) fiyat
hareketlerini analiz ederek **Anomaly Detection** gerçekleştiren uçtan
uca bir pipeline sunar.

Temel amaç; - Geleneksel **istatistiksel yöntemleri** - Modern **derin
öğrenme (Deep Learning)** mimarilerini

bir arada kullanarak piyasadaki **olağan dışı (aykırı) hareketleri**
tespit etmektir.

Projenin en kritik özelliği, zaman serisi çalışmalarında sıkça yapılan
**Data Leakage (veri sızıntısı)** hatasını önlemesidir.\
Bu, özellikle: - **Scaling** - **Train / Test ayrımı** - **Model
eğitimi**

adımlarının zamansal bütünlük korunarak tamamen izole edilmesiyle
sağlanmıştır.

------------------------------------------------------------------------

## 🛠 Kullanılan Metotlar (Methodology)

Pipeline içerisinde **4 farklı yaklaşım** entegre edilmiştir:

### 1️⃣ Statistical Methods

-   **Z-Score**\
    Standart sapma tabanlı sapmaları yakalayarak ani fiyat hareketlerini
    tespit eder.
-   **EWMA (Exponentially Weighted Moving Average)**\
    Dinamik ağırlıklı ortalama ile trend dışı davranışları belirler.

### 2️⃣ Unsupervised LSTM (Forecasting-Based)

-   Model, geçmiş zaman adımlarını kullanarak **bir sonraki fiyatı
    tahmin eder**.
-   **Prediction Error**, belirlenen bir **threshold** değerini
    aştığında ilgili zaman noktası **anomali** olarak işaretlenir.
-   Etiket gerektirmeyen, tamamen **unsupervised** bir yaklaşımdır.

### 3️⃣ Supervised LSTM (Classification-Based)

-   İstatistiksel yöntemlerden elde edilen anomali etiketleri **ground
    truth** olarak kullanılır.
-   Model, yeni gelen verinin **normal mi anomali mi** olduğunu doğrudan
    sınıflandırır.
-   Zamansal bağımlılıkları öğrenebilen bir **sequence classification**
    problemidir.

### 4️⃣ Hybrid: LSTM Autoencoder + One-Class SVM

-   **LSTM Autoencoder**, zaman serisini düşük boyutlu bir **latent
    space**'e sıkıştırır.
-   **One-Class SVM (OCSVM)**, bu latent uzaydaki aykırı noktaları
    yüksek hassasiyetle tespit eder.
-   Özellikle **non-linear** ve karmaşık anomaliler için güçlü bir
    hibrit yaklaşımdır.

------------------------------------------------------------------------

## 📂 Dosya Yapısı (File Structure)

    .
    ├── main.py               # Pipeline giriş noktası (data → train → evaluate)
    ├── config.py             # Tüm hyperparameter ayarları
    ├── statistic.py          # Veri çekme (CCXT) ve feature engineering
    ├── lstm_unsupervised.py  # Forecasting tabanlı LSTM
    ├── lstm_supervised.py    # Classification tabanlı LSTM
    ├── lstm_AE.py            # LSTM Autoencoder + OCSVM
    ├── evaluation.py         # ROC-AUC, PR, F1-score hesaplamaları
    └── graphs.py             # Anomali görselleştirmeleri

------------------------------------------------------------------------

## ⚙️ Kurulum ve Kullanım (Setup & Usage)

### 1️⃣ Gereksinimleri Yükleyin

``` bash
pip install ccxt pandas numpy tensorflow scikit-learn matplotlib seaborn
```

### 2️⃣ Projeyi Çalıştırın

``` bash
python main.py
```

------------------------------------------------------------------------

## 📊 Değerlendirme (Evaluation)

Model performansı aşağıdaki metriklerle değerlendirilir: - **ROC-AUC** -
**Precision -- Recall Curve** - **F1-Score**

Anomali tespit sonuçları, fiyat serisi üzerinde **görsel olarak** da
analiz edilebilir.

------------------------------------------------------------------------

## 🔒 Data Leakage Önleme Stratejisi

-   Scaling **sadece training set** üzerinde yapılır\
-   Test verisi geleceği temsil eder, geçmişten bilgi sızdırılmaz\
-   Sliding window yapıları zamansal sırayı bozmadan oluşturulur

Bu sayede sonuçlar **gerçekçi ve güvenilir** kalır.
