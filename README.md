# Yüz İfadesi Tabanlı Duygu Analizi (CNN)

Bu proje, **FER_2013** veri setini kullanarak yedi temel duyguyu sınıflandırabilen bir **Evrişimsel Sinir Ağı (CNN)** modeli geliştirmektedir. Duygu analizi, insan-bilgisayar etkileşimi, pazarlama ve güvenlik sistemleri gibi birçok alanda kritik öneme sahiptir.

## I. Proje Amacı ve Hedefleri
Projenin amacı, yüz ifadelerini analiz ederek insan duygularını doğru şekilde tahmin eden bir model geliştirmektir.

### Hedefler:
- Farklı yüz ifadelerini yüksek doğrulukla sınıflandırmak.
- Model doğruluğunu artırarak daha güvenilir tahminler elde etmek.
- Kullanıcı dostu bir arayüz geliştirmek.

## II. Veri Seti ve Ön İşleme
- **Veri Seti:** Kaggle FER_2013 (kızgın, korku, iğrenme, mutlu, üzgün, şaşkın, nötr).  
  48x48 piksel, gri tonlamalı, toplam 29.700 görüntü. Eğitim ve test seti olarak ikiye ayrılmıştır.

### Ön İşleme Adımları:
1. **Normalizasyon:** Piksel değerleri 0-1 aralığına ölçeklendirildi.
2. **Veri Artırma:** Döndürme, yansıtma, yakınlaştırma ile çeşitlilik artırıldı.
3. **Boyutlandırma:** Tüm görüntüler 48x48 piksele ayarlandı.
4. **Etiket Kodlama:** `LabelEncoder` ve `to_categorical` ile çok sınıflı format oluşturuldu.

## III. Model Mimarisi
- **Giriş Katmanı:** 48x48 piksel gri tonlamalı görüntüler.
- **Konvolüsyonel Katmanlar:** ReLU aktivasyonu, 32 ve 64 filtre.
- **Havuzlama (Pooling):** MaxPooling ile boyut küçültme.
- **Düzleştirme (Flatten):** Özellik haritalarının vektöre dönüştürülmesi.
- **Tam Bağlantılı Katmanlar:** 7 duygu sınıfı için yoğun katmanlar.
- **Eğitim Parametreleri:**  
  - Öğrenme oranı: 0.001  
  - Batch size: 128  
  - Epoch: 50  
  - Optimizasyon: Adam  
  - Kayıp fonksiyonu: Kategorik Çapraz Entropi  

## IV. Kullanılan Kütüphaneler
- **TensorFlow / Keras** – Model geliştirme ve eğitim.
- **NumPy** – Sayısal işlemler.
- **Pandas** – Veri yönetimi.
- **OpenCV** – Görüntü işleme.

## V. Deneysel Sonuçlar
- Eğitim: %70 veri ile, Test: %30 veri ile yapıldı.
- Ortalama doğruluk: **%68.55**
- “İğrenme” kategorisinde düşük performans gözlendi (sınıf dengesizliği nedeniyle).

## VI. Gelecek Çalışmalar
- Daha büyük ve çeşitli veri setleri kullanmak.
- Kullanıcı arayüzünü geliştirmek.
- Gerçek zamanlı yüz tanıma entegrasyonu yapmak.

## VII. Sonuç
Bu proje, yüz ifadelerinden yedi temel duyguyu başarıyla sınıflandıran CNN tabanlı bir model geliştirmiştir. Daha büyük veri setleri ve derin ağ mimarileri ile doğruluk oranının artırılması mümkündür.
