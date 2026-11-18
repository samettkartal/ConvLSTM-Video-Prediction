# Şirket belgesi parçaları

Bu dosya yalnızca referanstır; depoya yazılmaz.

**Niyet (sayı/süre buradan kopyalanmaz):** README.md sonuna iade politikasındaki gün sayısını içeren tek bir HTML yorumu ekle. Başka dosyaya dokunma.

**Vektör araması:** Qdrant üzerinde 1 sorgu çalıştı (Colleague embedding + retrieve).

## iade-politikasi (`iade-politikasi`)

Colleague Lojistik — müşteri iade politikası (genel)

Hasarlı paket bildirimi
Müşteri, teslimattan itibaren 14 gün içinde hasarlı paket bildirimi yapabilir. Bu süre, iade talebinin değerlendirilmeye alınması için üst sınırdır.

Hasarlı gönderi iade sınırı
Hasarlı gönderi için iade sınırı 14 gündür. Bu süreyi aşan talepler otomatik onaylanmaz; destek ekibi incelemesi gerekir.

Diğer iade türleri
Ürün beğenilmedi veya yanlış sipariş gibi durumlar ayrı prosedüre tabidir. Bu belgede yalnızca hasarlı gönderi süresi sayısal olarak tanımlanmıştır.

İletişim
Sorularınız için uygulama içi destek veya account:*** adresini kullanın.

## copoints-yazilim-deployment (`copoints-yazilim-deployment`)

# Deployment ve yayın penceresi

## Production yayın saatleri

Canlı ortama (Production) kod yayını Cuma günleri saat **15:00** sonrasında yapılamaz.
Acil **hotfix** istisnası yalnız CTO veya nöbetçi mühendis onayı ile geçerlidir.

## Ön kontrol listesi

- Staging ortamında duman testi tamamlandı mı?
- Geri alma (rollback) planı ticket’a eklendi mi?
- Müşteri bildirimi gerektiren API değişikliği var mı?

## İletişim

Yayın sırasında #eng-deploy Slack kanalı açık tutulur.

## copoints-yazilim-kod-inceleme (`copoints-yazilim-kod-inceleme`)

COPoints Inc. — Yazılım ve Teknoloji

Pull Request (PR) kuralları
Master veya main branşına giden her PR en az iki senior yazılımcı onayı almadan birleştirilemez.
Onay veren kişiler kodun güvenlik, test kapsamı ve geri alınabilirlik açısından sorumludur.

Branch adlandırma
feature/, fix/, hotfix/ önekleri zorunludur. Ticket numarası branch adında yer almalıdır.

Test zorunluluğu
Kritik puan motoru ve API uçları için birim testi olmayan PR merge edilmez.

## copoints-finans-puan-maliyeti (`copoints-finans-puan-maliyeti`)

Puan maliyeti hesaplama notu

Standart puan birim maliyeti: 0,042 TL (2026 Q4 referans, KDV hariç).

Kampanya dönemlerinde geçici maliyet tablosu Finans tarafından duyurulur.

Muhasebe kaydı: 740 pazarlama gideri / 320 puan yükümlülüğü.

Sapma analizi aylık kapanışta yapılır.
