# Şirket belgesi parçaları

Bu dosya yalnızca referanstır; depoya yazılmaz.

**İstek:** P1 ilk yanıt 15 dakika; 5.000–50.000 TL masrafı Finans Direktörü onaylar — bunu README'ye not olarak ekle

**Vektör araması:** Qdrant üzerinde 1 sorgu çalıştı (Colleague embedding + retrieve).

## copoints-destek-kritik-eskalasyon (`copoints-destek-kritik-eskalasyon`)

Kritik eskalasyon akışı
COPoints Inc. | Şirket içi | Doküman
P1: 15 dakika içinde ilk yanıt, 2 saat çözüm hedefi.
İlk adım: müşteri iletişim sorumlusuna bilgi, ikinci adım: mühendislik nöbetçisi.
Durum güncellemesi en geç 30 dakikada bir ticket’a yazılır.

## copoints-yazilim-on-call-nobet (`copoints-yazilim-on-call-nobet`)

Nöbet (on-call) prosedürü
COPoints Inc. | Şirket içi | Doküman
P1 olayında nöbetçi mühendis 10 dakika içinde Slack #incidents kanalında yanıt verir.
P1 tanımı: API çökmesi veya puan işlenememesi (Müşteri İlişkileri SLA P1 ile uyumlu).
Eskalasyon sırası: nöbetçi → tech lead → CTO.
Olay kapanış raporu 24 saat içinde Confluence’a yazılır.

## copoints-destek-sla-matrisi (`copoints-destek-sla-matrisi`)

COPoints Inc. — Müşteri İlişkileri ve Destek

| Seviye | Sorun türü | İlk yanıt | Çözüm hedefi |
| P1 | Kritik API çökmesi / puan işlenememesi | 15 dakika | 2 saat |
| P2 | Raporlama veya panel erişimsizliği | 1 saat | 6 saat |
| P3 | Genel bilgi ve hesap ayarları | 4 saat | 24 saat |

P1 ticket açıldığında otomatik olarak #support-war-room kanalına düşer.

## copoints-yazilim-uretim-erisim (`copoints-yazilim-uretim-erisim`)

Üretim ortamı erişim listesi (gizli)

Bu belge yalnız yönetici (manager) rolüne açıktır.

Production SSH ve veritabanı break-glass erişimi: CTO, Platform Lead, Finans Direktörü (salt okunur rapor).

Erişim talepleri Jira SEC- erişim ticket’ı ile açılır; onay süresi en fazla 4 saattir.

Her break-glass oturumu otomatik kayda alınır ve haftalık denetimde incelenir.

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
