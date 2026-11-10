# Şirket belgesi parçaları

Bu dosya yalnızca referanstır; depoya yazılmaz.

**Niyet (sayı/süre buradan kopyalanmaz):** P1 destek ve masraf onay kurallarını şirket belgesinden okuyup README'ye kısa not olarak ekle

**Vektör araması:** Qdrant üzerinde 1 sorgu çalıştı (Colleague embedding + retrieve).

## copoints-finans-masraf-limitleri (`copoints-finans-masraf-limitleri`)

COPoints Inc. — Finans ve Muhasebe

5.000 TL ve altı harcamalar: Departman Yöneticisi onayı yeterlidir.
5.000 TL ile 50.000 TL arası: Finans Direktörü onayı gereklidir.
50.000 TL üzeri: CEO ve Yönetim Kurulu bilgilendirmesi zorunludur.

Masraf fişleri her ayın 25'ine kadar sisteme girilmelidir; aksi halde ödeme bir sonraki ay bordrosuna aktarılır.

## copoints-genel-rehber (`copoints-genel-rehber`)

COPoints | Inc.
KURUMSAL SADAKAT & PUANLAMA ÇÖZÜMLERİ
Genel İşletim Standartları ve Departman Prosedürleri
Doküman No: COP-POL-2026-v1 | Yayın Tarihi: Kasım 2026 | Gizlilik: Şirket İçi Kullanım
1. Şirket Genelinde Geçerli Temel Kurallar (All Departments)
COPoints ailesinin tüm mensupları, pozisyon ve departman fark etmeksizin aşağıdaki temel çalışma
standartlarına ve etik kurallara uymakla yükümlüdür:
Güvenlik ve Veri Gizliliği: Tüm müşteri puan verileri, API anahtarları ve kullanıcı veritabanı bilgileri yüksek
gizlilik derecesine sahiptir. İki faktörlü doğrulama (2FA) tüm kurumsal hesaplarda zorunludur.
Mesai ve Çalışma Modeli: Şirketimiz hibrit çalışma düzenini benimser. Çekirdek çalışma saatleri 10:00 -
16:00 arasındadır. Toplantılar ve senkron iletişim bu saatler aralığında planlanır.
İletişim Kanalları: Anlık iletişim için Slack, resmi onaylar ve harici bildirimler için kurumsal e-posta adresi
(@copoints.com) kullanılır.
Önemli Etik Kuralı
COPoints altyapısı üzerinde doğrudan veya dolaylı olarak puan manipülasyonu yapmak,
test ortamları haricinde kişisel hesaplara sanal puan tanımlamak kesin iş akdi feshi
sebebidir.
2. Departman Bazlı Özel Prosedürler
2.1. Yazılım & Teknoloji Departmanı (Engineering)
Sistem güvenliği ve performans standartlarını korumak amacıyla yazılım ekibi şu kuralları uygular:
Kod İncelemesi (Code Review): Master/Main branşına atılan her Pull Request (PR) en az 2 senior yazılımcı
tarafından onaylanmalıdır.
Deployment Saatleri: Cuma günleri saat 15:00'ten sonra acil hotfix dışında canlı ortama (Production) kod
yayını yapılamaz.
2.2. Müşteri İlişkileri & Destek Departmanı (Customer Success)
SLA Seviyesi Sorun Türü İlk Yanıt Süresi Çözüm Hedefi
P1 - Kritik API Çökmesi / Puan İşlenememesi < 15 Dakika 2 Saat
P2 - Yüksek Raporlama/Panel Erişimsizliği < 1 Saat 6 Saat
P3 - Normal Genel Bilgi ve Hesap Ayarları < 4 Saat 24 Saat
2.3. Finans & Muhasebe Departmanı
Harcama ve masraf onay süreçleri aşağıdaki limitlere tabi olarak yürütülür:
5.000 TL ve altı: Departman Yöneticisi onayı yeterlidir.
5.000 TL - 50.000 TL arası: Finans Direktörü onayı gereklidir.
Masraf Bildirimleri: Her ayın 25'ine kadar sisteme girilmeyen harcama fişleri bir sonraki ayın bordrosuna
aktarılır.
COPoints Inc. © 2026 — İç Doküman Sayfa 1 / 1

## copoints-destek-sla-matrisi (`copoints-destek-sla-matrisi`)

COPoints Inc. — Müşteri İlişkileri ve Destek

| Seviye | Sorun türü | İlk yanıt | Çözüm hedefi |
| P1 | Kritik API çökmesi / puan işlenememesi | 15 dakika | 2 saat |
| P2 | Raporlama veya panel erişimsizliği | 1 saat | 6 saat |
| P3 | Genel bilgi ve hesap ayarları | 4 saat | 24 saat |

P1 ticket açıldığında otomatik olarak #support-war-room kanalına düşer.

## copoints-yazilim-kod-inceleme (`copoints-yazilim-kod-inceleme`)

COPoints Inc. — Yazılım ve Teknoloji

Pull Request (PR) kuralları
Master veya main branşına giden her PR en az iki senior yazılımcı onayı almadan birleştirilemez.
Onay veren kişiler kodun güvenlik, test kapsamı ve geri alınabilirlik açısından sorumludur.

Branch adlandırma
feature/, fix/, hotfix/ önekleri zorunludur. Ticket numarası branch adında yer almalıdır.

Test zorunluluğu
Kritik puan motoru ve API uçları için birim testi olmayan PR merge edilmez.

## copoints-yazilim-on-call-nobet (`copoints-yazilim-on-call-nobet`)

Nöbet (on-call) prosedürü
COPoints Inc. | Şirket içi | Doküman
P1 olayında nöbetçi mühendis 10 dakika içinde Slack #incidents kanalında yanıt verir.
P1 tanımı: API çökmesi veya puan işlenememesi (Müşteri İlişkileri SLA P1 ile uyumlu).
Eskalasyon sırası: nöbetçi → tech lead → CTO.
Olay kapanış raporu 24 saat içinde Confluence’a yazılır.
