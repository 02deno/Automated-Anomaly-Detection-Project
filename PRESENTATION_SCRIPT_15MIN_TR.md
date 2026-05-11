# AutoAD 15 Dakikalik Sunum Scripti

Hedef sure: 13:30-14:30 dakika. Kalan 30-90 saniye soru payi.

Demo oncesi hazirlik:

```powershell
cd "C:\Users\LENOVO\Desktop\telecom\bahar 25\ai\anomaly_detection\Automated-Anomaly-Detection-Project\api"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Tarayici: `http://127.0.0.1:8000/`

Demo icin en hizli dosya: `data/test_data.txt`

## 0:00-0:45 - Acilis

Merhaba, biz Team Anomaly. Bu projede amacimiz, farkli CSV tabanli veri setlerinde manuel model secimi ve elle threshold ayari ihtiyacini azaltan otomatik bir anomaly detection sistemi gelistirmekti.

Kisa fikir su: kullanici bir tablo yukluyor; sistem verinin sayisal kolonlarini analiz ediyor, uygun modelleri secip calistiriyor, skorlarini normalize edip birlestiriyor, sonra hangi satirlarin anomali oldugunu UI uzerinden gosteriyor.

## 0:45-1:20 - Agenda

Sunumu dort parcaya indirecegim: once problem ve mimari, sonra veri ve sentetik anomaly injection, sonra evaluation ve robustness, en sonda da live demo. Demo kismina ozellikle zaman ayiracagim cunku sistemin asil katkisi UI ile birlikte daha net gorunuyor.

## 1:20-2:30 - Workflow

Workflow uc ana adimdan olusuyor: data ingestion, modelleme ve karar/veri gorsellestirme.

Ilk adimda FastAPI backend CSV dosyasini aliyor. `api/main.py` icinde `/upload`, `/eda`, `/synthetic-preview`, `/synthetic-export` ve `/overfit-check` endpointleri var.

Ikinci adimda `AdvancedAnomalySystem` devreye giriyor. Sistem once veriyi analiz ediyor: sample sayisi, feature sayisi, missing rate, variance, skewness, kurtosis, korelasyon ve sparsity gibi meta ozellikleri cikariyor.

Sonra bu profile gore Isolation Forest, One-Class SVM, LOF, KNN distance, autoencoder ve LSTM gibi modellerden uygun olanlari calistiriyor. Her modelin anomaly score'u ortak bir olcege normalize ediliyor ve ensemble score olusturuluyor.

## 2:30-3:20 - Architecture

Mimaride backend ve frontend ayrik. Backend FastAPI; UI ise `ui/index.html` icinde statik dashboard olarak servis ediliyor.

Kod seviyesinde moduller ayrilmis durumda:

`eda_report.py` yalnizca kesifsel veri analizi uretiyor; ML calistirmiyor.

`synthetic_injection.py` kontrollu anomaly senaryolari uretiyor.

`advanced_system.py` asil pipeline: preprocessing, model secimi, optimizasyon, ensemble ve threshold.

`overfit_diagnostic.py` ise threshold seciminin label bilgisine fazla uyup uymadigina dair risk analizi yapiyor.

Bu ayrim onemli, cunku demo sirasinda da UI'da uc farkli is parcasini ayri kartlar olarak gorecegiz.

## 3:20-4:10 - Datasets

Projede iki tur veriyle calistik.

Birincisi kucuk demo verileri: mesela `data/test_data.txt` icinde CPU usage, memory usage, network traffic, response time ve label kolonlari var. Bu dosya UI demosu icin hizli calisir.

Ikincisi real-data evaluation tarafinda Annthyroid ve KDD'99 HTTP/SMTP gibi label'li veri setleri kullanildi. Burada label kolonu modelin feature'i olarak kullanilmiyor; sadece evaluation asamasinda precision, recall, F1, ROC-AUC ve PR-AUC hesaplamak icin kullaniliyor.

## 4:10-5:20 - Synthetic Anomaly Injection

Sentetik anomaly injection projenin test edilebilirlik kismi. Gercek veride her zaman temiz label bulmak zor oldugu icin, kontrollu bozulmalar ekleyip sistemin bunlari yakalayip yakalamadigini olctuk.

Kodda sekiz senaryo var: `spike_single`, `joint_shift`, `scale_burst`, `dead_sensor`, `sign_flip`, `temporal_block`, `categorical_flip` ve `missing_value`.

Ornegin `spike_single` belirli satirlarda bir sayisal kolona standart sapma cinsinden buyuk bir artis ekliyor. `dead_sensor` sensorun sabit deger uretmesi gibi davraniyor. `temporal_block` ise ardisik bir zaman blogunu bozuyor.

Onemli nokta: injection sonucunda `y_true` uretiliyor ama detector bu kolonu feature olarak gormuyor. `y_true` sadece evaluation icin kullaniliyor.

## 5:20-6:40 - Evaluation Methodology

Evaluation iki seviyede yapildi.

Birinci seviye threshold bagimli metrikler: precision, recall, F1 ve confusion matrix. Bunlar "hangi satirlari anomali diye isaretledik?" sorusunu cevapliyor.

Ikinci seviye threshold bagimsiz metrikler: ROC-AUC ve PR-AUC. Bunlar skor siralamasinin kaliteli olup olmadigini olcuyor. Bu ayrim onemli, cunku model iyi siralama yapabilir ama threshold cok katiysa recall dusuk kalabilir.

Pipeline default olarak anomaly score'u threshold ile karsilastiriyor. Kodda `PostProcessingLayer.label` mantigi basit: score threshold'dan buyukse anomali.

## 6:40-7:45 - Results ve Robustness

Sonuclar sunumun ana mesajini destekliyor: tek bir model her veri setinde en iyi degil.

Kucuk sentetik benchmarkta ensemble, spike ve joint shift gibi belirgin senaryolarda cok guclu calisiyor. Real-data tarafinda ise model davranisi veri setine gore degisiyor; KDD HTTP'de OCSVM ve freeze gibi kaynaklar guclu cikarken Annthyroid'de ensemble ve temporal-change daha anlamli skorlar verebiliyor.

Bu nedenle sistemde ensemble ve meta-selection fikri var: amac tek bir modeli sabit secmek degil, veri profilinden ve skor davranisindan daha uyarlanabilir bir karar vermek.

## 7:45-8:45 - Threshold ve Overfitting Problemi

Anomaly detection'ta threshold kritik. Cok dusuk threshold false positive'i artirir; cok yuksek threshold gercek anomalileri kacirir.

Eger label'li veri varsa threshold'u sadece ayni veri uzerinde en iyi F1'a gore secmek overfitting riski dogurur. Yani sistem "bu veri setinde iyi gorunur" ama yeni veri geldiginde ayni performansi vermeyebilir.

Bu yuzden kodda iki katman var: `/upload` cevabinda hizli bir `overfit_hint` donuyor, ayrica UI'da istenirse `/overfit-check` ile subsampled train/test check calistirilabiliyor. Bu ikinci kontrol daha yavas ama threshold seciminin ne kadar stabil oldugunu daha iyi gosteriyor.

## 8:45-12:30 - Live Demo

Simdi sistemi calistiriyorum. Backend FastAPI ile ayakta ve UI `127.0.0.1:8000` adresinden geliyor.

Ilk kart EDA. Burada `data/test_data.txt` dosyasini yukluyorum ve Run EDA diyorum. Bu kisimda model calismiyor; sadece veri profili cikiyor. Kolon tiplerini, missing value durumunu, numeric summary'yi, outlier ipuclarini, correlation heatmap'i ve scatter plot'u gorebiliyoruz. Bu adim bize "modelden once veri neye benziyor?" sorusunun cevabini veriyor.

Ikinci kart synthetic anomaly preview. Ayni dosyayi seciyorum, senaryo olarak mesela `spike_single` birakiyorum, seed 42. Preview dedigimde backend dosyayi aliyor, belirli satirlara kontrollu bozulma ekliyor ve UI'da before/after tabloyu gosteriyor. Sari vurgulu satirlar injected anomaly olan satirlar. Burada dikkat edilmesi gereken nokta: preview sadece ilk N satiri gostermek icin; export butonu ise butun bozulmus CSV'yi indiriyor.

Ucuncu kart full pipeline analysis. Burada yine `data/test_data.txt` dosyasini pipeline upload olarak seciyorum ve Run Analysis diyorum. Backend bu kez asil anomaly detection pipeline'ini calistiriyor.

Sonucta summary kisminda toplam kac satirin anomali oldugu geliyor. Evaluation card'da label kolonu varsa precision, recall, F1, accuracy, ROC-AUC ve PR-AUC gorunuyor. Score vs row index grafiginde kirmizi noktalar anomaly olarak isaretlenen satirlar. Histogram ise score dagilimini ve threshold'un nerede kaldigini gosteriyor.

Bir de dataset profile and decision rule karti var. Burada hangi numeric kolonlar kullanildi, hangi modeller secildi, threshold stratejisi neydi, PCA kullanildi mi gibi pipeline detaylarini gorebiliyoruz.

Son olarak overfitting and threshold sanity kartina bakiyorum. Bu kart, threshold secimi label'a ne kadar bagimli olabilir diye hizli bir uyari veriyor. Eger label'li daha buyuk bir dosya kullansaydik, buradaki "Run subsampled train/test check" butonu ile daha derin bir train/test kontrolu baslatabilirdik.

Bu demo uc seyi gosteriyor: veri profili ayri, sentetik test ayri, asil detection pipeline'i ayri. Yani sistem sadece "CSV yukle, sonuc al" degil; ayni zamanda sonucu yorumlamaya yardim eden bir dashboard.

## 12:30-13:30 - Limitations ve Future Work

Limitations tarafinda en onemli konu su: unsupervised anomaly detection'ta label yoksa kesin dogru/yanlis ayrimi yapmak zor. Bu yuzden metrikler ya label'li veri setlerinde ya da sentetik injection ile anlamli hale geliyor.

Ikinci limitasyon threshold secimi. En iyi F1'i bulmak kolay gorunuyor ama label'a fazla uyarsa generalization problemi yaratabilir. Bu nedenle holdout-based validation ve overfit diagnostic'i future work icin kritik.

Future work olarak daha fazla real dataset, daha iyi meta-selector calibration, daha hizli deep model opsiyonlari ve UI'da threshold'u interaktif ayarlama eklenebilir.

## 13:30-14:20 - Conclusion

Ozetle bu proje, yeni CSV veri setleri icin anomaly detection surecini daha otomatik ve aciklanabilir hale getirmeyi hedefliyor.

Katkilarimiz uc baslikta toplanabilir: otomatik model secimi ve ensemble pipeline'i, kontrollu synthetic anomaly injection ile test edilebilirlik, ve EDA/evaluation/overfit kontrollerini bir araya getiren kullanilabilir bir web dashboard.

Ana mesajimiz su: anomaly detection'ta tek model ve tek threshold her durumda yeterli degil. Bu yuzden veri profilini, model skorlarini, threshold davranisini ve evaluation riskini birlikte gosteren bir sistem daha guvenilir bir karar destek araci oluyor.

Tesekkurler, sorularinizi alabiliriz.

## Demo Sirasinda Tiklanacak Kisa Akis

1. `http://127.0.0.1:8000/` ac.
2. EDA kartinda `data/test_data.txt` sec, `Run EDA`.
3. Synthetic kartinda ayni dosyayi sec ya da pipeline dosyasina fallback kullan, `spike_single`, seed `42`, `Preview synthetic injection`.
4. Istersen `Download full corrupted CSV` butonunu goster ama indirmeyi bekletme.
5. Full pipeline kartinda `data/test_data.txt` sec, `Run Analysis`.
6. Sirasiyla Summary, Evaluation results, Overfitting & threshold sanity, Dataset profile, Score chart ve Results table alanlarini goster.

## Kisa Cevaplar

Soru: Label kolonu modeli etkiliyor mu?

Cevap: Hayir. `label`, `target`, `ground_truth`, `is_anomaly` gibi kolonlar feature'lardan cikariliyor; sadece evaluation icin kullaniliyor.

Soru: Threshold nasil seciliyor?

Cevap: Label yoksa default/fallback percentile veya meta-selected contamination mantigi kullaniliyor. Label varsa auto modda F1 tabanli secim yapilabiliyor; bu yuzden overfit hint ve train/test diagnostic ekledik.

Soru: Neden ensemble?

Cevap: Cunku farkli anomaly tipleri farkli modeller tarafindan daha iyi yakalaniyor. Ensemble, tek modele bagimliligi azaltmak icin skor kaynaklarini birlestiriyor.

Soru: Synthetic injection gercek anomaly yerine gecer mi?

Cevap: Hayir, tamamen yerine gecmez. Amaci kontrollu test ve robustness analizi. Real-data evaluation ile birlikte okunmasi gerekiyor.
