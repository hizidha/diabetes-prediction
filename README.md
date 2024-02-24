# Laporan Proyek Machine Learning - Adisaputra Zidha Noorizki


## Domain Proyek
Diabetes Mellitus merupakan salah satu masalah kesehatan global yang serius, dengan jumlah penderita yang terus meningkat setiap tahunnya. Menurut International Diabetes Federation (IDF), pada tahun 2019 saja, sekitar 537 juta orang di seluruh dunia didiagnosis menderita diabetes dengan rentang umur 20–79 tahun [1]. Angka ini diperkirakan akan terus meningkat menjadi 700 juta pada tahun 2045, jika tidak ada tindakan preventif yang dilakukan. Dampak yang ditimbulkan oleh diabetes juga sangat signifikan, termasuk komplikasi serius seperti penyakit jantung, stroke, gagal ginjal, gangguan penglihatan, dan amputasi [2]. Namun, sebagian besar kasus diabetes dapat dicegah atau dikendalikan dengan deteksi dini dan adopsi gaya hidup sehat.
Penggunaan teknologi Machine Learning dalam proyek ini menjadi sangat penting mengingat kompleksitas dan dampak dari masalah diabetes. Salah satu pendekatan yang menjanjikan adalah menggunakan Machine Learning untuk memprediksi risiko diabetes pada individu. Dengan memprediksi risiko diabetes, individu yang berisiko tinggi dapat diberikan perhatian khusus dan dapat mengadopsi gaya hidup yang lebih sehat atau menerima perawatan lebih awal untuk mengelola kondisi mereka [3]. Selain itu, dengan adanya prediksi ini, sistem kesehatan dapat mengalokasikan sumber daya dengan lebih efisien, mengarahkan perawatan lebih awal kepada individu yang berisiko tinggi, dan mengurangi beban penyakit yang terkait dengan diabetes.


## Business Understanding
Pentingnya pencegahan diabetes telah mendorong pengembangan berbagai pendekatan untuk mengidentifikasi individu yang berisiko tinggi mengembangkan penyakit ini. Salah satu pendekatan yang menjanjikan adalah menggunakan Machine Learning untuk memprediksi risiko diabetes pada individu berdasarkan data kesehatan. Dengan memanfaatkan data kesehatan yang tersedia, seperti riwayat medis, pola makan, aktivitas fisik, dan faktor-faktor risiko lainnya, model Machine Learning dapat dibangun untuk mengidentifikasi pola atau karakteristik yang mengindikasikan kemungkinan seseorang mengembangkan diabetes di masa depan. Hal ini memungkinkan untuk melakukan tindakan pencegahan atau intervensi lebih awal kepada individu yang berisiko tinggi, serta mengalokasikan sumber daya kesehatan dengan lebih efisien.

### Problem Statements
- Bagaimana cara mengidentifikasi individu dengan risiko tinggi mengembangkan diabetes secara dini?
- Bagaimana cara meningkatkan efisiensi pengelolaan sumber daya kesehatan dalam menangani diabetes dengan adanya prediksi yang tepat tentang populasi yang berisiko tinggi dengan pemanfaatan teknologi Machine Learning?

### Goals
- Membangun model Machine Learning yang dapat memprediksi risiko diabetes pada individu dengan tingkat akurasi yang tinggi, sehingga memungkinkan untuk identifikasi dini individu yang berisiko tinggi dengan menggunakan metriks evaluasi accuracy, precision, recall, dan F1-score.
- Mengembangkan sistem prediksi diabetes yang dapat digunakan untuk mengalokasikan sumber daya kesehatan dengan lebih efisien, sehingga mengoptimalkan perawatan untuk individu yang berisiko tinggi dan mengurangi beban penyakit secara keseluruhan.

### Solution statements
- Penggunaan Multiple Algorithms: Melakukan pemodelan menggunakan beberapa algoritma Machine Learning seperti Random Forest, Gaussian Naive Bayes, dan Gradient Boosting untuk membandingkan performa dan memilih model terbaik.
- Optimasi Hyperparameter: Melakukan penyetelan hyperparameter pada model yang dipilih untuk meningkatkan performa prediksi, seperti tuning parameter untuk mendapatkan model yang lebih akurat dan efisien.
- Metrik Evaluasi: Evaluasi performa model menggunakan metrik yang relevan seperti accuracy, precision, recall, dan F1-score sehingga dapat mengukur efektivitas dari setiap solusi yang diusulkan secara objektif.


## Data Understanding
Dataset prediksi Diabetes adalah kumpulan data medis dan demografis dari pasien, disertai dengan status diabetes mereka (positif atau negatif). Data ini mencakup berbagai fitur seperti usia, jenis kelamin, indeks massa tubuh (BMI), hipertensi, penyakit jantung, riwayat merokok, kadar HbA1c, dan kadar glukosa darah. Sumber data ini berasal dari open-source dataset yang diunggah oleh Mohammed Mustafa dengan judul "**Diabetes Prediction**" yang dapat diakses melalui dan dapat diakses melalui [link ini](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Dataset ini tercatat terakhir diperbarui 10 bulan yang lalu, saat ini dataset tersebut memiliki 1 file dengan nama `diabetes_prediction_dataset.csv`.

Dataset ini dapat digunakan untuk membangun model pembelajaran mesin untuk memprediksi diabetes pada pasien berdasarkan riwayat medis dan informasi demografis mereka. Hal ini dapat berguna bagi para profesional kesehatan dalam mengidentifikasi pasien yang mungkin berisiko terkena diabetes dan dalam mengembangkan rencana perawatan yang dipersonalisasi. Selain itu, dataset ini dapat digunakan oleh para peneliti untuk mengeksplorasi hubungan antara berbagai faktor medis dan demografis dan kemungkinan terkena diabetes.

Fitur pada Dataset akan dipaparkan dalam tabel berikut:
| Nama variabel | Deskripsi |
| ------------- | --------- |
| gender | jenis kelamin (biologis) |
| age | usia |
| hypertension | tekanan darah (normal/tinggi) |
| heart_disease | penderita penyakit jantung (ya/tidak) |
| smoking_history | riwayat merokok (6 categori) |
| bmi | ukuran lemak tubuh berdasarkan berat dan tinggi badan |
| HbA1c_level | level kadar gula (Hemoglobin A1c) |
| blood_glucose_level | kadar glukosa darah |
| diabetes | arget prediksi (1 atau 0) |

Selanjutnya adalah memvalidasi apakah ada data outliers di dalam dataset yang telah dijelaskan sebelumnya. Dengan menggunakan bantuan box plotting, proses tersebut dapat dilihat pada Gambar 01.

![outliers_data](https://github.com/hizidha/diabetes-prediction/blob/main/assets/data_outliers.png?raw=true)

*Gambar 1. Outliers data pada Dataset*

Terdapat beberapa data outliers yang terdeteksi, namun setelah dilakukan **drop data outliers**, ada varian data di beberapa fitur dataset hilang, terutama pada fitur yang hanya memiliki 2 varian data saja seperti `hypertension`, `heart_disease`, serta fitur yang berguna untuk label prediksi `diabetes` juga kehilangan data yang bernilai 0. Maka dari itu, studi kasus ini akan menggunakan data tanpa menghilangkan data outliers. Karena data outliers juga dapat berisi informasi penting atau representasi dari kejadian langka yang memang valid. 

Sebelum melangkah ke tahap berikutnya, ada hal yang perlu dilakukan yaitu mengkategorikan fitur pada dataset menjadi dua bagian, yaitu numerical features dan categorical features. 
```
numerical = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
categorical = ['gender', 'smoking_history']
```
Tahap berikutnya adalah proses analisis data yang akan terbagi menjadi 2 bagian utama antara lain:
### Univariate Analysis
Analisis ini bertujuan untuk mengetahui distribusi dan karakteristik dari setiap variabel secara terpisah.
##### Categorical

![gender](https://github.com/hizidha/diabetes-prediction/blob/main/assets/gender.png?raw=true)

*Gambar 2. Distribusi data `gender` pada Dataset*

![smoking_his](https://github.com/hizidha/diabetes-prediction/blob/main/assets/smoking_history.png?raw=true)

*Gambar 3. Distribusi data `smoking_history` pada Dataset*

dengan keterangan berbentuk tabel sebagai berikut:
|   | jumlah sampel | persentase |
|---| ------------- | --------- |
| No Info     | 35810  | 35.8 %
| never       | 35092  | 35.1 %
| former      |  9352  | 9.4 %
| current     | 9286   | 9.3 %
| not current | 6439   | 6.4 %
| ever        | 4003   | 4.0 %

##### Numerik

![numerik_analyst](https://github.com/hizidha/diabetes-prediction/blob/main/assets/numerik.png?raw=true)

*Gambar 4. Distribusi data numerik pada dataset*

### Multivariate Analysis
Analisis ini bertujuan untuk mengetahui hubungan antara dua atau lebih variabel pada dataset. Berikut merupakan grafik atau plot untuk korelasi antar fitur yang memiliki data numerik serta grafik untuk menampilkan relasi antara data categorical dan fitur `diabetes` yang dapat dilihat pada Gambar 5, 6, dan 7.

![relatif](https://github.com/hizidha/diabetes-prediction/blob/main/assets/relatif.png?raw=true)

*Gambar 5. Relatif antara `gender` terhadap `diabetes`*

![relatif2](https://github.com/hizidha/diabetes-prediction/blob/main/assets/relatif2.png?raw=true)

*Gambar 6. Relatif antara `smoking_history` terhadap `diabetes`*

![korelasi](https://github.com/hizidha/diabetes-prediction/blob/main/assets/pairplot.png?raw=true)

*Gambar 7. Korelasi data numerik antar fitur pada dataset*

![correlation-matrix](https://github.com/hizidha/diabetes-prediction/blob/main/assets/matrix.png?raw=true)

*Gambar 8. Correlation Matrix untuk fitur numerik*

Setelah melakukan beberapa tahap analisis, terdapat beberapa hal yang dapat disimpulkan dari proses analisis di atas antara lain:
1. Hubungan Antar Fitur
  * Terdapat beberapa hubungan positif yang signifikan antara beberapa pasangan fitur, seperti:
    * Usia (age) memiliki korelasi positif yang cukup kuat dengan indeks massa tubuh (bmi) (0.34).
    * Tingkat HbA1c (HbA1c_level) memiliki korelasi positif yang signifikan dengan tingkat glukosa darah (blood_glucose_level) (0.17).
    * Diabetes memiliki korelasi yang kuat dengan HbA1c_level (0.40) dan tingkat glukosa darah (0.42).
  * Namun, sebagian besar korelasi antara fitur-fitur numerik tidak sangat tinggi, yang menunjukkan bahwa fitur-fitur tersebut tidak saling tergantung secara linear.

2. Korelasi dengan Variabel Target (Diabetes):
  * Korelasi antara variabel target (diabetes) dengan variabel prediktor menunjukkan beberapa hubungan yang signifikan:
    * Tingkat HbA1c memiliki korelasi tertinggi dengan diabetes (0.40), menunjukkan bahwa tingkat HbA1c dapat menjadi indikator yang kuat untuk risiko diabetes.
    * Tingkat glukosa darah juga memiliki korelasi yang cukup tinggi dengan diabetes (0.42), menunjukkan hubungan yang kuat antara tingkat glukosa darah dan diabetes.
    * Usia (0.26) dan indeks massa tubuh (0.21) juga memiliki korelasi yang cukup tinggi dengan diabetes, menunjukkan bahwa kedua fitur ini juga dapat digunakan sebagai prediktor risiko diabetes.


## Data Preparation
Pada tahap ini, akan ada beberapa langkah atau tahapan yang harus dilakukan agar dataset dapat siap digunakan sebagai bahan pelatihan model machine learning. Berikut tahapan yang dilakukan:

### Preprocessing data kategorical
Dalam tahap preprocessing data kategorikal, salah satu langkah yang diterapkan adalah penggunaan **one hot encoder** untuk mengubah fitur atau kolom yang memiliki tipe data kategorikal menjadi representasi biner yang sesuai. Dalam konteks ini, fitur `gender` dan `smoking_history` adalah contoh kolom dengan tipe data kategorikal. Seperti misalnya pada fitur `gender` terdapat dua variasi data yaitu `Male` dan `Female` yang kemudian agar diubah menjadi numerik agar dapat dikenali oleh model.

| gender |
| ------ |
| Male   |
| Female |
| Female |

setelah dilakukan proses `OneHotEncoder`, tabel menjadi seperti berikut:

| gender_Female | gender_Male | 
| ------------- | ----------- |
| 0 | 1 |
| 1 | 0 |
| 1 | 0 |


### Split dataset
Dataset yang dimiliki akan dibagi menjadi dua bagian utama: satu untuk `training` dan satu lagi untuk `test`. Tujuannya adalah untuk menggunakan bagian training untuk melatih model, sementara bagian test untuk mengevaluasi kinerja model. Biasanya, dataset dibagi secara acak dengan proporsi seperti 80% data untuk pelatihan dan 20% untuk test memastikan representasi yang baik dari seluruh dataset. Proses ini penting dalam pengembangan model machine learning untuk memastikan bahwa model yang dihasilkan mampu melakukan prediksi dengan akurasi yang baik pada data baru yang belum pernah dilihat sebelumnya.

| keterangan    | jumlah data/shape | 
| ------------- | ----------------- |
| data training | (79985, 14)       |
| data testing  | (19997, 14)       |


### Standarisasi nilai data
Proses standarisasi nilai data pada setiap fitur dataset mengacu pada proses mengubah skala nilai pada setiap fitur sehingga memiliki mean 0 dan deviasi standar 1. Ini dilakukan dengan mengurangkan rata-rata fitur dari setiap titik data dan kemudian membagi hasilnya dengan deviasi standarnya. Dengan melakukan standarisasi, kita dapat memastikan bahwa model tidak akan terlalu dipengaruhi oleh perbedaan skala antarfitur, sehingga memungkinkan interpretasi yang lebih baik dari koefisien model dan hasil yang lebih konsisten saat diterapkan pada data baru.

### Implementasi Metode RUS (*Random Under Sampling*)
RUS merupakan teknik yang digunakan untuk menangani ketidakseimbangan kelas dalam dataset dengan mengurangi jumlah sampel dari kelas mayoritas sehingga seimbang dengan jumlah sampel dari kelas minoritas. Proses ini dilakukan dengan cara secara acak menghapus sebagian sampel dari kelas mayoritas hingga jumlahnya seimbang dengan jumlah sampel dari kelas minoritas.

Sehingga setelah melewati beberapa tahapan pada **Data Preparation** ini, berikut adalah sebaran data yang akan digunakan untuk pelatihan model:
| Kelas/label | Jumlah data Sebelum DP | Jumlah data Sesudah DP |
|-------------|------------------------|------------------------|
| 0 (normal)  | 73260                  | 6725                   |
| 1 (diabetes)| 6725                   | 6725                   |

Dapat dilihat dari tabel di atas bahwa jumlah data pada salah satu label memiliki perbedaan yang signifikan dengan label lainnya. Hal ini dapat mengakibatkan model memiliki kecenderungan untuk memihak kepada label yang jumlahnya mayoritas saja, tanpa memperhatikan label lain yang memiliki jumlah lebih sedikit. Oleh karena itu, implementasi metode Random Under-Sampling (RUS) diperlukan dalam studi kasus ini.


## Modeling
Dalam tahap ini menggunakan beberapa algoritma machine learning antara lain Random Forest, Gaussian Naive Bayes, dan XGBoost. Memilih ketiga model tersebut karena memiliki kriteria yang termasuk ke dalam kelebihan dan kekurangan mereka masing-masing yang dapat dilihat melalui tabel berikut:

1. Random Forest
   
   Random Forest adalah algoritma ensemble learning yang memanfaatkan sejumlah besar decision tree [4]. Pada setiap iterasi, algoritma ini membagi dataset menjadi subset acak dan membangun decision tree pada setiap subset ini. Proses ini dilakukan sejumlah `n_estimators` kali, di mana setiap decision tree memiliki kemungkinan untuk mengambil subset yang berbeda dari data.

2. Gaussian Naive Bayes

   Gaussian Naive Bayes adalah salah satu varian dari Naive Bayes, sebuah algoritma klasifikasi yang berdasarkan pada teorema Bayes dengan asumsi bahwa fitur-fitur yang digunakan dalam model adalah independen satu sama lain [5]. Dalam kasus Gaussian Naive Bayes, diasumsikan bahwa distribusi nilai fitur-fitur pada setiap kelas mengikuti distribusi Gaussian (normal). Ini berarti bahwa fitur-fitur numerik dianggap sebagai berdistribusi normal di antara kelas-kelas yang berbeda.
   
   Proses pelatihan model Gaussian Naive Bayes melibatkan perhitungan mean dan variansi dari setiap fitur pada setiap kelas. Selama inferensi, probabilitas posterior untuk setiap kelas dihitung berdasarkan teorema Bayes, yang melibatkan perkalian probabilitas prior dan likelihood dari setiap fitur.

3. XGBoost

   XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang bekerja dengan membangun serangkaian model kecil secara bertahap. Pada setiap iterasi, XGBoost memperbarui model dengan menambahkan pohon keputusan baru yang dirancang untuk memperbaiki kesalahan dari model sebelumnya [6]. Proses ini dilakukan dengan meminimalkan fungsi objektif yang ditentukan, yang biasanya berupa fungsi kerugian.
   
   XGBoost memiliki beberapa parameter penting seperti `learning_rate` yang mengontrol kontribusi setiap pohon keputusan dalam memperbarui model, `max_depth` yang mengatur kedalaman maksimum dari setiap pohon, dan `min_child_weight` yang merupakan bobot minimum yang diperlukan untuk membagi node dalam pohon.

Atau secara singkat, dapat dilihat melalui tabel dibawah ini:

| Keterangan | Random Forest | Gaussian Naive Bayes | XG Boosting |
|------------|---------------|---------------|-------------|
| Tahapan Pemodelan | Menggunakan ensemble learning untuk membuat sejumlah besar decision tree yang diambil secara acak dari subset data | Membangun model berdasarkan asumsi bahwa fitur-fitur adalah independen satu sama lain, diberikan kelasnya | Menggunakan teknik boosting yang membangun model secara bertahap dengan mengurangi kesalahan dari iterasi sebelumnya |
| Parameter yang Digunakan | `n_estimators`, `max_depth`, `max_features` | `Tidak ada parameter yang dapat diatur secara langsung` | `eta`, `num_boost_round`, `max_depth` |
| Kelebihan | Tidak cenderung overfitting karena mengambil rata-rata prediksi dari banyak pohon | GNB memiliki sedikit atau bahkan tidak ada parameter yang perlu disetel. | Tidak rentan terhadap overfitting karena menggunakan regularisasi. |
| Kekurangan | Cenderung lebih lambat dalam pelatihan dan inferensi daripada model decision tree tunggal | GNB sering kali kalah dalam hal performa dengan model yang lebih kompleks, terutama pada dataset yang besar dan kompleks. | Membutuhkan penyetelan parameter yang lebih teliti untuk mengoptimalkan kinerja |

Berdasarkan proses pelatihan yang telah dilakukan serta hasil yang telah didapat, maka model yang memberikan nilai untuk setiap metrik evaluasi terbaik adalah model `XGBoots`, yang kemudian akan dibahas lebih lanjut pada bagian **Evaluation**. Dengan demikian, berikut secara lengkap parameter yang digunakan:
```
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    random_state=42
)
```

| Parameter          | Deskripsi |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| n_estimators  | Jumlah pohon keputusan yang akan dibangun dalam ensemble      |
| learning_rate | Tingkat pembelajaran yang digunakan dalam proses optimisasi   |
| max_depth     | Kedalaman maksimum dari setiap pohon keputusan dalam ensemble |
| min_child_weight  | Bobot minimum yang diperlukan untuk membagi node pohon    |
| gamma         | Mengontrol penurunan minimum dalam nilai fungsi objektif |
| subsample     | Subset dari dataset yang akan digunakan dalam setiap iterasi pembelajaran |
| colsample_bytree   | Subset dari fitur yang akan digunakan dalam pembangunan setiap pohon |
| scale_pos_weight   | Bobot untuk menyeimbangkan kelas positif dan negatif dalam kasus ketidakseimbangan kelas |
| random_state       | Seed untuk mengontrol randomness dalam pembangunan model |


## Evaluation
Setelah melewati berbagai tahapan yang telah dipaparkan sebelumnya, tahap ini akan berfokus pada evaluasi kinerja model dalam melakukan prediksi diabetes dengan data kuantitatif. Dalam studi kasus ini menggunakan dua metrik evaluasi utama yaitu `accuracy`, `precision`, `recall`, `f1-score`.

1. Accuracy/Akurasi
   
   Akurasi mengukur seberapa banyak prediksi model yang benar dari total prediksi yang dilakukan. Ini adalah metrik yang umum digunakan dalam klasifikasi dan memberikan pemahaman tentang seberapa baik model dapat mengklasifikasikan data dengan benar. Akurasi dihitung dengan membagi jumlah prediksi yang benar dengan total jumlah prediksi.

   ![akurasi](https://github.com/hizidha/diabetes-prediction/blob/main/assets/akurasi.png?raw=true)

   *Gambar 9. Rumus metrik evaluasi `accuracy`*
   
2. Precision/Presisi

   Presisi mengukur seberapa banyak dari prediksi positif yang sebenarnya benar. Ini memberikan pemahaman tentang seberapa baik model dalam menghindari memberikan hasil positif palsu. Presisi dihitung dengan membagi jumlah prediksi positif yang benar dengan total prediksi positif yang dilakukan.

   ![presisi](https://github.com/hizidha/diabetes-prediction/blob/main/assets/presisi.png?raw=true)

   *Gambar 10. Rumus metrik evaluasi `precision`*

3. Recall
   
   Recall, juga dikenal sebagai sensitivitas, mengukur seberapa banyak dari kelas yang sebenarnya positif yang telah diidentifikasi dengan benar oleh model. Ini memberikan pemahaman tentang seberapa baik model dapat mengenali semua contoh yang positif.   

   ![recall](https://github.com/hizidha/diabetes-prediction/blob/main/assets/recall.png?raw=true)

   *Gambar 11. Rumus metrik evaluasi `recall`*
   
4. F1-Score
   
   F1-score adalah ukuran yang menggabungkan presisi dan recall menjadi satu angka tunggal. Ini memberikan pemahaman tentang keseimbangan antara presisi dan recall dari model. F1-score adalah rata-rata harmonis dari presisi dan recall, dan memberikan bobot yang sama terhadap keduanya.   

   ![f1_score](https://github.com/hizidha/diabetes-prediction/blob/main/assets/f1_score.png?raw=true)

   *Gambar 12. Rumus metrik evaluasi `f1-score`

keterangan terkait rumus metrik evaluasi di atas:
| Metrik         | Deskripsi    |
|----------------|--------------|
| True Positive (TP)  | Jumlah sampel positif yang berhasil diprediksi dengan benar oleh model.     |
| True Negative (TN)  | Jumlah sampel negatif yang berhasil diprediksi dengan benar oleh model.     |
| False Positive (FP) | Jumlah sampel negatif yang salah diprediksi sebagai positif oleh model.   |
| False Negative (FN) | Jumlah sampel positif yang salah diprediksi sebagai negatif oleh model.   |

Berikut bentuk visualisasi terkait nilai yang dihasilkan masing-masing model setelah melakukan pelatihan dataset yang sebelumnya telah diproses sedemikian rupa.

![confusion_metrik](https://github.com/hizidha/diabetes-prediction/blob/main/assets/confusion_metrik.png?raw=true)

*Gambar 13. Perbandingan nilai metrik evaluasi dari setiap model (Random Forest, Gaussian Naive Bayes, XGBoosting)*

| Model                | Accuracy | Precision | Recall  | F1-Score |
|----------------------|----------|-----------|---------|----------|
| Random Forest        | 0.955 | 0.949  | 0.961 | 0.955 |
| Gaussian Naive Bayes | 0.832 | 0.853  | 0.801 | 0.826 |
| XGBoost              | 0.996 | 0.996  | 0.995 | 0.996 |


## Conclusion
Berdasarkan metrik evaluasi yang disajikan, terlihat bahwa model XGBoost menunjukkan performa yang paling baik dibandingkan dengan Random Forest dan Gaussian Naive Bayes. XGBoost memiliki nilai tertinggi untuk semua metrik evaluasi, yaitu akurasi (0.996), presisi (0.996), recall (0.995), dan F1-Score (0.996). Hal ini menunjukkan bahwa XGBoost berhasil mengklasifikasikan data uji dengan sangat baik, dengan tingkat akurasi yang sangat tinggi dan kemampuan yang sangat baik dalam mengidentifikasi kelas positif (recall) serta menghindari memberikan hasil positif palsu (presisi).

Di sisi lain, meskipun Random Forest menunjukkan performa yang baik dengan akurasi 0.955 dan F1-Score 0.955, model ini sedikit kalah dibandingkan dengan XGBoost dalam hal presisi dan recall. Gaussian Naive Bayes menunjukkan performa yang paling rendah di antara ketiga model, dengan akurasi 0.832 dan F1-Score 0.826. Meskipun memiliki presisi yang sedikit lebih tinggi daripada Random Forest, Gaussian Naive Bayes memiliki recall yang lebih rendah, yang menunjukkan bahwa model ini lebih cenderung untuk melewatkan contoh positif.

Dengan demikian, berdasarkan metrik evaluasi yang digunakan, **model XGBoost** dapat disimpulkan sebagai yang paling baik dalam melakukan klasifikasi pada dataset yang diberikan.


## Reference:
[1] Pecoits-Filho, R., Jimenez, B. Y., Ashuntantang, G. E., de Giorgi, F., De Cosmo, S., Groop, P. H., ... & Ceriello, A. (2023). A policy brief by the International Diabetes Federation and the International Society of Nephrology. Diabetes Research and Clinical Practice, 203.

[2] Hendrawan, S., Nathaniel, F., Satyanegara, W. G., Wijaya, D. A. W., Kusuma, K. F., Gracienne, G., ... & Santoso, A. H. (2023). KEGIATAN PENGABDIAN MASYARAKAT BERUPA PENYULUHAN DAN SKRINING HBA1C DALAM RANGKA MENINGKATKAN KESADARAN MASYARAKAT TERHADAP DIABETES MELITUS TIPE 2. Community Development Journal: Jurnal Pengabdian Masyarakat, 4(6), 12077-12083.

[3] Cusi, K., Isaacs, S., Barb, D., Basu, R., Caprio, S., Garvey, W. T., ... & Younossi, Z. (2022). American Association of Clinical Endocrinology clinical practice guideline for the diagnosis and management of nonalcoholic fatty liver disease in primary care and endocrinology clinical settings: co-sponsored by the American Association for the Study of Liver Diseases (AASLD). Endocrine Practice, 28(5), 528-562.

[4] Ghiasi, M. M., & Zendehboudi, S. (2021). Application of decision tree-based ensemble learning in the classification of breast cancer. Computers in biology and medicine, 128, 104089.

[5] Bafjaish, S. S. (2020). Comparative analysis of Naive Bayesian techniques in health-related for classification task. Journal of Soft Computing and Data Mining, 1(2), 1-10.

[6] Shehadeh, A., Alshboul, O., Al Mamlook, R. E., & Hamedat, O. (2021). Machine learning models for predicting the residual value of heavy construction equipment: An evaluation of modified decision tree, LightGBM, and XGBoost regression. Automation in Construction, 129, 103827.