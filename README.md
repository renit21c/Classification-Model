# Orange vs Grapefruit Classification
Klasifikasi buah orange dan grapefruit menggunakan tiga algoritma machine learning: Decision Tree, Naive Bayes, dan Support Vector Machine (SVM)

## Struktur Project

```
Classification_Model/
├── classification.py          # Script Python
├── citrus.csv                 # Dataset
├── requirements.txt           # Library
├── eda_plots.png              # Output: distribusi fitur
├── correlation_heatmap.png    # Output: heatmap
├── confusion_matrices.png     # Output: confusion matrix ketiga model
├── model_comparison.png       # Output: grafik perbandingan model
├── decision_tree_plot.png     # Output: visualisasi decision tree
└── README.md
```

## 📦 Dataset

- **Sumber**: [Kaggle – Oranges vs Grapefruit](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)
- **Nama file**: `citrus.csv`
- **Jumlah data**: ± 10.000 baris
- **Fitur**:

| Kolom      | Tipe    | Keterangan                           |
|------------|---------|--------------------------------------|
| `name`     | String  | Label kelas: `orange` / `grapefruit` |
| `diameter` | Float   | Diameter buah (cm)  |
| `weight`   | Float   | Berat buah (gram)    |
| `red`      | Float   | Nilai warna merah (RGB) |
| `green`    | Float   | Nilai warna hijau (RGB  |
| `blue`     | Float   | Nilai warna biru (RGB)  |


## ⚙️ Instalasi & Cara Menjalankan
### 1. Clone Repository
```bash
git clone https://github.com/renit21c/Classification-Model.git
cd Classification-Model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
- Download `citrus.csv` dari [Kaggle](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)
- Letakkan file `citrus.csv` di folder yang sama dengan `classification.py`

### 4. Jalankan Script
```bash
python classification.py
```

## 🔄 Tahapan Pembuatan Model
 
### Step 1: Load Dataset
Dataset CSV dimuat menggunakan `pandas`. Dilakukan pengecekan shape, kolom, dan tipe data untuk memahami struktur awal data.
 
```python
df = pd.read_csv('citrus.csv')
```
 
---
 
### Step 2: Exploratory Data Analysis (EDA)
Analisis visual untuk memahami distribusi data dan hubungan antar fitur:
- **Histogram** distribusi setiap fitur berdasarkan kelas
- **Pie chart** proporsi kelas (orange vs grapefruit)
- **Heatmap korelasi** antar fitur numerik
  
Output: `eda_plots.png`, `correlation_heatmap.png`
 
---

 ### Step 3: Preprocessing Data
 
| Langkah              | Detail                                              |
|----------------------|-----------------------------------------------------|
| **Label Encoding**   | Kolom `name` diubah jadi angka: `orange=1`, `grapefruit=0` |
| **Feature Selection**| Fitur X: `diameter, weight, red, green, blue`  |
| **Train-Test Split** | 80% training, 20% testing, stratified, random_state=42 |
| **Feature Scaling**  | StandardScaler — wajib untuk SVM                   |
 
```python
le = LabelEncoder()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
 
---
 
### Step 4: Training Model
 
#### 1. Decision Tree
**Cara kerja**: Membangun pohon keputusan dengan membagi data berdasarkan fitur yang memberikan *information gain* tertinggi (Gini Index).
 
```python
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42
)
dt_model.fit(X_train, y_train)
```
 
**Parameter penting**:
- `max_depth=5`  membatasi kedalaman pohon agar tidak overfit
- `criterion='gini'`  menggunakan Gini Impurity untuk pemilihan fitur
---

#### 2. Naive Bayes (Gaussian)
**Cara kerja**: Menghitung probabilitas kelas menggunakan Teorema Bayes dengan asumsi setiap fitur berdistribusi Gaussian (normal) dan independen satu sama lain.
 
```python
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
```
 
**Kelebihan**: Sangat cepat, tidak perlu scaling  
**Kelemahan**: Asumsi independensi fitur tidak selalu berlaku
 
---
 
#### 3. Support Vector Machine (SVM)
**Cara kerja**: Mencari *hyperplane* optimal yang memisahkan kedua kelas dengan margin terbesar. Menggunakan kernel RBF untuk menangani data yang tidak linear.
 
```python
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)  # Wajib menggunakan data yang sudah di-scale
```
 
**Parameter penting**:
- `kernel='rbf'` Radial Basis Function kernel untuk pola non-linear
- `C=1.0` regularization parameter
- `gamma='scale'` otomatis menentukan gamma berdasarkan jumlah fitur
---

### Step 5: Prediksi
Model melakukan prediksi pada data test (`X_test`). SVM menggunakan `X_test_scaled`.
 
```python
y_pred_dt  = dt_model.predict(X_test)
y_pred_nb  = nb_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test_scaled)
```
 
---
 
### Step 6: Evaluasi Model
 
Metrik yang digunakan untuk mengukur performa setiap model:
 
| Metrik        | Keterangan                                           |
|---------------|------------------------------------------------------|
| **Accuracy**  | Proporsi prediksi yang benar dari seluruh data |
| **Precision** | Dari yang diprediksi positif, berapa yang benar  |
| **Recall**    | Dari yang sebenarnya positif, berapa yang terdeteksi |
| **F1-Score**  | Rata-rata harmonik antara Precision dan Recall  |
 
Output: `confusion_matrices.png`
 
---
 
### Step 7: Cross Validation
Cross-validation dilakukan untuk menguji apakah performa model konsisten, tidak hanya baik pada satu split data saja.
 
```python
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
```
 
Output: Mean accuracy dan standard deviation untuk setiap model.
 
---
 
### Step 8: Visualisasi Hasil
- **Confusion Matrices**: menampilkan True Positive, True Negative, False Positive, False Negative
- **Bar Chart**: perbandingan Accuracy, Precision, Recall, F1-Score ketiga model
- **Box Plot**: distribusi skor 5-Fold CV
- **Decision Tree Plot**: visualisasi struktur pohon (max_depth=3 ditampilkan)
Output: `model_comparison.png`, `decision_tree_plot.png`
 
---
### Step 9: Kesimpulan
 
Berdasarkan eksperimen dengan dataset Orange vs Grapefruit:
 
| Aspek              | Decision Tree | Naive Bayes | SVM        |
|--------------------|---------------|-------------|------------|
| Akurasi            | Sangat Tinggi | Tinggi      | Tertinggi  |
| Kecepatan Training | Cepat         | Sangat Cepat| Sedang     |
| Membutuhkan Scaling| Tidak         | Tidak       | **Ya**     |
| Interpretabilitas  | Tinggi ✅     | Sedang      | Rendah     |
| Kompleksitas       | Rendah        | Sangat Rendah| Sedang    |
 
**Model Terbaik: SVM** — menghasilkan akurasi tertinggi karena mampu menemukan batas keputusan yang optimal di ruang fitur yang lebih tinggi melalui kernel RBF.
 
---
 
## 📊 Output yang Dihasilkan
 
| File                      | Isi                                       |
|---------------------------|-------------------------------------------|
| `eda_plots.png`           | Histogram distribusi fitur + pie chart  |
| `correlation_heatmap.png` | Korelasi antar fitur |
| `confusion_matrices.png`  | Confusion matrix ketiga model |
| `model_comparison.png`    | Bar chart metrik + boxplot CV  |
| `decision_tree_plot.png`  | Visualisasi decision Tree       |
 
---
 
## 🛠️ Library yang Digunakan
 
| Library        | Fungsi                              |
|----------------|-------------------------------------|
| `pandas`       | Manipulasi dan analisis data  |
| `numpy`        | Operasi numerik  |
| `matplotlib`   | Visualisasi data   |
| `seaborn`      | Visualisasi statistik   |
| `scikit-learn` | Model ML, preprocessing, evaluasi  |
 
---
