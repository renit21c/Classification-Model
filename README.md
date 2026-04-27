# Orange vs Grapefruit Classification
Proyek klasifikasi buah orange dan grapefruit menggunakan tiga algoritma machine learning: Decision Tree, Naive Bayes, dan Support Vector Machine (SVM)

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
| `diameter` | Float   | Diameter buah (cm)                   |
| `weight`   | Float   | Berat buah (gram)                    |
| `red`      | Float   | Nilai warna merah (RGB)              |
| `green`    | Float   | Nilai warna hijau (RGB)              |
| `blue`     | Float   | Nilai warna biru (RGB)               |


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

 ### Step 3 — Preprocessing Data
 
| Langkah              | Detail                                              |
|----------------------|-----------------------------------------------------|
| **Label Encoding**   | Kolom `name` diubah jadi angka: `orange=1`, `grapefruit=0` |
| **Feature Selection**| Fitur X: `diameter, weight, red, green, blue`       |
| **Train-Test Split** | 80% training, 20% testing, stratified, random_state=42 |
| **Feature Scaling**  | StandardScaler — wajib untuk SVM                   |
 
```python
le = LabelEncoder()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
 
---
 
### Step 4 — Training Model
 
#### 🌳 1. Decision Tree
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
- `max_depth=5` → membatasi kedalaman pohon agar tidak overfit
- `criterion='gini'` → menggunakan Gini Impurity untuk pemilihan fitur
---
