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

