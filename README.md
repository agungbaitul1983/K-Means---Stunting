# 1. Business Understanding
Mengkelompokan Jumlah Balita Stunting Menurut Puskesmas Tahun 2020 berdasarkan nilai jumlah dan persentase (%)

Jumlah Balita Stunting Menurut Puskesmas Tahun 2020 dibagi menjadi tiga kelompok, yaitu:

* kelompok puskesmas memiliki data balita stunting terkecil, 
* kelompok puskesmas memiliki data balita stunting terbanyak.

Puskesmas yang <b>wajib</b> mengadakan <b>sosialisasi balita stunting</b> adalah puskesmas yang masuk dalam <b>kelompok puskesmas memiliki data balita stunting terbanyak perlu gencar sosialisasi</b>

# 2. Data Understanding
Data saya ambil dari https://data.tasikmalayakota.go.id/dinas-kesehatan/jumlah-balita-stunting-menurut-puskesmas-tahun-2020/ dan diberi nama <b>stunting.csv</b>, data tersebut berisi atribut sebagai berikut:

* Puskesmas, adalah nama puskesmas, sebagai identity
* Jumlah , adalah jumlah balita stunting (dalam satuan tahun)
* Persentase, adalah nilai persentase balita stunting (dalam satuan persen)

# 3. Data Preparation

# import library-libary yang digunakan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# load dataset stunting.csv
nilai = pd.read_csv('datasets/stunting.csv')

# menampilkan lima data teratas
nilai.head(5)

# cek dimensi data
nilai.shape

# menampilkan nama-nama atribut
nilai.columns

# menampilkan tipe data masing-masing atribut
nilai.info()

### Cek Missing Value (hilangnya beberapa data)

# cek apakah terdapat missing value
nilai.isnull().sum()

Data stunting.csv tidak terdapat missing value

<b>Atribute/Feature Selection</b><br>
Pada bagian data understanding, telah dijelaskan bahwa puskesmas sebagai identity, sehingga tidak digunakan dalam pemodelan.

#filter hanya atribut JUMLAH dan PERSENTASE yang digunakan dalam pemodelan
nilai_fs = nilai[['JUMLAH', 'PERSENTASE']]

# statistic descriptive (Membuat Ringkasan Data)
nilai_fs.describe()

# cek lima data teratas setelah dilakukan feature selection
nilai_fs.head()

# cek dimensi data setelah dilakukan feature selection
nilai_fs.shape

# menampilkan distribusi kedua atribut
fig = plt.figure(figsize=(10,5))

# Fungsi Sub plot mengambil tiga argumen yang menjelaskan layout gambar.
# Layout diatur dalam baris dan kolom, yang diwakili oleh argumen pertama dan kedua.


fig.add_subplot(2,1,1)
# gambar tersebut memiliki 2 baris, 1 kolom, dan plot ini adalah plot pertama.
sns.distplot(nilai['JUMLAH'])

fig.add_subplot(2,1,2)
sns.distplot(nilai['PERSENTASE'])

fig.tight_layout()

# melihat distribusi kedua atribut dalam bentuk box plot
fig = plt.figure(figsize=(10,5))

fig.add_subplot(1,2,1)
plt.boxplot(nilai['JUMLAH'])

fig.add_subplot(1,2,2)
plt.boxplot(nilai['PERSENTASE'])

plt.show()


# hubungan korelasi antara JUMLAH dan PERSENTASE
nilai_fs.corr()

# Bivariate analysis antara atribut JUMLAH dan PERSENTASE dengan scatter plot
plt.scatter(nilai_fs['JUMLAH'], nilai_fs['PERSENTASE'])
plt.xlabel('JUMLAH')
plt.ylabel('PERSENTASE')
plt.title('Scatter Plot Jumlah Vs. Persentase')
plt.show()

Scatter plot menunjukkan korelasi positif <br>
Semakin jumlah naik, nilai persentase semakin tinggi

# 4. Modeling
Dalam tahapan modeling, algoritme machine learning yang digunakan adalah algoritme K-Means

Nilai K yang digunakan adalah tiga, karena performa puskesmas akan dibagi menjadi tiga kelompok

kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(nilai_fs)
kmeans.cluster_centers_

## 5. Evaluation

x = nilai['JUMLAH']
y = nilai['PERSENTASE']

group = kmeans.labels_

colors = ['red', 'green']
fig, ax = plt.subplots()

for g in set(kmeans.labels_):
    xi = [x[j] for j in range(len(x)) if group[j]==g]
    yi = [y[j] for j in range(len(y)) if group[j]==g]
    ax.scatter(xi, yi, c=colors[g], label=g)
    ax.scatter(223.46666667, 14.39 , c='pink')
    ax.scatter(625.57142857,  25.63571429, c='blue')
    plt.xlabel('JUMLAH')
    plt.ylabel('PERSENTASE')
    plt.title('JUMLAH Vs. PERSENTASE')

ax.legend()
plt.show()

* puskesmas data balita stuntting terbanyak masuk ke dalam Cluster 0 (warna merah)

* puskesmas data balita stuntting terkecil masuk ke dalam Cluster 1 (warna hijau)

* Dalam tahapan evaluasi, akan dihitung berapa jumlah kelompok (nilai K) optimal dari algoritme K-Means saat diterapkan pada dataset stunting.csv

# mancari nilai K optimal
inertia = []
silhouette = []

K = range(2,10)
for k in K:
    kmeans = KMeans(n_clusters = k)
    kmeans = kmeans.fit(nilai_fs)
    kmeans.labels_
    
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(nilai_fs, kmeans.labels_))

# visualisasi plotting nilai K
fig = plt.figure(figsize=(10,4))

fig.add_subplot(1,2,1)
plt.plot(K, inertia, marker='x')
plt.xlabel('k')
plt.ylabel('Nilai Inertia')

fig.add_subplot(1,2,2)           
plt.plot(K, silhouette, marker='o')
plt.xlabel('k')
plt.ylabel('Nilai Silhoutte')

Dari grafik di atas terlihat bahwa nilai K optimal adalah dua, sehingga sudah tepat membagi kelompok menjadi dua kelompok

# 6. Deployment
Menampilkan data semua puskesmas dengan menambahkan atribut puskesmas dan Cluster

nilai_fs['PUSKESMAS'] = nilai['PUSKESMAS']
nilai_fs['JUMLAH'] = nilai_fs['JUMLAH']
nilai_fs['PERSENTASE'] = nilai_fs['PERSENTASE']
nilai_fs['Cluster']=kmeans.labels_
nilai_fs

Menampilkan kelompok puskesmas memiliki data balita stunting terbanyak (berada di Cluster 0)

puskesmas_data_balita_stuntting_terbanyak = nilai_fs[nilai_fs.Cluster==0]
puskesmas_data_balita_stuntting_terbanyak

# jumlah kelompok puskesmas memiliki data balita stunting terbanyak dihitung berdasarkan count puskesmas
puskesmas_data_balita_stuntting_terbanyak['PUSKESMAS'].count()

puskesmas_data_balita_stuntting_terkecil = nilai_fs[nilai_fs.Cluster >=1]
puskesmas_data_balita_stuntting_terkecil

# SELESAI
