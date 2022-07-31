## TfIdf for getting topic clusters

Please use the TfIdf.ipynb notebook for step-by-step method for extracting topic clusters.

## Visualization of clusters with k=5

![Cluster Visualization with KMeans](tfidf-cluster-visualization.png?raw=true "Cluster Visualization with KMeans (k=5)")

## Installation

```bash
pip install -r ../../requirements.txt
```

## Command to extract topic clusters

```bash
python3 tfidf.py --file_path ../../Dataset/dataset.csv --outputfile_path <file where the clusters should be written>
```