# Informative Embedding Ensemble

## Constructing a Consensus Space for Reliable Clustering Evaluation in Deep Learning

Deep clustering methods learn lower-dimensional embeddings that make high-dimensional data easier to cluster. However, evaluating and comparing these models remains challenging when ground-truth labels are unavailable.

This project implements an informative embedding ensemble that combines candidate embeddings into a shared consensus space. This space supports the calculation of internal clustering metrics and comparison of their rankings against external benchmarks, including normalized mutual information (NMI), clustering accuracy (ACC), and the adjusted Rand index (ARI).

## Requirements

The scripts require Python and the following packages:

- NumPy
- SciPy
- scikit-learn 1.5
- h5py

The implementation relies on scikit-learn's internal t-SNE APIs. Use the version specified above to ensure compatibility.

## Real-data evaluation

1. Follow the [ACE instructions](https://github.com/ZWStatLab/ACE) to obtain the real-data model outputs and the corresponding `file_list` directory. The outputs are available in the [ACE Google Drive folder](https://drive.google.com/drive/folders/1HBmYO_BThSW9ysfCPKPRuwm4kg1bKzSM?usp=drive_link).
2. Place the `datasets` folder, all task output folders, and the `file_list` directory in the working directory. Store dataset files under `datasets/<dataset>/` and model outputs under `<task>/deep_clustering_outputs/`.
3. Run the real-data evaluation:

   ```bash
   python3 iee_real_data.py --task <task> --dataset <dataset>
   ```

Supported tasks are `JULE_hyper`, `JULE_num`, `DEPICT_hyper`, and `DEPICT_num`. Specify the dataset name exactly as it appears in the ACE dataset directory (for example, `FRGC` or `COIL-20`).

## Simulation evaluation

1. Follow the [ACE instructions](https://github.com/ZWStatLab/ACE) to generate the simulation data and make it accessible from the working directory. If the data is stored elsewhere, update the simulation data path in `iee_simulation.py`.
2. Run the simulation evaluation:

   ```bash
   python3 iee_simulation.py --task <task> --dataset <replication_id>
   ```

Supported simulation tasks are `sim_sparse_1.0`, `sim_sparse_2.0`, `sim_dense_1.0`, and `sim_dense_2.0`, which represent different simulation settings. The `--dataset` argument specifies the replication ID (for example, `1`).

Both evaluation scripts also accept `--n_components` and `--perplexity` to configure the consensus embedding. The defaults are 2 and 30, respectively.

## Results

Each script saves a pickle file containing the consensus embedding and associated scores. It also prints Kendall's tau and Spearman's rank correlation between the internal clustering metrics and external benchmarks.