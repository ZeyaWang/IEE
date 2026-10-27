# Informative Embedding Ensemble

## Towards a Consensus Space for Reliable Clustering Evaluation in Deep Learning

Deep clustering methods learn lower-dimensional embeddings that make clustering high-dimensional data more manageable. However, evaluating and comparing these models remains difficult when ground-truth labels are unavailable.

This project implements an informative embedding ensemble that combines candidate embeddings into a shared consensus space. The resulting space can be used to calculate internal clustering metrics and compare their rankings with external benchmarks, including normalized mutual information (NMI) and clustering accuracy.

## Requirements

The scripts require Python and the following packages:

- NumPy
- SciPy
- scikit-learn 1.5
- h5py

The implementation relies on scikit-learn's t-SNE internals, so use the version specified above for compatibility.

## Real-data evaluation

1. Follow the [ACE instructions](https://github.com/ZWStatLab/ACE) to obtain the real-data model outputs and corresponding `file_list` directory. The outputs are available through the [ACE Google Drive folder](https://drive.google.com/drive/folders/1HBmYO_BThSW9ysfCPKPRuwm4kg1bKzSM?usp=drive_link).
2. Arrange the files using the ACE application layout, with the dataset files, task output folders, and `file_list` directory available from the working directory. Dataset files should be under `datasets/<dataset>/`, and model outputs under `<task>/deep_clustering_outputs/`.
3. Run the real-data evaluation:

   ```bash
   python3 towards_real_data.py --task <task> --dataset <dataset>
   ```

Supported tasks are `JULE_hyper`, `JULE_num`, `DEPICT_hyper`, and `DEPICT_num`. Use the dataset name exactly as it appears in the ACE dataset directory (for example, `FRGC` or `COIL-20`).

## Simulation evaluation

1. Follow the [ACE instructions](https://github.com/ZWStatLab/ACE) to generate or download the simulation data, and make it accessible from the working directory. Update the simulation data path in `towards_simulation.py` if your data is stored elsewhere.
2. Run the simulation evaluation:

   ```bash
   python3 towards_simulation.py --task <task> --dataset <replication_id>
   ```

Supported simulation tasks are `sim_sparse_1.0`, `sim_sparse_2.0`, `sim_dense_1.0`, and `sim_dense_2.0`. The dataset argument is the simulation replication ID (for example, `1`).

Both evaluation scripts also accept `--n_components` and `--perplexity` to configure the consensus embedding. Their defaults are 2 and 30, respectively. The simulation script additionally accepts `--setting`.

## Results

Each script saves a pickle file containing the consensus embedding and related scores. It also prints Kendall's tau and Spearman's rank correlation between internal clustering metrics and the external NMI and accuracy benchmarks.