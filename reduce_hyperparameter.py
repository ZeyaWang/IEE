import os

# Each setting writes the selected model files to its own output root,
# named 'file_list_<JULE or DEPICT>_<number of hyperparameter combinations>'.

# To run IEE on the real data using the reduced hyperparameter search space, 
# copy all files from the corresponding output root directory into the original
# `file_list` directory, overwriting the existing model list files, and then run `iee_real_data.py`.

depict_lrs = [0.0001, 0.001, 0.01]
depict_settings = {
    'file_list_DEPICT_9': [0.1, 1.0, 10.0],
    'file_list_DEPICT_6': [0.1, 10.0],
}

jule_lrs = [0.001, 0.01, 0.1]
jule_settings = {
    'file_list_JULE_6':  [0.2, 0.9],
    'file_list_JULE_9':  [0.2, 0.5, 0.9],
    'file_list_JULE_15': [0.2, 0.3, 0.5, 0.8, 0.9],
}

true_num = {
    'USPS':10, 'UMist':20, 'COIL-20':20, 'COIL-100':100, 'YTF':41, 'FRGC':20, 'MNIST-test':10, 'CMU-PIE':68
}
depict_datasets = ['USPS', 'MNIST-test', 'FRGC', 'YTF', 'CMU-PIE']
jule_datasets = ['COIL-100', 'COIL-20', 'MNIST-test', 'USPS', 'CMU-PIE', 'FRGC', 'UMist', 'YTF']


root = 'file_list'
def load_lines(tk, ds):
    with open(os.path.join(root, tk, f'{ds}.txt'), "r") as f:
        return [line.strip() for line in f]

def save_lines(root2, tk, ds, lines):
    os.makedirs(os.path.join(root2, tk), exist_ok=True)
    with open(os.path.join(root2, tk, f'{ds}.txt'), "w") as f:
        for line in lines:
            f.write(line + "\n")

def joins(m1, m2):
    return [m for m in m1 if m in m2]

def select(root2, task, dataset, ms2):
    ms = load_lines(task, dataset)
    m = joins(ms, ms2)
    print(root2, task, dataset, len(m), len(ms))
    save_lines(root2, task, dataset, m)


task = 'DEPICT_hyper'
for root2, depict_rhs in depict_settings.items():
    for dataset in depict_datasets:
        tk = true_num[dataset]
        ms2 = ["output{}_{}_{}_{}_1.0.npz".format(dataset, tk, lr, rh)
               for lr in depict_lrs for rh in depict_rhs]
        select(root2, task, dataset, ms2)

task = 'JULE_hyper'
for root2, etas in jule_settings.items():
    for dataset in jule_datasets:
        ms2 = ["{}{}{}".format(dataset, lr, eta) for eta in etas for lr in jule_lrs]
        select(root2, task, dataset, ms2)
