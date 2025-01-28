import os
import re
import numpy as np
import matplotlib.pyplot as plt

def get_mean_std(array):
    """Compute mean and standard deviation along the first axis."""
    return np.mean(array, axis=0), np.std(array, axis=0)

# Directory containing the results
results_dir = "./results_new"

# Regex to match filenames and extract parameters
# pattern = r"(?P<method>.*?)_regret(?:_d_(?P<d>\d+))?(?:_K_(?P<K>\d+))?(?:_T_(?P<T>\d+))?(?:_cs_(?P<cs>[\d.]+))?(?:_as_(?P<as>[\d.]+))?(?:_b_(?P<b>\d+))?"
pattern = r"(?P<method>.*?)_regret(?:_d_(?P<d>\d+))?(?:_K_(?P<K>\d+))?(?:_T_(?P<T>\d+))?(?:_cs_(?P<cs>[\d.]+))?(?:_as_(?P<as>[\d.]+))?(?:_b_(?P<b>\d+))_(?P<data_method>.*?)?"
# Loop through all files in the directory
for filename in os.listdir(results_dir):
    if filename.endswith(".npy"):  # Only process .npy files
        match = re.match(pattern, filename)
        if match:
            params = match.groupdict()
            method = params['method']
            d = int(params['d']) if params['d'] else None
            K = int(params['K']) if params['K'] else None
            T = int(params['T']) if params['T'] else None
            cs = float(params['cs']) if params['cs'] else None
            as_ = float(params['as']) if params['as'] else None
            b = int(params['b']) if params['b'] else None
            
            # Load data
            file_path = os.path.join(results_dir, filename)
            ucb = np.load(file_path)
            if T is not None:
                ucb = ucb[:, :T]  # Slice up to T if T is available
            ucb_mean, ucb_std = get_mean_std(ucb)
            
            # Plotting
            x = np.arange(1, ucb_mean.shape[0] + 1)
            plt.plot(x, ucb_mean, linewidth=2.0, linestyle=':', label=f"{method}: d={d},  cs={cs}, as={as_}, b={b}")
            plt.fill_between(x, ucb_mean - ucb_std, ucb_mean + ucb_std, alpha=0.2)
            plt.xlabel('Episodes')
            plt.ylabel('Regret')
            plt.legend()
            plt.grid(True)
        
plt.savefig("figures/regret.pdf")
