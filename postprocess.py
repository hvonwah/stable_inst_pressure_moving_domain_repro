import pandas as pd
from scipy import signal
import numpy as np

example = 1
method = 'TH'
nu = 0.01
k = 2
h = 0.1
dt = 0.14
gpv = 0.1
gpp = 0.1
gd = 0.0
stab = 'sufficient'
geo = 1
subdiv = 0
cip = 0.001

if example == 1:
    if method == 'TH':
        filename = f'results/Example1_{method}{k}hogeo{geo}subdiv{subdiv}hmax{h}BDF2'
        filename += f'dt{dt}nu{nu}gpv{gpv}gpp{gpp}gp{stab}graddiv{gd}.txt'
    elif method == 'EO':
        filename = f'results/Example1_{method}{k}hogeo{geo}subdiv{subdiv}hmax{h}BDF2'
        filename += f'dt{dt}nu{nu}gpv{gpv}gpp{gpp}gp{stab}graddiv{gd}cip{cip}.txt'
    elif method == 'SV':
        filename = f'results/Example1_{method}{k}hmax{h}BDF2dt{dt}nu{nu}gpv{gpv}gpp{gpp}.txt'
elif example == 2:
    if method == 'TH':
        filename = f'results/Example2_{method}{k}hmax{h}BDF2'
        filename += f'dt{dt}nu{nu}gpv{gpv}gpp{gpp}gp{stab}graddiv{gd}.txt'
    elif method == 'EO':
        filename = f'results/Example2_{method}{k}hmax{h}BDF2dt{dt}nu{nu}'
        filename += f'gpv{gpv}gpp{gpp}gp{stab}graddiv{gd}cip{cip}.txt'
    elif method == 'SV':
        filename = f'results/Example2_{method}{k}hmax{h}BDF2dt{dt}nu{nu}gpv{gpv}gpp{gpp}.txt'

data = pd.read_csv(filename, delimiter='\t', header=0)
sig = data['dragP2'].to_numpy()
sos = signal.butter(2, 2, 'hp', output='sos', fs=1 / dt + 1)
noise = signal.sosfiltfilt(sos, sig)

idx1, idx2 = min(data[data['time'] > 1].index), max(data[data['time'] < 9].index)
time = data['time'].to_numpy()[idx1: idx2]
sig = sig[idx1: idx2]
noise = noise[idx1: idx2]
print(f'{max(np.abs(noise)):.5f} {np.mean(np.abs(noise)):.5f}')

