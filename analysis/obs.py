# %%
import glob
import os, sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from network.network import MultiCellNetwork as Net
from network.base_station import BaseStation as BS

mean_stats, std_stats = [pd.DataFrame(BS.annotate_obs(pd.read_csv(f).values.T)) 
                         for f in glob.glob('*(*, 1*)_*.csv')]

# %%
mean_stats

# %%
mean_stats.describe().T.sort_values(by='mean', ascending=False)

# %%
mean_stats.mean(axis=0).plot(legend=False)

# %%
mean_stats.std(axis=0).plot(legend=False)  # std among agents

# %%
std_stats.mean(axis=0).plot(legend=False)  # std over time

# %%
mean_stats[mean_stats > 20].dropna(axis=1, how='all')

# %%
std_stats[std_stats > 20].dropna(axis=1, how='all')

# %%
const_stats = mean_stats[std_stats < 1e-8].dropna(axis=1, how='any')
const_stats

# ======================================================

# %%
mean_stats, std_stats = [pd.DataFrame(Net.annotate_obs(pd.read_csv(f).values.T)) 
                         for f in glob.glob('*(*, 8*)_*.csv')]


mean_stats = mean_stats.iloc[0,:Net.global_obs_dim]
std_stats = std_stats.iloc[0,:Net.global_obs_dim]

# %%
mean_stats.plot()

# %%
std_stats.plot()

# %%
mean_stats[mean_stats > 20].dropna(how='all')
