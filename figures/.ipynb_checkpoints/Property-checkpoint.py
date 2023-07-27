import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Set constants
fract_val = 0.15
figurewidth, figureheight = 12, 10
font = {'family': 'sans-serif', 'weight': 'light', 'size': 14}
tickfont = {'family': 'sans-serif', 'weight': 'bold', 'size': 12}
plot_box_thickness = 2
figuredpi = 300

df = pd.read_csv("../data/IL/model_data.csv")
print(df.head())
print(df.info())

# +
# ## Find the outliers for each column
columns = ['Temperature', 'DynViscosity', 'Density', 'ElecConductivity']

# mask = pd.Series(True for _ in range(len(df)))
# for col in columns:
#     series = df[col]
    
#     mu = series.mean()
#     std = series.std()
    
#     # outlier if outside of 2 std deviations
#     max_stds = 3
#     within = (mu - std * max_stds < series) & (series < mu + std * max_stds)
# #     print(sum(1 if w else 0 for w in within))
# #     print(series.std())
    
#     mask = mask & within
    
# # Custom filtering
# mask = mask & (df['DynViscosity'] < 1)
# # mask = mask & (df['Electrical conductivity'] < 3)
# -

filtered = df#[mask]
print(filtered.head())
print(filtered.info())

fig, axs = plt.subplots(nrows=2, ncols=len(columns)//2, figsize=(figurewidth, figureheight))
labels = ['Temperature (K)', 'Dynamic Viscosity (Pa•s)', 'Density (kg/m$^3$)', 'Electrical Conductivity (S/m)']
ticks = [[263, 290, 318, 345, 373],
         [0.25 * i for i in range(5)],
         [875, 110, 1300, 1500, 1735],
         [0.9 * i for i in range(5)]]
for i, (col, label, ax) in enumerate(zip(columns, labels, axs.ravel())):
#     ax.violinplot(filtered[col])
    sns.violinplot(x=col, data=filtered, ax=ax, color=sns.color_palette("colorblind")[i], cut=0, inner="quartile")
    ax.set_title(" ".join(label.split()[:-1]), fontdict=font)
    ax.set_xlabel(label, fontdict=font)
    ax.set_ylabel("Frequency", fontdict=font)
    ax.tick_params(axis='both', which='major', labelsize=tickfont['size'], direction="in")
    r = filtered[col].max()-filtered[col].min()
    ax.set_xticks([filtered[col].min()+i*r/4 for i in range(5)])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(plot_box_thickness)
        spine.set_color('black')
plt.subplots_adjust(wspace=0.15, hspace=0.35)        

fig, axs = plt.subplots(nrows=2, ncols=len(columns)//2, figsize=(figurewidth, figureheight))
labels = ['Temperature (K)', 'Dynamic Viscosity (Pa•s)', 'Density (kg/m$^3$)', 'Electrical Conductivity (S/m)']
ticks = [[263 + 27.5*i//1 for i in range(5)],
         [0.25 * i for i in range(5)],
         [875 + i*215 for i in range(5)],
         [0.9 * i for i in range(5)]]
for i, (col, label, ax) in enumerate(zip(columns, labels, axs.ravel())):
#     ax.violinplot(filtered[col])
    sns.violinplot(x=col, data=filtered, ax=ax, color=sns.color_palette("colorblind")[i], cut=0)
    ax.set_xlabel(label, fontdict=font)
    ax.set_ylabel("Frequency", fontdict=font)
    ax.tick_params(axis='both', which='major', labelsize=tickfont['size'], direction="in")
    r = filtered[col].max()-filtered[col].min()
    ax.set_xticks(ticks[i])
    for spine in ax.spines.values():
        spine.set_linewidth(plot_box_thickness)
        spine.set_color('black')
plt.subplots_adjust(wspace=0.15, hspace=0.20)        

fig, axs = plt.subplots(nrows=2, ncols=len(columns)*2, figsize=(figurewidth, figureheight))
labels = ['Temperature (K)', 'Dynamic Viscosity (Pa•s)', 'Density (kg/m$^3$)', 'Electrical Conductivity (S/m)']
ticks = [[263 + 27.5*i//1 for i in range(5)],
         [0.25 * i for i in range(5)],
         [875 + i*215 for i in range(5)],
         [0.9 * i for i in range(5)]]
for i, (col, label) in enumerate(zip(columns, labels)):
#     ax.violinplot(filtered[col])
    sns.violinplot(x=col, data=filtered, ax=ax, color=sns.color_palette("colorblind")[i], cut=0)
    ax.set_xlabel(label, fontdict=font)
    ax.set_ylabel("Frequency", fontdict=font)
    ax.tick_params(axis='both', which='major', labelsize=tickfont['size'], direction="in")
    r = filtered[col].max()-filtered[col].min()
    ax.set_xticks(ticks[i])
    for spine in ax.spines.values():
        spine.set_linewidth(plot_box_thickness)
        spine.set_color('black')
plt.subplots_adjust(wspace=0.15, hspace=0.20)        

(1735-875)/4

# filtered["Electrical conductivity"].min()
filtered["DynViscosity"].min()

filtered = filtered.drop(columns="Pressure/kPa")
filtered.column
