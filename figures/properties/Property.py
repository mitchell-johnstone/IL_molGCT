import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import ExactMolWt

## Set constants
fract_val = 0.15
figurewidth, figureheight = 16, 8
font = {'family': 'sans-serif', 'weight': 'light', 'size': 14}
tickfont = {'family': 'sans-serif', 'weight': 'light', 'size': 12}
plot_box_thickness = 2
figuredpi = 300

df = pd.read_csv("../../data/IL/model_data.csv")
print(df.head())
print(df.info())


## Add the molecular weight
def wt(smiles):
    try:
        return ExactMolWt(MolFromSmiles(smiles))
    except:
        return -1
df["MW"] = df["src"].apply(wt)
df["MW"][df["MW"] == -1] = df["MW"][df["MW"]!=-1].mean()
df["MW"].min(), df["MW"].max()

# +
## Find the outliers for each column
columns = ['Temperature', 'MW', 'DynViscosity', 'Density', 'ElecConductivity']


# mask = pd.Series(True for _ in range(len(df)))
# for col in columns:
#     series = df[col]
#     
#     mu = series.mean()
#     std = series.std()
#     
#     # outlier if outside of 2 std deviations
#     max_stds = 3
#     within = (mu - std * max_stds < series) & (series < mu + std * max_stds)
# #     print(sum(1 if w else 0 for w in within))
# #     print(series.std())
#     if col == 'MW':
#         continue
#     mask = mask & within
#     
# # Custom filtering
# mask = mask & (df['DynViscosity'] < 1)
# # mask = mask & (df['Electrical conductivity'] < 3)
# # -
# 
# filtered = df[mask]
# print(filtered.head())
# print(filtered.info())
filtered = df

# +
fig = plt.figure(figsize=(figurewidth, figureheight))
axs = [None]*5
axs[0] = plt.subplot2grid(shape=(2,6), loc=(0,1), colspan=2)
axs[1] = plt.subplot2grid(shape=(2,6), loc=(0,3), colspan=2)
axs[2] = plt.subplot2grid(shape=(2,6), loc=(1,0), colspan=2)
axs[3] = plt.subplot2grid(shape=(2,6), loc=(1,2), colspan=2)
axs[4] = plt.subplot2grid(shape=(2,6), loc=(1,4), colspan=2)
axs = np.array(axs)

labels = ['Temperature (K)', 'Molecular Weight (g/mol)', 'Dynamic Viscosity (Pa·s)', 'Density (kg/m$^3$)', 'Electrical Conductivity (S/m)']
ticks = [[263 + 27.5*i//1 for i in range(5)],
         [105 + 220 * i for i in range(5)],
         [0.95 * i for i in range(5)],
         [875 + i*215 for i in range(5)],
         [2.1 * i for i in range(5)]]
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
plt.subplots_adjust(wspace=0.35, hspace=0.20)
plt.savefig("Property Distributions.png", dpi=figuredpi, bbox_inches='tight')
# -

# filtered["Electrical conductivity"].min()
filtered["DynViscosity"].min()

# filtered = filtered.drop(columns="Pressure/kPa")
filtered.columns

# +
# filtered.to_csv("full.csv", index=False)
# filtered.iloc[range(len(filtered)*4//5)].to_csv("train.csv", index=False)
# filtered.iloc[range(len(filtered)*4//5, len(filtered))].to_csv("test.csv", index=False)
# -


