import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import pandas as pd
import os

def checkdata(opt):
    # fpath = "data/moses/prop_temp.csv"
    results = pd.read_csv(opt.load_data)

    figure, axs = plt.subplots(nrows=1, ncols=opt.cond_dim, figsize=(10,4), constrained_layout=True)
    
    units = ["Kelvin", "Pa•s", "kg/m3", "S/m"]

    ret = []
    for i, (label, ax) in enumerate(zip(opt.cond_labels, axs)):
        sns.violinplot(y=label, data=results, ax=ax, color=sns.color_palette()[i])
        ax.set(xlabel=label, ylabel=units[i])
        quantiles = get_quantiles(results[label])
        # for quantile in quantiles: # every quantile
        #     text = ax.text(0, quantile, f'{quantile:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        #     text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
        # record max/min of the values
        ret.append((max(quantiles[-1], results[label].min()), min(quantiles[0], results[label].max())))

    plt.show()
    os.makedirs("figures", exist_ok=True)
    figure.savefig("figures/Property_Boundaries.png", dpi=300)#, bbox_inches='tight')
    return ret


def get_quantiles(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    UAV = Q3 + 1.5 * IQR
    LAV = Q1 - 1.5 * IQR
    return [UAV, Q3, Q1, LAV]
