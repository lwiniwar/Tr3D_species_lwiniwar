import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

def main(cf_matrix):
    cf_matrix = cf_matrix.astype(float)
    cf_matrix[cf_matrix == 0] = np.nan
    plt.figure(figsize=(8,8), dpi=200)
    no_diag = cf_matrix.copy()
    no_diag[np.diag_indices_from(no_diag)] = np.nan
    ax1 = sns.heatmap(cf_matrix, annot=True, cmap='Reds', fmt='.0f', square=True, cbar=False)
    diag_only = np.diag(cf_matrix)
    diag_only = np.diag(diag_only)
    diag_only[diag_only == 0] = np.nan
    ax2 = sns.heatmap(diag_only, annot=True, cmap='Blues', fmt='.0f', square=True, cbar=False)
    for i in range(cf_matrix.shape[0]):
        ax2.add_patch(Rectangle((i,i),1,1, fill=False, edgecolor='green', lw=1))
    plt.xlabel('predicted class')
    plt.ylabel('reference class')
    plt.tight_layout()
    # plt.show()
    return plt.gcf()

if __name__ == '__main__':
    sns.set()
    inf = r"COST-Challenge\models\cm_latest_3SAs_larger_SPECIES_winvsqr_8kpts_nllLoss.txt"
    # read in file
    cf_matrix = np.loadtxt(inf)
    main(cf_matrix)