import matplotlib.pyplot as plt
import numpy as np

dataset = 'SEER' ##'FLC', 'SUPPORT', 'SEER'

eps = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
C_I_std=[]
if dataset == 'FLC':

    C_I = [0.7958, 0.8029, 0.8048, 0.8052, 0.8041, 0.8034]
    C_I_std = [0.0049, 0.0021, 0.0010, 0.0003, 0.0002, 0.0003]

    F_I = [0.0015, 0.0075, 0.0233, 0.0946, 0.3943, 1.3786]
    F_I_std = [0.0004, 0.0007, 0.0016, 0.0029, 0.0085, 0.0205]

    F_G = [0.0086, 0.0201, 0.0469, 0.1664, 0.6692, 2.2501]
    F_G_std = [0.0007, 0.0013, 0.0029, 0.0049, 0.0141, 0.0322]

    F_S = [0.0227, 0.0472, 0.1025, 0.3384, 1.1538, 2.4707]
    F_S_std = [0.0030, 0.0040, 0.0082, 0.0153, 0.0271, 0.0269]

    F_A = [0.0110, 0.0249, 0.0576, 0.1998, 0.7391, 2.0331]
    F_A_std = [0.0013, 0.0018, 0.0040, 0.0075, 0.0164, 0.0247]
elif dataset == 'SUPPORT':
    C_I = [0.5453, 0.5438, 0.5397, 0.5384, 0.5472, 0.5735]
    C_I_std = [0.0118, 0.0080, 0.0065, 0.0035, 0.0027, 0.0019]

    F_I = [0.0005, 0.0009, 0.0022, 0.0100, 0.0215, 0.0378]
    F_I_std = [0.0009, 0.0002, 0.0003, 0.0006, 0.0009, 0.0013]

    F_G = [0.0025, 0.0026, 0.0032, 0.0029, 0.0062, 0.0214]
    F_G_std = [0.0008, 0.0005, 0.0009, 0.0012, 0.0023, 0.0030]

    F_S = [0.0081, 0.0067, 0.0081, 0.0092, 0.0243, 0.0729]
    F_S_std = [0.0018, 0.0010, 0.0012, 0.0033, 0.0074, 0.0094]

    F_A = [0.0037, 0.0034, 0.0045, 0.0074, 0.0173, 0.0440]
    F_A_std = [0.0010, 0.0004, 0.0007, 0.0015, 0.0034, 0.0044]
else:
    C_I = [0.6381, 0.7002, 0.7190, 0.7359, 0.7427, 0.7427]
    C_I_std = [0.0185, 0.0137, 0.0064, 0.0042, 0.0032, 0.0032]

    F_I = [0.0042, 0.0094, 0.0190, 0.1185, 0.4982, 0.5053]
    F_I_std = [0.0016, 0.0021, 0.0030, 0.0086, 0.0225, 0.0230]

    F_G = [0.0107, 0.0175, 0.0237, 0.1414, 0.7120, 0.7241]
    F_G_std = [0.0050, 0.0048, 0.0061, 0.0217, 0.0970, 0.0989]

    F_S = [0.0820, 0.0810, 0.0804, 0.1609, 0.5647, 0.5713]
    F_S_std = [0.0070, 0.0074, 0.0082, 0.0287, 0.0630, 0.0631]

    F_A = [0.0323, 0.0360, 0.0410, 0.1402, 0.5916, 0.6002]
    F_A_std = [0.0039, 0.0042, 0.0048, 0.0188, 0.0557, 0.0563]


# F_A = (1/3)*(np.asarray(F_I)+np.asarray(F_G)+np.asarray(F_S))


fig, ax = plt.subplots()

l1 = ax.errorbar(eps,C_I, yerr=C_I_std, color='red', label=r'c-index', linewidth=2, fmt='-o')
# ax.errorbar(np.asarray(eps),np.asarray(C_I))
ax.set_xlabel(r"$\alpha$", fontsize = 15, fontweight='bold')
ax.set_ylabel(r'c-index$\uparrow$',fontsize=15, fontweight='bold')
# if dataset == 'FLC':
#     ax.set_ylim((0.78, 0.81))

ax2=ax.twinx()
l2 =ax2.errorbar(eps,F_I, yerr=F_I_std, color='navy', label=r'F$_I$',linewidth=2, fmt='-o')
l3 =ax2.errorbar(eps,F_G, yerr=F_G_std, color='blue', label=r'F$_G$',linewidth=2, fmt='-o')
l4 =ax2.errorbar(eps,F_S, yerr=F_S_std, color='royalblue', label=r'F$_S$',linewidth=2, fmt='-o')
l5 =ax2.errorbar(eps,F_A, yerr=F_A_std, color='lightsteelblue', label=r'F$_A$',linewidth=2, fmt='-o')
ax2.set_ylabel(r'(F$_I$, F$_G$, F$_S$, F$_A$) Fairness$\downarrow$',fontsize=15, fontweight='bold', color='blue')

plt.xticks(eps,eps)
plt.legend(handles=[l1, l2,l3, l4, l5],prop={'size': 10, 'weight':'bold'}, loc='lower right')
plt.title('{} dataset (linear setting)'.format(dataset), fontsize = 15, fontweight='bold')

# plt.show()
fig.savefig('./figs/{}_sensitive_analysis.png'.format(dataset), bbox_inches='tight', pad_inches = 0)
