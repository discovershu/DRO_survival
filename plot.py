import matplotlib.pyplot as plt
import numpy as np

dataset = 'SEER' ##'FLC', 'SUPPORT', 'SEER'

eps = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

if dataset == 'FLC':

    C_I = [0.7669, 0.7788, 0.7864, 0.7905, 0.7893, 0.7882]

    F_I = [0.0013, 0.0057, 0.0196, 0.0872, 0.3554, 1.1860]

    F_G = [0.0069, 0.0153, 0.0386, 0.1514, 0.5957, 1.9051]

    F_S = [0.0188, 0.0351, 0.0816, 0.2967, 1.0165, 2.1382]
elif dataset == 'SUPPORT':
    C_I = [0.5326, 0.5445, 0.5434, 0.5403, 0.5500, 0.5684]

    F_I = [0.0000, 0.0003, 0.0018, 0.0103, 0.0230, 0.0417]

    F_G = [0.0023, 0.0022, 0.0026, 0.0019, 0.0100, 0.0310]

    F_S = [0.0050, 0.0072, 0.0089, 0.0102, 0.0318, 0.0912]
else:
    C_I = [0.6058, 0.6566, 0.6942, 0.7032, 0.6989, 0.6987]

    F_I = [0.0046, 0.0068, 0.0179, 0.1272, 0.6080, 0.6181]

    F_G = [0.0151, 0.0074, 0.0186, 0.1797, 0.8318, 0.8435]

    F_S = [0.1073, 0.1014, 0.0939, 0.1953, 0.6970, 0.7088]


F_A = (1/3)*(np.asarray(F_I)+np.asarray(F_G)+np.asarray(F_S))


fig, ax = plt.subplots()

l1, = ax.plot(eps,C_I, color='red', label=r'c-index', linewidth=2)
ax.set_xlabel(r"$\alpha$", fontsize = 15, fontweight='bold')
ax.set_ylabel(r'c-index$\uparrow$',fontsize=15, fontweight='bold')
# ax.set_ylim((0.5, 0.7))

ax2=ax.twinx()
l2, =ax2.plot(eps,F_I,color='navy', label=r'F$_I$',linewidth=2)
l3, =ax2.plot(eps,F_G,color='blue', label=r'F$_G$',linewidth=2)
l4, =ax2.plot(eps,F_S,color='royalblue', label=r'F$_S$',linewidth=2)
l5, =ax2.plot(eps,F_A,color='lightsteelblue', label=r'F$_A$',linewidth=2)
ax2.set_ylabel(r'(F$_I$, F$_G$, F$_S$, F$_A$) Fairness$\downarrow$',fontsize=15, fontweight='bold', color='blue')

plt.xticks(eps,eps)
plt.legend(handles=[l1,l2,l3, l4, l5],prop={'size': 10, 'weight':'bold'}, loc='lower right')
plt.title('{} dataset (linear setting)'.format(dataset), fontsize = 15, fontweight='bold')

plt.show()
# fig.savefig('./figs/{}_sensitive_analysis.png'.format(dataset), bbox_inches='tight', pad_inches = 0)
