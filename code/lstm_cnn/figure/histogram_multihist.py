import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

colors = ['darkgreen', 'forestgreen', 'limegreen', 'greenyellow']
label = ['Precision', 'Recall', 'F1-score', 'Best Accuracy']

n_groups = 12 

#std_1 = (2, 3, 4, 1, 2)
#std_2 = (3, 5, 2, 3, 3)

precision = (97.26, 95.98, 94.77, 97.64, 97.34, 97.29, 97.77, 97.95, 97.36, 94.22, 97.46, 94.26)
recall = (97.17, 94.44, 94.11, 97.77, 97.28, 97.24, 97.71, 97.89, 97.38, 94.37, 97.58, 94.87)
f1 = (97.10, 94.94, 93.56, 97.68, 97.24, 97.20, 97.63, 97.86, 97.31, 93.64, 97.43, 94.44)
ba = (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100)

fig, ax = plt.subplots(figsize=(12, 6))

index = np.arange(n_groups)
bar_width = 0.2

opacity = 1.0
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, precision, bar_width,
                alpha=opacity, color=colors[0],
                yerr=0, error_kw=error_config,
                label=label[0])

rects2 = ax.bar(index + bar_width, recall, bar_width,
                alpha=opacity, color=colors[1],
                yerr=0, error_kw=error_config,
                label=label[1])

rects1 = ax.bar(index + bar_width * 2, f1, bar_width,
                alpha=opacity, color=colors[2],
                yerr=0, error_kw=error_config,
                label=label[2])

rects2 = ax.bar(index + bar_width * 3, ba, bar_width,
                alpha=opacity, color=colors[3],
                yerr=0, error_kw=error_config,
                label=label[3])


ax.set_xlabel('Number of Conv + Number of Full')
ax.set_ylabel('Percentage (100%)')
ax.set_title('The performances of models with different Conv + Full')
ax.set_xticks(index + bar_width * 2)
ax.set_xticklabels(('2+2', '2+3', '2+4', '3+2', '3+3', '3+4', '5+2', '5+3', '5+4', '7+2', '7+3', '7+4'))
ax.legend()
fig.tight_layout()
ax.set_ylim(92,102)
#plt.ylim([93,102])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.show()



