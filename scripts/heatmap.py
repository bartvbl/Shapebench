import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys

a = np.loadtxt(open(sys.argv[1], "rb"), delimiter=",", skiprows=0)


heatmap = plt.imshow(a, cmap='hot', interpolation='nearest')

if len(sys.argv) > 2 and sys.argv[2] == 'true':
	def fmt(x, pos):
		a, b = '{:.2e}'.format(x).split('e')
		b = int(b)
		return r'${} \times 10^{{{}}}$'.format(a, b)


	plt.colorbar(heatmap, format=ticker.FuncFormatter(fmt))

plt.show()