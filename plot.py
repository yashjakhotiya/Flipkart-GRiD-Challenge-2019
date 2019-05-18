from matplotlib import pyplot as plt
from matplotlib import patches 
import numpy as np 
import pandas, logging, imageio

#logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

output = pandas.read_csv("output.csv")
indices = np.random.choice(output.shape[0], size=(2, 2))

fig, ax = plt.subplots(2, 2)
for i in range(2):
	for j in range(2):
		index = indices[i][j]
		ax[i][j].imshow(imageio.imread("../images/{}".format(output["image_name"][index])))
		x1, x2, y1, y2 = output["x1"][index], output["x2"][index], output["y1"][index], output["y2"][index]
		print(x1, x2, y1, y2)
		rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), linewidth=1, edgecolor='r', facecolor='none')
		ax[i][j].add_patch(rect)
plt.show() 