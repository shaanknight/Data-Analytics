from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.cure import cure;
from pyclustering.utils import read_sample;
from pyclustering.samples.definitions import FCPS_SAMPLES;

# Input data in following format [ [0.1, 0.5], [0.3, 0.1], ... ].
input_data = read_sample(FCPS_SAMPLES.SAMPLE_CHAINLINK);
lines = open("t4.8k","r")
inp = []
for line in lines:
	cords = line.split()
	if len(cords) != 2:
		continue
	inp.append([float(cords[0]), float(cords[1])])

# Allocate clusters.
cure_instance = cure(inp, 6);
cure_instance.process();
clusters = cure_instance.get_clusters();

# Visualize allocated clusters.
visualizer = cluster_visualizer();
visualizer.append_clusters(clusters, inp);
visualizer.show();
