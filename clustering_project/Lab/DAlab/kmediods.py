from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
# Load list of points for cluster analysis.
sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)
lines = open("t4.8k","r")
inp = []
for line in lines:
	cords = line.split()
	if len(cords) != 2:
		continue
	inp.append([float(cords[0]), float(cords[1])])
	

# Set random initial medoids.
initial_medoids = [1, 800, 1400, 672, 763, 926]

# Create instance of K-Medoids algorithm.
kmedoids_instance = kmedoids(inp, initial_medoids)

# Run cluster analysis and obtain results.
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()

# Display clusters.
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, inp)
visualizer.show()