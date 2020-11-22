from sklearn.cluster import KMeans


class MyKMeans:
    def __init__(self, n_clusters, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def __call__(self, inputs):
        k_means = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(inputs)
        return k_means.cluster_centers_, k_means.labels_
















