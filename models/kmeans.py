import numpy as np

class KMeans:
    def __init__(self, n_clusters, tol, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = {}

    def euclidian_distance(self, p1, p2):
        return sum((p1 - p2) ** 2) ** 0.5

    # O algoritmo inicia escolhendo k pontos de dado aleatórios do dataset para serem os medoids iniciais
    def initialize_medoids(self, data):
        random_indexes = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        for i in range(self.n_clusters):
            self.centroids[i] = data[random_indexes[i]]

    def fit(self,data):
        self.initialize_medoids(data)

        # para cada etapa de iteração
        for epoch in range(self.max_iter):
            self.classifications = {}

            # inicializar o dicionário de classificações
            for i in range(self.n_clusters):
                self.classifications[i] = []

            # para cada linha do dataset
            for feature_set in data:
                # calcular as distâncias da linha para cada centroid
                distances = [self.euclidian_distance(feature_set, self.centroids[centroid]) for centroid in self.centroids]
                # obter o índice da menor distância
                min_distance_index = distances.index(min(distances))
                # adicionar à linha, a classificação
                self.classifications[min_distance_index].append(feature_set)

            # salvar os centroids atuais como anteriores para atualização
            previous_centroids = self.centroids

            # para cada rótulo de classificação
            for key in self.classifications:
                self.centroids[key] = np.average(self.classifications[key], axis=0)

            for c in self.centroids:
                original_centroid = previous_centroids[c]
                current_centroid = self.centroids[c]
                
                if np.sum((current_centroid - original_centroid) / (original_centroid * 100)) < self.tol:
                    break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[c]) for c in self.centroids]
        return distances.index(min(distances))