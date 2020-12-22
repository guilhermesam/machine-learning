import numpy as np

class KMedoids:
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.converged = False
        self.medoids = []
        self.medoids_cost = [0] * n_clusters
        self.trained = False

    # Cálculo da distância euclidiana entre dois pontos p1 e p2
    def euclidian_distance(self, p1, p2):
        return sum((p1 - p2) ** 2) ** 0.5

    # Verifica se o modelo convergiu. O modelo converge caso o novo set de medoids escolhido seja
    # igual aos medoids atuais do modelo.
    def __has_converged(self, new_medoids):
        return set([tuple(x) for x in self.medoids]) == set([tuple(x) for x in new_medoids])

    # O algoritmo inicia escolhendo k pontos de dado aleatórios do dataset para serem os medoids iniciais
    def initialize_medoids(self, data):
        random_indexes = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        self.medoids = data[random_indexes]

    # A partir da lista de medoids ideais para cada cluster, a função recebe um conjunto de dados em
    # formato array numpy e retorna um dicionário, tendo como chave o rótulo de um cluster e como valor
    # uma lista com cada registro do conjunto de dados. Os dados são atribuídos ao cluster através da
    # obtenção da menor distância euclidiana ponto -> medoid do cluster.
    def classify_data(self, data):
        if not self.trained:
            raise Exception("Modelo não foi treinado ainda!")

        classifications = {}
        for index_cluster in range(self.n_clusters):
            classifications[index_cluster] = []
        for data_point in data:
            distances = []
            for medoid in self.medoids:
                distances.append(self.euclidian_distance(data_point, medoid))

            min_distance_index = distances.index(np.min(distances))
            classifications[min_distance_index].append(data_point)
        
        return classifications

    # Recebe uma lista dos clusters que irão agrupar dada ponto do conjunto de dados.
    def __change_medoids(self, clusters):
        new_medoids = []
        for i in range(0, self.n_clusters):
            new_medoid = self.medoids[i]
            old_medoids_cost = self.medoids_cost[i]
            for j in range(len(clusters[i])):
                
                # O custo dos atuais pontos de dados a serem comparados com o custo atual
                current_medoids_cost = 0
                for index_datapoint in range(len(clusters[i])):
                    current_medoids_cost += self.euclidian_distance(clusters[i][j], clusters[i][index_datapoint])
                
                # Se o custo atual é menor, então o novo medoid do cluster será o atual ponto de dado
                if current_medoids_cost < old_medoids_cost:
                    new_medoid = clusters[i][j]
                    old_medoids_cost = current_medoids_cost
            
            new_medoids.append(new_medoid)

        return new_medoids

    def update_medoids(self, data, new_labels):
        # Obter os novos rótulos e atribuí-los para cada registro do conjunto de dados        
        clusters = []
        for i in range(0,self.n_clusters):
            cluster = []
            for j in range(len(data)):
                if (new_labels[j] == i):
                    cluster.append(data[j])
            clusters.append(cluster)
        
        # Obter o novo conjunto de medoids baseado nos novos clusters
        new_medoids = self.__change_medoids(clusters)
        
        # Caso o modelo ainda não tenha convergido, os novos medoids são adicionados como "medoids atuais"
        # Caso contrário, ou seja, caso os medoids do modelo não tenham mudado, o atributo self.converged é
        # setado como True
        if not self.__has_converged(new_medoids):
            self.medoids = new_medoids
        else:
            self.converged = True

    def fit(self, data):
        self.initialize_medoids(data)
    
        for i in range(self.max_iter):
            # armazena quais são os rótulos da iteração atual
            current_labels = []
            for medoid in range(0,self.n_clusters):
                # A dissimilaridade dos clusters é calculada
                self.medoids_cost[medoid] = 0
                for k in range(data.shape[0]):
                    # Armazenar a distância do ponto de dado atual do dataset para cada medoid
                    current_distances = []                    
                    for j in range(0,self.n_clusters):
                        current_distances.append(self.euclidian_distance(self.medoids[j], data[k]))
                    # Adicionar à lista de rótulos atuais o índice da lista de distâncias com o menor valor para o
                    # ponto de dados atual. 
                    current_labels.append(current_distances.index(min(current_distances)))
                    
                    self.medoids_cost[medoid] += min(current_distances)

            # Chamar a função de atualização dos medoids com os novos rótulos
            self.update_medoids(data, current_labels)
            
            # Caso o modelo tenha convergido, a execução é interrompida
            if self.converged:
                self.trained = True
                break

        return self.medoids