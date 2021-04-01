try:
	import argparse
	import logging
	import sys
	import numpy as np

except ImportError as error:
	print(error)
	print()
	print("You must install the requirements:")
	print("  pip3 install --upgrade pip")
	print("  pip3 install -r requirements.txt ")
	print()
	sys.exit(-1)

class KMeans:
    def __init__(self, n_clusters, error_tolerance, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.error_tolerance = error_tolerance
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
                
                if np.sum((current_centroid - original_centroid) / (original_centroid * 100)) < self.error_tolerance:
                    break
        
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[c]) for c in self.centroids]
        return distances.index(min(distances))

def imprime_config(args):
	'''
	Mostra os argumentos recebidos e as configurações processadas
	:args: parser.parse_args
	'''
	logging.info("Argumentos:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
	logging.info("Configurações:")
	for k, v in sorted(vars(args).items()):
		logging.info("\t{0}: {1}".format(k, v))
	logging.info("")

def main():
	'''
	Programa principal
	:return:
	'''

	# Definição de argumentos
	parser = argparse.ArgumentParser(description='Trabalho APA - KMeans')

	help_msg = "número de clusters"
	parser.add_argument("--nclusters", "-k", help=help_msg, default=6, type=int)

	help_msg = "número de iterações"
	parser.add_argument("--niterations", "-iters", help=help_msg, default=1000, type=int)

	help_msg = "número de iterações"
	parser.add_argument("--errortol", "-tol", help=help_msg, default=1, type=int)

	help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--verbosity", "-v", help=help_msg, default=logging.INFO, type=int)

	# Lê argumentos da linha de comando
	args = parser.parse_args()

	# configura o mecanismo de logging
	if args.verbosity == logging.DEBUG:
		# mostra mais detalhes
		logging.basicConfig(format='%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s',
							datefmt='%Y-%m-%d,%H:%M:%S', level=args.verbosity)

	else:
		logging.basicConfig(format='%(message)s',
							datefmt='%Y-%m-%d,%H:%M:%S', level=args.verbosity)

	# imprime configurações para fins de log
	imprime_config(args)

	kmeans = KMeans(args.nclusters, args.errortol, args.niterations)

if __name__ == '__main__':
    sys.exit(main())