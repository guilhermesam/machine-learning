def split_x_y(matrix):
    """
    separa um array numpy baseado em colunas para treinamento
    de modelos de ML

    :param matrix: uma matriz numpy
    :return x: variáveis independentes
    :return y: variável dependente
    """
    
    x = matrix[:,:-1]
    y = matrix[:,-1:]

    return x,y