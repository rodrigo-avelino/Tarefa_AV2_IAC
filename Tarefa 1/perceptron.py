import numpy as np

class Perceptron:
    def __init__(self, taxa_aprendizado=0.01, max_epocas=1000):
        self.eta = taxa_aprendizado
        self.max_epocas = max_epocas
        self.w = None
        self.historico_erros = [] # Guarda a quantidade de erros por época para plotarmos a curva de aprendizado

    def funcao_ativacao(self, u):
        """
        Função Degrau Bipolar.
        Retorna 1 se o campo induzido (u) for >= 0, caso contrário -1.
        """
        return np.where(u >= 0, 1.0, -1.0)

    def fit(self, X, d):
        """
        Treina o modelo Perceptron.
        X: Matriz de entrada com shape (num_atributos, num_amostras) -> Ex: (3, N) com o bias já incluso
        d: Vetor de saídas desejadas com shape (num_amostras,)
        """
        num_atributos, num_amostras = X.shape
        
        # 1. Inicialização dos pesos (w) 
        self.w = np.random.uniform(-0.1, 0.1, (num_atributos, 1))
        
        # Transformando d em um vetor linha (1, N) para facilitar os índices
        d = d.reshape(1, -1)
        
        print("Iniciando treinamento do Perceptron...")
        
        for epoca in range(self.max_epocas):
            erros_na_epoca = 0
            
            # O treinamento do Perceptron é "online" (amostra por amostra)
            for i in range(num_amostras):
                # Pega a amostra i como um vetor coluna (num_atributos, 1)
                x_i = X[:, i].reshape(-1, 1)
                d_i = d[0, i]
                
                # 2. Calcula o campo induzido: u = w^T * x
                u = np.dot(self.w.T, x_i)
                
                # 3. Aplica a função de ativação para obter a saída da rede (y)
                y = self.funcao_ativacao(u)
                
                # 4. Calcula o erro: e = d - y
                erro = d_i - y[0, 0]
                
                # 5. Regra de Aprendizado do Perceptron: Atualiza os pesos se houver erro
                if erro != 0:
                    # w(t+1) = w(t) + eta * erro * x(t)
                    self.w = self.w + self.eta * erro * x_i
                    erros_na_epoca += 1
                    
            self.historico_erros.append(erros_na_epoca)
            
            # Critério de parada: se não errou nenhuma amostra, o modelo convergiu
            if erros_na_epoca == 0:
                print(f"-> Convergiu na época {epoca+1}!")
                break
                
        if erros_na_epoca != 0:
            print(f"-> Treinamento finalizado. Limite de {self.max_epocas} épocas atingido sem convergência (esperado para dados não-lineares).")
            
        return self.historico_erros

    def predict(self, X):
        """
        Realiza a predição para novas amostras.
        X: Matriz de entrada com shape (num_atributos, num_amostras)
        """
        # Multiplicação matricial direta para todas as amostras: u = w^T * X
        u = np.dot(self.w.T, X)
        
        # Aplica a ativação e retorna como um array 1D
        y = self.funcao_ativacao(u)
        return y[0, :]