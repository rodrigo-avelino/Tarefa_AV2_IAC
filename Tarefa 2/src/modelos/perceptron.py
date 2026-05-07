import numpy as np

class PerceptronOVR:
    def __init__(self, p, num_classes, taxa_aprendizado=0.01, max_epocas=1000):
        """
        p: Número de atributos de entrada (incluindo o bias).
        num_classes: Número de pessoas a serem reconhecidas (20).
        """
        self.p = p
        self.num_classes = num_classes
        self.eta = taxa_aprendizado
        self.max_epocas = max_epocas
        
        # Matriz de pesos: 20 linhas (um Perceptron por classe) x P colunas (atributos)
        # Inicialização idêntica à Parte 1 (-0.1 a 0.1)
        self.W = np.random.uniform(-0.1, 0.1, (self.num_classes, self.p))
        
        # Guarda a quantidade de erros por época para plotarmos a curva de aprendizado
        self.historico_erros = []

    def funcao_ativacao(self, u):
        """
        Função Degrau Bipolar.
        Retorna 1.0 se o campo induzido (u) for >= 0, caso contrário -1.0.
        """
        return np.where(u >= 0, 1.0, -1.0)

    def fit(self, X, D_ohe):
        """
        Treina o modelo Perceptron OVR.
        X: Matriz de entrada com shape (p, num_amostras)
        D_ohe: Matriz de alvos em One-Hot Encoding (num_classes, num_amostras)
        """
        _, num_amostras = X.shape
        
        print(f"Iniciando treinamento do Perceptron OVR (Classes: {self.num_classes}, Entradas: {self.p})...")
        
        for epoca in range(self.max_epocas):
            erros_na_epoca = 0
            
            # O treinamento do Perceptron é "online" (amostra por amostra)
            for i in range(num_amostras):
                # Pega a amostra i como um vetor coluna (p, 1)
                x_i = X[:, i].reshape(-1, 1)
                
                # Pega o alvo da amostra i (20, 1)
                d_i = D_ohe[:, i].reshape(-1, 1)
                
                # 1. Calcula o campo induzido para os 20 Perceptrons de uma vez
                u = np.dot(self.W, x_i)
                
                # 2. Aplica a função de ativação
                y = self.funcao_ativacao(u)
                
                # 3. Calcula os 20 erros (Vetor)
                erro = d_i - y
                
                # 4. Regra de Aprendizado do Perceptron: Atualiza os pesos se houver erro
                # Se a rede errou a classe de ALGUM dos 20 neurônios, atualizamos e contamos 1 erro
                if np.any(erro != 0):
                    # W(t+1) = W(t) + eta * erro * x(t)^T
                    self.W = self.W + self.eta * np.dot(erro, x_i.T)
                    erros_na_epoca += 1
                    
            self.historico_erros.append(erros_na_epoca)
            
            # Critério de parada: se não errou nenhuma amostra, o modelo convergiu
            if erros_na_epoca == 0:
                print(f"-> Convergiu na época {epoca+1}!")
                break
                
        if erros_na_epoca != 0:
            print(f"-> Treinamento finalizado. Limite de {self.max_epocas} épocas atingido sem convergência total (restam {erros_na_epoca} erros na última época).")
            
        return self.historico_erros

    def predict(self, X):
        """
        Realiza a predição para novas amostras usando a estratégia "Winner-Takes-All".
        X: Matriz de entrada com shape (p, num_amostras)
        """
        # Multiplicação matricial direta para todas as amostras e todas as 20 classes
        u = np.dot(self.W, X)
        
        # A classe prevista é o índice (linha) do neurônio que gerou o MAIOR campo induzido (u)
        y_pred_indices = np.argmax(u, axis=0)
        
        return y_pred_indices