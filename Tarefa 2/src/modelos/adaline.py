import numpy as np

class AdalineOVR:
    def __init__(self, p, num_classes, taxa_aprendizado=0.001, max_epocas=1000, precisao=1e-5):
        self.p = p
        self.num_classes = num_classes
        self.eta = taxa_aprendizado
        self.max_epocas = max_epocas
        self.precisao = precisao
        
        self.W = np.random.uniform(-0.01, 0.01, (self.num_classes, self.p))
        self.historico_eqm = []

    def fit(self, X, D_ohe):
        _, num_amostras = X.shape
        
        print(f"Iniciando treino do Adaline OVR (Classes: {self.num_classes}, Entradas: {self.p})...")
        
        for epoca in range(self.max_epocas):
            eqm_epoca = 0
            
            for i in range(num_amostras):
                x_i = X[:, i].reshape(-1, 1)
                d_i = D_ohe[:, i].reshape(-1, 1)
                
                # 1. Campo induzido
                u = np.dot(self.W, x_i)
                
                # 2. O Adaline usa a ativação puramente linear (y = u) para calcular o erro
                erro = d_i - u
                eqm_epoca += np.sum(erro**2)
                
                # 3. Atualização dos Pesos (Regra Delta)
                self.W = self.W + self.eta * np.dot(erro, x_i.T)
                
            # Calcula o EQM da época (soma dos erros de todas as 20 classes e divide pelas amostras)
            eqm_epoca /= (2 * num_amostras)
            self.historico_eqm.append(eqm_epoca)
            
            # Critério de Parada Padrão Ouro (Variação do Erro)
            if epoca > 0:
                variacao_erro = abs(self.historico_eqm[-1] - self.historico_eqm[-2])
                if variacao_erro <= self.precisao:
                    print(f"-> Adaline convergiu na época {epoca+1}! Variação do erro atingiu a precisão.")
                    break
        else:
            print(f"-> Adaline atingiu o limite de {self.max_epocas} épocas.")
            
        return self.historico_eqm

    def predict(self, X):
        """
        Retorna o índice da classe prevista (0 a 19).
        """
        u = np.dot(self.W, X)
        y_pred_indices = np.argmax(u, axis=0)
        return y_pred_indices