import numpy as np

class Adaline:
    def __init__(self, taxa_aprendizado=0.001, max_epocas=1000, precisao=1e-5):
        self.eta = taxa_aprendizado
        self.max_epocas = max_epocas
        self.precisao = precisao
        self.w = None
        self.historico_eqm = [] # Guarda o Erro Quadrático Médio por época

    def funcao_ativacao_treinamento(self, u):
        """No Adaline, a ativação durante o treino é linear (retorna o próprio u)."""
        return u

    def funcao_ativacao_classificacao(self, u):
        """Na hora de classificar, usamos o Degrau Bipolar."""
        return np.where(u >= 0, 1.0, -1.0)

    def fit(self, X, d):
        num_atributos, num_amostras = X.shape
        
        # 1. Inicialização dos pesos
        self.w = np.random.uniform(-0.1, 0.1, (num_atributos, 1))
        d = d.reshape(1, -1)
        
        print("Iniciando treinamento do ADALINE...")
        
        for epoca in range(self.max_epocas):
            eqm_epoca = 0
            
            # Treinamento Online (amostra por amostra)
            for i in range(num_amostras):
                x_i = X[:, i].reshape(-1, 1)
                d_i = d[0, i]
                
                # Campo induzido
                u = np.dot(self.w.T, x_i)
                
                # Saída linear
                y = self.funcao_ativacao_treinamento(u)
                
                # Erro linear
                erro = d_i - y[0, 0]
                
                # Acumula o erro quadrático para a época
                eqm_epoca += erro**2
                
                # Atualização dos pesos (Regra Delta)
                self.w = self.w + self.eta * erro * x_i
                
            # Calcula o EQM dividindo pelo número de amostras
            eqm_epoca = eqm_epoca / num_amostras
            self.historico_eqm.append(eqm_epoca)
            
            # 2. Critério de Parada: Variação do EQM
            if epoca > 0:
                variacao_erro = abs(self.historico_eqm[-1] - self.historico_eqm[-2])
                if variacao_erro <= self.precisao:
                    print(f"-> Convergiu na época {epoca+1}! Variação do EQM atingiu a precisão de {self.precisao}.")
                    break
        else:
            print(f"-> Treinamento finalizado. Limite de {self.max_epocas} épocas atingido.")
            
        return self.historico_eqm

    def predict(self, X):
        u = np.dot(self.w.T, X)
        y = self.funcao_ativacao_classificacao(u)
        return y[0, :]