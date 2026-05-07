import numpy as np

class MLP:
    def __init__(self, p, q_list, m, taxa_aprendizado=0.01, max_epocas=1000, precisao=1e-5, ativacao='tanh'):
        """
        p: Número de atributos de entrada (Já contando o bias. Para fotos 30x30, p=901)
        q_list: Lista com o número de neurônios em cada camada oculta. Ex: [15, 15]
        m: Número de neurônios na camada de saída. (Para este problema, m=20)
        """
        self.p = p
        self.q_list = q_list
        self.m = m
        self.eta = taxa_aprendizado
        self.max_epocas = max_epocas
        self.precisao = precisao
        self.ativacao = ativacao
        self.historico_eqm = []
        
        # 1. Criação Dinâmica das Matrizes de Pesos (W)
        self.W = []
        tamanhos = [self.p] + self.q_list + [self.m]
        
        for i in range(len(tamanhos) - 1):
            n_in = tamanhos[i]
            n_out = tamanhos[i+1]
            
            # Adiciona 1 neurônio extra nas entradas para atuar como o bias interno das camadas ocultas
            if i > 0:
                n_in += 1
                
            # Inicialização com valores aleatórios pequenos simétricos
            W_i = np.random.uniform(-0.1, 0.1, (n_out, n_in))
            self.W.append(W_i)

    def _funcao_ativacao(self, v):
        # Proteção contra Overflow
        v_seguro = np.clip(v, -250, 250)
        
        if self.ativacao == 'tanh':
            numerador = np.exp(v_seguro) - np.exp(-v_seguro)
            denominador = np.exp(v_seguro) + np.exp(-v_seguro)
            return numerador / denominador
            
        elif self.ativacao == 'sigmoide':
            return 1.0 / (1.0 + np.exp(-v_seguro))

    def _derivada_ativacao(self, y):
        if self.ativacao == 'tanh':
            return 1.0 - (y ** 2)
        elif self.ativacao == 'sigmoide':
            return y * (1.0 - y)

    def fit(self, X, D_ohe):
        """
        X: Matriz de características (P x N)
        D_ohe: Matriz de rótulos em One-Hot Encoding (20 x N)
        """
        print(f"Iniciando treinamento do MLP Multiclasse (Entradas: {self.p}, Ocultas: {self.q_list}, Saída: {self.m})")
        _, num_amostras = X.shape
        
        for epoca in range(self.max_epocas):
            eqm_epoca = 0
            
            # Treinamento Online (Amostra por Amostra)
            for i in range(num_amostras):
                x_i = X[:, i].reshape(-1, 1)       # (P x 1)
                d_i = D_ohe[:, i].reshape(-1, 1)   # (20 x 1)
                
                # ==========================================
                # PASSO 1: FEEDFORWARD
                # ==========================================
                Y = [x_i] 
                
                for j in range(len(self.W)):
                    in_atual = Y[-1]
                    if j > 0:
                        in_atual = np.vstack(( [[-1.0]], in_atual ))
                        
                    v = np.dot(self.W[j], in_atual)
                    y = self._funcao_ativacao(v)
                    Y.append(y)
                    
                y_out = Y[-1] # Saída final da rede (20x1)
                
                # ==========================================
                # PASSO 2: CÁLCULO DO ERRO
                # ==========================================
                erro = d_i - y_out
                # O erro da amostra é a soma do erro quadrático nos 20 neurônios de saída
                eqm_epoca += np.sum(erro**2)
                
                # ==========================================
                # PASSO 3: BACKPROPAGATION
                # ==========================================
                deltas = []
                
                # Gradiente Local da Camada de Saída (Vetorizado: 20x1)
                delta_out = erro * self._derivada_ativacao(y_out)
                deltas.insert(0, delta_out)
                
                # Gradientes Locais das Camadas Ocultas
                for j in range(len(self.W) - 1, 0, -1):
                    W_next = self.W[j]
                    W_next_no_bias = W_next[:, 1:] 
                    
                    delta_next = deltas[0]
                    y_curr = Y[j]
                    
                    delta_curr = np.dot(W_next_no_bias.T, delta_next) * self._derivada_ativacao(y_curr)
                    deltas.insert(0, delta_curr)
                    
                # ==========================================
                # PASSO 4: ATUALIZAÇÃO DOS PESOS
                # ==========================================
                for j in range(len(self.W)):
                    in_atual = Y[j]
                    if j > 0:
                        in_atual = np.vstack(( [[-1.0]], in_atual ))
                        
                    self.W[j] = self.W[j] + self.eta * np.dot(deltas[j], in_atual.T)
                    
            # Cálculo e verificação do critério de parada (EQM Geral da Época)
            eqm_epoca /= (2 * num_amostras)
            self.historico_eqm.append(eqm_epoca)
            
            if epoca > 0:
                variacao_erro = abs(self.historico_eqm[-1] - self.historico_eqm[-2])
                if variacao_erro <= self.precisao:
                    print(f"-> Convergiu na época {epoca+1}! (EQM: {eqm_epoca:.6f})")
                    break
        else:
            print(f"-> Limite de {self.max_epocas} épocas atingido. (EQM final: {eqm_epoca:.6f})")
            
        return self.historico_eqm

    def predict(self, X):
        """
        Retorna as classes previstas (índices de 0 a 19) para as amostras de teste.
        """
        _, num_amostras = X.shape
        y_pred_indices = np.zeros(num_amostras, dtype=int)
        
        for i in range(num_amostras):
            in_atual = X[:, i].reshape(-1, 1)
            for j in range(len(self.W)):
                if j > 0:
                    in_atual = np.vstack(( [[-1.0]], in_atual ))
                v = np.dot(self.W[j], in_atual)
                in_atual = self._funcao_ativacao(v)
                
            out_final = in_atual # Matriz (20x1) com o sinal de cada neurônio
            
            # O neurônio que "vence" (maior valor próximo a 1) dita a classe da imagem
            y_pred_indices[i] = np.argmax(out_final, axis=0)[0]
            
        return y_pred_indices