import numpy as np

class MLP:
    def __init__(self, p, q_list, m, taxa_aprendizado=0.01, max_epocas=1000, precisao=1e-5, ativacao='tanh'):
        """
        p: Número de entradas (Já contando o bias. No nosso caso, p=3)
        q_list: Lista com o número de neurônios em cada camada oculta. Ex: [5, 5] (duas camadas com 5 neurônios)
        m: Número de neurônios na camada de saída. Para classificação binária, m=1.
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
        # Ex: se p=3, q_list=[4, 4], m=1 -> tamanhos = [3, 4, 4, 1]
        tamanhos = [self.p] + self.q_list + [self.m]
        
        
        for i in range(len(tamanhos) - 1):
            n_in = tamanhos[i]
            n_out = tamanhos[i+1]
            
            # ATENÇÃO AO BIAS DA CAMADA OCULTA:
            # A primeira entrada (X) já vem com o bias (-1) inserido pelo main.py.
            # Mas para as conexões das camadas ocultas para frente, nós precisamos
            # adicionar 1 neurônio extra nas entradas para atuar como o bias interno delas.
            if i > 0:
                n_in += 1
                
            # Inicialização com valores aleatórios PEQUENOS (Crucial para o Backprop não travar)
            W_i = np.random.uniform(-0.5, 0.5, (n_out, n_in))
            self.W.append(W_i)

    def _funcao_ativacao(self, v):
        # Proteção contra Overflow (evita que o np.exp quebre com números gigantes)
        v_seguro = np.clip(v, -250, 250)
        
        if self.ativacao == 'tanh':
            # Implementação explícita da Tangente Hiperbólica (Sem a divisão do professor)
            numerador = np.exp(v_seguro) - np.exp(-v_seguro)
            denominador = np.exp(v_seguro) + np.exp(-v_seguro)
            return numerador / denominador
            
        elif self.ativacao == 'sigmoide':
            # Implementação explícita da Sigmoide Logística
            return 1.0 / (1.0 + np.exp(-v_seguro))

    def _derivada_ativacao(self, y):
        # A derivada é calculada em função da SAÍDA (y) e não do campo induzido (v)
        if self.ativacao == 'tanh':
            return 1.0 - y**2
        elif self.ativacao == 'sigmoide':
            return y * (1.0 - y)

    def fit(self, X, d):
        print(f"Iniciando treinamento do MLP com arquitetura: Entradas={self.p}, Ocultas={self.q_list}, Saída={self.m}")
        d = d.reshape(1, -1)
        _, num_amostras = X.shape
        
        for epoca in range(self.max_epocas):
            eqm_epoca = 0
            
            # Treinamento Online (Amostra por Amostra)
            for i in range(num_amostras):
                x_i = X[:, i].reshape(-1, 1)
                d_i = d[0, i]
                
                # ==========================================
                # PASSO 1: FEEDFORWARD (Ida)
                # ==========================================
                Y = [x_i] # Lista que guarda as saídas de cada camada para usar na volta
                
                for j in range(len(self.W)):
                    in_atual = Y[-1]
                    # Adiciona o bias (-1) no vetor que vai entrar na camada (se não for a primeira)
                    if j > 0:
                        in_atual = np.vstack(( [[-1.0]], in_atual ))
                        
                    # Calcula campo induzido (v) e passa pela ativação não-linear
                    v = np.dot(self.W[j], in_atual)
                    y = self._funcao_ativacao(v)
                    Y.append(y)
                    
                y_out = Y[-1] # Saída final da rede
                
                # ==========================================
                # PASSO 2: CÁLCULO DO ERRO
                # ==========================================
                erro = d_i - y_out[0, 0]
                eqm_epoca += erro**2
                
                # ==========================================
                # PASSO 3: BACKPROPAGATION (Retropropagação)
                # ==========================================
                deltas = []
                
                # 3.1 Calcula o Gradiente Local da Camada de Saída
                delta_out = erro * self._derivada_ativacao(y_out)
                deltas.insert(0, delta_out)
                
                # 3.2 Calcula os Gradientes Locais das Camadas Ocultas (De trás para frente)
                for j in range(len(self.W) - 1, 0, -1):
                    W_next = self.W[j]
                    # Retiramos a primeira coluna de W_next pois o erro não retropropaga para o Bias
                    W_next_no_bias = W_next[:, 1:] 
                    
                    delta_next = deltas[0]
                    y_curr = Y[j]
                    
                    # Regra da Cadeia: (Pesos Futuros transpostos * Delta Futuro) * Derivada Local
                    delta_curr = np.dot(W_next_no_bias.T, delta_next) * self._derivada_ativacao(y_curr)
                    deltas.insert(0, delta_curr)
                    
                # ==========================================
                # PASSO 4: ATUALIZAÇÃO DOS PESOS
                # ==========================================
                for j in range(len(self.W)):
                    in_atual = Y[j]
                    if j > 0:
                        in_atual = np.vstack(( [[-1.0]], in_atual ))
                        
                    # W_novo = W_atual + taxa * delta * entrada^T
                    self.W[j] = self.W[j] + self.eta * np.dot(deltas[j], in_atual.T)
                    
            # Cálculo e verificação do critério de parada (EQM)
            eqm_epoca /= num_amostras
            self.historico_eqm.append(eqm_epoca)
            
            if epoca > 0:
                variacao_erro = abs(self.historico_eqm[-1] - self.historico_eqm[-2])
                if variacao_erro <= self.precisao:
                    print(f"-> Convergiu na época {epoca+1}! Variação do EQM atingiu a precisão.")
                    break
                    
        else:
            print(f"-> Treinamento finalizado. Limite de {self.max_epocas} épocas atingido.")
            
        return self.historico_eqm

    def predict(self, X):
        _, num_amostras = X.shape
        y_pred = np.zeros(num_amostras)
        
        # Faz o Feedforward para cada amostra para prever a classe final
        for i in range(num_amostras):
            in_atual = X[:, i].reshape(-1, 1)
            for j in range(len(self.W)):
                if j > 0:
                    in_atual = np.vstack(( [[-1.0]], in_atual ))
                v = np.dot(self.W[j], in_atual)
                in_atual = self._funcao_ativacao(v)
                
            out_final = in_atual[0, 0]
            # Como a saída é Tanh (varia de -1 a 1), usamos o degrau bipolar (0 como limiar)
            y_pred[i] = 1.0 if out_final >= 0 else -1.0
            
        return y_pred