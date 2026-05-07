import os
import numpy as np
import cv2

def carregar_e_processar_imagens(caminho_base, tamanho_alvo=(30, 30)):
    """
    Lê a pasta RecFac, redimensiona as imagens usando OpenCV, 
    cria a matriz X e o vetor Y.
    """
    X_lista = []
    y_labels = []
    
    # Pega as 20 pastas em ordem alfabética (cada pasta é uma classe/pessoa)
    nomes_classes = sorted(os.listdir(caminho_base)) 
    
    print(f"Carregando imagens e redimensionando para {tamanho_alvo} com OpenCV...")
    
    for idx_classe, nome_pessoa in enumerate(nomes_classes):
        caminho_pessoa = os.path.join(caminho_base, nome_pessoa)
        
        if not os.path.isdir(caminho_pessoa):
            continue
            
        for nome_imagem in os.listdir(caminho_pessoa):
            if nome_imagem.endswith(('.png', '.jpg', '.jpeg')):
                caminho_img = os.path.join(caminho_pessoa, nome_imagem)
                
                # Lê a imagem já em tons de cinza
                img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Redimensiona a imagem
                    img_redimensionada = cv2.resize(img, tamanho_alvo)
                    
                    # Achata a imagem para vetor 1D
                    img_array = img_redimensionada.flatten()
                    
                    X_lista.append(img_array)
                    y_labels.append(idx_classe)
                
    # Converte para Numpy arrays
    X_raw = np.array(X_lista).T # Transpõe para (Atributos x Amostras)
    y_raw = np.array(y_labels)
    
    print(f"Total de imagens carregadas: {X_raw.shape[1]}")
    print(f"Dimensão original de cada imagem achatada: {X_raw.shape[0]}")
    
    # 1. Padronização (Z-Score)
    medias = np.mean(X_raw, axis=1, keepdims=True)
    desvios = np.std(X_raw, axis=1, keepdims=True)
    desvios[desvios == 0] = 1.0 # Evita divisão por zero
    
    X_norm = (X_raw - medias) / desvios
    
    # 2. Adicionar o Bias (-1) na primeira linha
    n_amostras = X_norm.shape[1]
    bias = -np.ones((1, n_amostras))
    X_final = np.vstack((bias, X_norm))
    
    # 3. One-Hot Encoding (Bipolar: +1 e -1)
    num_classes = len(nomes_classes)
    Y_ohe = -np.ones((num_classes, n_amostras)) 
    
    for i, rotulo in enumerate(y_raw):
        Y_ohe[rotulo, i] = 1.0 
        
    print(f"Dimensão da Matriz X (com Bias): {X_final.shape}")
    print(f"Dimensão da Matriz Y (OHE): {Y_ohe.shape}")
    
    return X_final, Y_ohe, nomes_classes

if __name__ == "__main__":
    # Caminho ajustado considerando que você vai rodar da raiz do projeto
    caminho_recfac = os.path.join("..", "..", "dados", "RecFac") 
    
    if os.path.exists(caminho_recfac):
        X, Y, classes = carregar_e_processar_imagens(caminho_recfac, tamanho_alvo=(30, 30))
        print("\nClasses carregadas:", classes[:5], "...")
        print("Valores únicos no Y_ohe (devem ser -1 e 1):", np.unique(Y))
    else:
        print(f"Pasta não encontrada: {caminho_recfac}")