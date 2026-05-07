import numpy as np

def calcular_acuracia(y_true, y_pred):
    """
    Calcula a taxa de acerto.
    y_true e y_pred devem ser vetores 1D com os índices das classes (0 a 19).
    """
    acertos = np.sum(y_true == y_pred)
    total = len(y_true)
    return acertos / total if total > 0 else 0.0

def calcular_matriz_confusao(y_true, y_pred, num_classes=20):
    """
    Constrói a Matriz de Confusão 20x20 manualmente.
    Linhas = Classes Reais, Colunas = Classes Preditas.
    """
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for real, predita in zip(y_true, y_pred):
        matriz[real, predita] += 1
    return matriz

def treinar_testar_split_manual(X, Y_ohe, y_labels_idx, test_size=0.2):
    """
    Divide os dados embaralhando os índices.
    X: Matriz de entrada com Bias (P+1 x N)
    Y_ohe: Matriz de rótulos One-Hot (20 x N)
    y_labels_idx: Vetor 1D com os índices das classes (0 a 19) para facilitar validação
    """
    num_amostras = X.shape[1]
    indices = np.arange(num_amostras)
    np.random.shuffle(indices)
    
    split_idx = int(num_amostras * (1 - test_size))
    idx_train, idx_test = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[:, idx_train], X[:, idx_test]
    Y_ohe_train, Y_ohe_test = Y_ohe[:, idx_train], Y_ohe[:, idx_test]
    y_idx_train, y_idx_test = y_labels_idx[idx_train], y_labels_idx[idx_test]
    
    return X_train, X_test, Y_ohe_train, Y_ohe_test, y_idx_train, y_idx_test