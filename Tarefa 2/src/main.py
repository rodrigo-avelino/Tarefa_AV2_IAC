import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importando nossos módulos (Ajuste os caminhos conforme sua estrutura)
from utilidades.pre_processamento import carregar_e_processar_imagens
from utilidades.metricas import calcular_acuracia, calcular_matriz_confusao, treinar_testar_split_manual

from modelos.perceptron import PerceptronOVR
from modelos.adaline import AdalineOVR
from modelos.mlp import MLP

# Pega a pasta 'src' onde o main.py está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Volta uma pasta (..) e entra em analises/resultados
PASTA_RESULTADOS = os.path.join(BASE_DIR, '..', 'analises', 'resultados')
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

def plotar_matriz_20x20(matriz, nomes_classes, titulo, caminho_salvar):
    """Plota a matriz de confusão gigante com Seaborn"""
    plt.figure(figsize=(14, 12)) # Figura grande para caber as 20 classes
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=nomes_classes, yticklabels=nomes_classes,
                annot_kws={"size": 9}) # Fonte menor nos números
    
    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Classe Real', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(caminho_salvar, dpi=300)
    plt.close()

def assistente_de_relatorio(matriz, nomes_classes, modelo_nome, extremo):
    """Analisa a matriz e imprime no console ONDE o modelo errou para ajudar na discussão."""
    print(f"\n--- ANÁLISE PARA O RELATÓRIO: {modelo_nome} ({extremo} Acurácia) ---")
    
    acertos_por_classe = np.diag(matriz)
    total_por_classe = np.sum(matriz, axis=1)
    
    # Evita divisão por zero caso a classe não tenha caído no conjunto de teste desta rodada
    acc_por_classe = np.zeros(20)
    for i in range(20):
        if total_por_classe[i] > 0:
            acc_por_classe[i] = acertos_por_classe[i] / total_por_classe[i]
            
    piores_classes = np.argsort(acc_por_classe)[:3] # Pega os 3 piores
    
    print("Categorias com as PIORES predições (Escreva isso na discussão):")
    for idx in piores_classes:
        if total_por_classe[idx] > 0:
            print(f"  -> '{nomes_classes[idx]}': Acertou {acertos_por_classe[idx]} de {total_por_classe[idx]} fotos.")
            
            # Acha com quem ele mais confundiu
            erros_nesta_linha = matriz[idx, :].copy()
            erros_nesta_linha[idx] = 0 # Zera os acertos para focar nos erros
            if np.sum(erros_nesta_linha) > 0:
                confundido_com = np.argmax(erros_nesta_linha)
                print(f"      Motivo: Confundiu frequentemente com '{nomes_classes[confundido_com]}'.")

def executar_simulacao(X, Y_ohe, y_labels_idx, nomes_classes, rodadas=10):
    print(f"\nINICIANDO MONTE CARLO ({rodadas} RODADAS) - RECONHECIMENTO FACIAL")
    
    p = X.shape[0] # Número de atributos (ex: 901)
    num_classes = len(nomes_classes) # 20
    
    nomes_modelos = ['Perceptron OVR', 'Adaline OVR', 'MLP']
    memoria = {mod: {'acc': [], 'dados_rodada': []} for mod in nomes_modelos}
    
    for r in range(rodadas):
        print(f"\n-> Rodada {r+1}/{rodadas}...")
        X_train, X_test, Y_ohe_train, Y_ohe_test, y_idx_train, y_idx_test = treinar_testar_split_manual(X, Y_ohe, y_labels_idx, test_size=0.2)
        
        # Instanciando Modelos (Ajuste os hiperparâmetros conforme seus testes)
        modelos = {
            'Perceptron OVR': PerceptronOVR(p=p, num_classes=num_classes, taxa_aprendizado=0.01, max_epocas=500),
            'Adaline OVR': AdalineOVR(p=p, num_classes=num_classes, taxa_aprendizado=1e-5, max_epocas=10000), 
            'MLP': MLP(p=p, q_list=[25,22], m=num_classes, taxa_aprendizado=0.001, max_epocas=1000)
        }
        
        for nome, mod in modelos.items():
            # CORREÇÃO 1: Garante que a função fit() sempre seja executada!
            mod.fit(X_train, Y_ohe_train)
            
            # CORREÇÃO 2: Resgata o histórico de forma segura
            if hasattr(mod, 'historico_eqm'):
                hist = mod.historico_eqm
            elif hasattr(mod, 'historico_erros'):
                hist = mod.historico_erros
            else:
                hist = [0, 0] # Placeholder caso o modelo não rastreie nenhum histórico

            y_pred = mod.predict(X_test)
            acc = calcular_acuracia(y_idx_test, y_pred)
            conf = calcular_matriz_confusao(y_idx_test, y_pred, num_classes)
            
            memoria[nome]['acc'].append(acc)
            memoria[nome]['dados_rodada'].append((hist, conf))

    # =================================================================
    # PASSOS 6: EXTRAIR EXTREMOS, PLOTAR MATRIZES E CURVAS
    # =================================================================
    print("\nProcessando Extremos e Gerando Imagens (Passo 6)...")
    for mod_nome in nomes_modelos:
        vetor_acc = memoria[mod_nome]['acc']
        idx_max = np.argmax(vetor_acc)
        idx_min = np.argmin(vetor_acc)
        
        casos = [("MAIOR", idx_max), ("MENOR", idx_min)]
        
        for extremo, idx in casos:
            acc_val = vetor_acc[idx]
            hist, conf = memoria[mod_nome]['dados_rodada'][idx]
            nome_arquivo = mod_nome.replace(" ", "_")
            
            # 1. Matriz de Confusão
            titulo_matriz = f"Matriz de Confusão: {mod_nome}\n{extremo} Acurácia ({acc_val*100:.1f}%)"
            plotar_matriz_20x20(conf, nomes_classes, titulo_matriz, f"{PASTA_RESULTADOS}/{nome_arquivo}_Matriz_{extremo}.png")
            
            # Assistente de Relatório (Imprime os padrões de erro)
            assistente_de_relatorio(conf, nomes_classes, mod_nome, extremo)
            
            # 2. Curva de Aprendizado (CORREÇÃO 3: Agora plota para TODOS os modelos)
            plt.figure(figsize=(8, 5))
            plt.plot(hist, color='blue', lw=2)
            plt.title(f"Curva de Aprendizado: {mod_nome} ({extremo} Acurácia: {acc_val*100:.1f}%)", fontweight='bold')
            plt.xlabel("Épocas")
            plt.ylabel("Erro / EQM")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(f"{PASTA_RESULTADOS}/{nome_arquivo}_Curva_{extremo}.png", dpi=300)
            plt.close()

    # =================================================================
    # PASSO 7: TABELA E BOXPLOT DA ACURÁCIA
    # =================================================================
    print("\nGerando Tabela e Boxplot (Passo 7)...")
    
    # BOXPLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    dados_boxplot = [memoria[mod]['acc'] for mod in nomes_modelos]
    bplot = ax.boxplot(dados_boxplot, patch_artist=True, labels=nomes_modelos)
    
    cores = ['#ff9999', '#ffcc99', '#99ccff']
    for patch, color in zip(bplot['boxes'], cores):
        patch.set_facecolor(color)
        
    ax.set_title(f"Boxplot da Acurácia - Reconhecimento Facial ({rodadas} Rodadas)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Acurácia (0 a 1)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{PASTA_RESULTADOS}/Comparativo_Acuracia_Boxplot.png", dpi=300)
    plt.close()

    # TABELA ESTATÍSTICA (Visual e Console)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    colunas = ['Modelos', 'Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor']
    dados_tabela = []
    
    print("\n" + "="*70)
    print(f"{'TABELA DE ACURÁCIA (RODADAS: ' + str(rodadas) + ')':^70}")
    print("="*70)
    print(f"{'Modelo':<20} | {'Média':<10} | {'Desvio':<10} | {'Max':<10} | {'Min':<10}")
    print("-" * 70)
    
    for mod in nomes_modelos:
        vetor = np.array(memoria[mod]['acc'])
        linha = [
            mod,
            f"{np.mean(vetor):.4f}",
            f"{np.std(vetor):.4f}",
            f"{np.max(vetor):.4f}",
            f"{np.min(vetor):.4f}"
        ]
        dados_tabela.append(linha)
        print(f"{mod:<20} | {np.mean(vetor):.4f}     | {np.std(vetor):.4f}     | {np.max(vetor):.4f}     | {np.min(vetor):.4f}")
        
    tabela = ax.table(cellText=dados_tabela, colLabels=colunas, loc='center', cellLoc='center')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(12)
    tabela.scale(1.0, 2.0)
    tabela.auto_set_column_width(col=list(range(len(colunas))))
    
    for key, cell in tabela.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d9e2f3')
        elif key[1] == 0:
            cell.set_text_props(weight='bold', ha='left')
            
    plt.title(f"Resultados da Acurácia ({rodadas} Rodadas)", fontweight='bold', fontsize=14, pad=15)
    plt.savefig(f"{PASTA_RESULTADOS}/Tabela_Acuracia.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ Processo Finalizado! Verifique a pasta 'analises/resultados_parte2'.")

if __name__ == "__main__":
    np.random.seed(42) # Semente isolada no main
    
    # 1. Carrega os dados (Usando o BASE_DIR para não ter erro de caminho)
    caminho_recfac = os.path.join(BASE_DIR, "..", "dados", "RecFac") 
    X_final, Y_ohe, nomes_classes = carregar_e_processar_imagens(caminho_recfac, tamanho_alvo=(50, 50))
    
    # Recuperando o y bruto (índices) a partir do Y_ohe para usar na acurácia
    y_labels_idx = np.argmax(Y_ohe, axis=0) 
    
    # 2. Executa a Batalha
    executar_simulacao(X_final, Y_ohe, y_labels_idx, nomes_classes, rodadas=100)