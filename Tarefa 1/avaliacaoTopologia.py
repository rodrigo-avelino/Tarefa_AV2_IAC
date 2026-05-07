import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlp import MLP

# Pastas e Arquivos
PASTA_VISUAL = 'topologias_visual_vertical'
PASTA_BOXPLOT = 'topologias_boxplots_puros'
ARQUIVO_TXT = 'tabelas_topologias.txt'

os.makedirs(PASTA_VISUAL, exist_ok=True)
os.makedirs(PASTA_BOXPLOT, exist_ok=True)

# Limpa o arquivo TXT no início para não misturar com testes velhos
with open(ARQUIVO_TXT, 'w', encoding='utf-8') as f:
    f.write("% Arquivo gerado automaticamente com os resultados das topologias\n\n")

def carregar_dados(caminho):
    dados = np.loadtxt(caminho, delimiter=',')
    X = dados[:, :2]
    d = dados[:, 2]
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_final = np.hstack((-np.ones((X_norm.shape[0], 1)), X_norm))
    return X_final.T, d

def train_test_split_manual(X, d, test_size=0.2):
    num_amostras = X.shape[1]
    indices = np.arange(num_amostras)
    np.random.shuffle(indices) 
    split_idx = int(num_amostras * (1 - test_size))
    return X[:, indices[:split_idx]], X[:, indices[split_idx:]], d[indices[:split_idx]], d[indices[split_idx:]]

def calcular_metricas(d_true, y_pred):
    TP = np.sum((d_true == 1) & (y_pred == 1))
    TN = np.sum((d_true == -1) & (y_pred == -1))
    FP = np.sum((d_true == -1) & (y_pred == 1))
    FN = np.sum((d_true == 1) & (y_pred == -1))
    acc = (TP + TN) / len(d_true) if len(d_true) > 0 else 0.0
    sensibilidade = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    especificidade = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precisao = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade) if (precisao + sensibilidade) > 0 else 0.0
    matriz_confusao = np.array([[TN, FP], [FN, TP]])
    return acc, sensibilidade, especificidade, precisao, f1_score, matriz_confusao

def plotar_fronteira_mlp_ax_matplotlib(X, d, modelo, titulo, ax):
    x1_vals = X[1, :]
    x2_vals = X[2, :]
    x_min, x_max = x1_vals.min() - 0.5, x1_vals.max() + 0.5
    y_min, y_max = x2_vals.min() - 0.5, x2_vals.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    grid_pontos = np.c_[xx.ravel(), yy.ravel()]
    X_grid = np.hstack((-np.ones((grid_pontos.shape[0], 1)), grid_pontos)).T
    Z = modelo.predict(X_grid)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    cores = np.where(d == 1, 'gold', 'indigo') 
    ax.scatter(x1_vals, x2_vals, c=cores, s=30, edgecolor='k')
    ax.set_title(titulo, fontsize=12, fontweight='bold')

def salvar_tabela_latex_txt(res, nome_topologia):
    linhas = [
        "\n% ======================================",
        f"% TABELA LATEX PARA: {nome_topologia}",
        "% ======================================",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Resumo Estatístico - Topologia {nome_topologia.replace('_', ' ')}}}",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Métrica} & \\textbf{Média} & \\textbf{Desv. Pad} & \\textbf{Máximo} & \\textbf{Mínimo} \\\\",
        "\\midrule"
    ]
    
    nomes = ['Acurácia', 'Sensibilidade', 'Especificidade', 'Precisão', 'F1-Score']
    chaves = ['acc', 'sens', 'spec', 'prec', 'f1']
    
    for nome, chave in zip(nomes, chaves):
        v = np.array(res[chave])
        linhas.append(f"{nome} & {np.mean(v):.4f} & {np.std(v):.4f} & {np.max(v):.4f} & {np.min(v):.4f} \\\\")
        
    linhas.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "% ======================================\n"
    ])
    
    # Salva adicionando no arquivo TXT
    with open(ARQUIVO_TXT, 'a', encoding='utf-8') as f:
        f.write("\n".join(linhas) + "\n")
    print(f"   => Tabela salva no arquivo {ARQUIVO_TXT}!")

def avaliar_topologias_monte_carlo(X, d, rodadas=10):
    print("\n" + "="*80)
    print(f"MONTE CARLO DE TOPOLOGIAS ({rodadas} RODADAS)")
    print("="*80)
    
    topologias = {
        "Regra_do_Valor_Medio": [2],                 
        "Kolmogorov": [5],                  
        "Erro_Tolerado": [14],             
        "Saturacao_Analitica": [280],            
        "Lang_Witbrock": [15, 15],    
        "Superdimensionada": [100, 100, 50] 
    }
    
    for i, (nome, q_list) in enumerate(topologias.items(), start=1):
        print(f"\n🚀 Simulação: {nome} (Camadas: {q_list})")
        epocas = 1500 if "Superdimensionada" in nome else 800
        
        res = {'acc':[], 'sens':[], 'spec':[], 'prec':[], 'f1':[]}
        melhor_acc = -1
        melhor_dados = None
        
        for r in range(rodadas):
            print(f"   -> Rodada {r+1}/{rodadas}...", end="\r")
            X_train, X_test, d_train, d_test = train_test_split_manual(X, d, test_size=0.2)
            modelo = MLP(p=3, q_list=q_list, m=1, taxa_aprendizado=0.05, max_epocas=epocas)
            historico = modelo.fit(X_train, d_train)
            
            y_pred_test = modelo.predict(X_test)
            acc, sens, spec, prec, f1, conf = calcular_metricas(d_test, y_pred_test)
            
            res['acc'].append(acc)
            res['sens'].append(sens)
            res['spec'].append(spec)
            res['prec'].append(prec)
            res['f1'].append(f1)
            
            if acc > melhor_acc:
                melhor_acc = acc
                melhor_dados = (historico, conf, X_test, d_test, modelo)
                
        # SALVA GRÁFICO VERTICAL
        historico, conf, X_test, d_test, modelo = melhor_dados
        fig1, axes1 = plt.subplots(3, 1, figsize=(5.5, 13))
        fig1.suptitle(f"{nome} | Melhor Acc: {melhor_acc*100:.1f}%", fontsize=14, fontweight='bold', y=0.98)
        
        axes1[0].plot(historico, color='royalblue', lw=2)
        axes1[0].set_title("Curva de Aprendizado", fontsize=12)
        plotar_fronteira_mlp_ax_matplotlib(X_test, d_test, modelo, "Fronteira de Decisão", axes1[1])
        sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=['-1', '+1'], yticklabels=['-1', '+1'], ax=axes1[2], cbar=False)
        axes1[2].set_title("Matriz de Confusão", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        fig1.subplots_adjust(top=0.93, hspace=0.35)
        fig1.savefig(f"{PASTA_VISUAL}/{i}_{nome}_Visual.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # SALVA BOXPLOT
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        dados_box = [res['acc'], res['sens'], res['spec'], res['prec'], res['f1']]
        bplot = ax2.boxplot(dados_box, patch_artist=True, labels=['Acc', 'Sens', 'Spec', 'Prec', 'F1'])
        ax2.set_title(f"Distribuição Estatística - {nome}", fontsize=12, fontweight='bold')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle='--', alpha=0.5)
        for patch in bplot['boxes']: patch.set_facecolor('lightblue')
        fig2.savefig(f"{PASTA_BOXPLOT}/{i}_{nome}_Boxplot.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # SALVA A TABELA NO TXT
        salvar_tabela_latex_txt(res, nome)

if __name__ == "__main__":
    np.random.seed(42) 
    X, d = carregar_dados('spiral_d.csv')
    avaliar_topologias_monte_carlo(X, d, rodadas=500)