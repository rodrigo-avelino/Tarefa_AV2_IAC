import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from perceptron import Perceptron
from adaline import Adaline
from mlp import MLP

PASTA_BATALHA_VISUAL = 'batalha_visual_vertical'
PASTA_BATALHA_BOXPLOT = 'batalha_boxplots_puros'
ARQUIVO_TXT_FINAL = 'tabelas_batalha_final.txt'

os.makedirs(PASTA_BATALHA_VISUAL, exist_ok=True)
os.makedirs(PASTA_BATALHA_BOXPLOT, exist_ok=True)

with open(ARQUIVO_TXT_FINAL, 'w', encoding='utf-8') as f:
    f.write("% Arquivo gerado com os resultados do simulador final\n\n")

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

def plotar_fronteira_matplotlib(X, d, modelo, titulo, ax):
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

def salvar_plot_extremo_vertical(modelo_nome, metrica_nome, extremo, valor, dados_rodada):
    modelo, historico, conf, X_test, d_test = dados_rodada
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 13))
    fig.suptitle(f"{modelo_nome} | {extremo} {metrica_nome}: {valor:.4f}", fontsize=14, fontweight='bold', y=0.98)
    cor = 'firebrick' if modelo_nome == 'Perceptron' else 'darkorange' if modelo_nome == 'Adaline' else 'royalblue'
    
    axes[0].plot(historico, color=cor, lw=2)
    axes[0].set_title(f"Curva de Aprendizado", fontsize=12)
    plotar_fronteira_matplotlib(X_test, d_test, modelo, "Fronteira de Decisão", axes[1])
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=['-1', '+1'], yticklabels=['-1', '+1'], ax=axes[2], cbar=False)
    axes[2].set_title("Matriz de Confusão", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.35) 
    fig.savefig(f"{PASTA_BATALHA_VISUAL}/{modelo_nome}_{metrica_nome}_{extremo}.png", dpi=300, bbox_inches='tight')
    plt.close(fig) 

def salvar_tabela_comparativa_txt(memoria, chave_metrica, nome_bonito):
    linhas = [
        "\n% ======================================",
        f"% TABELA LATEX PARA: {nome_bonito} (Comparativo)",
        "% ======================================",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Desempenho Comparativo - {nome_bonito}}}",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Modelos} & \\textbf{Média} & \\textbf{Desv. Pad} & \\textbf{Máximo} & \\textbf{Mínimo} \\\\",
        "\\midrule"
    ]
    
    for mod in ['Perceptron', 'Adaline', 'MLP']:
        v = np.array(memoria[mod][chave_metrica])
        linhas.append(f"{mod} & {np.mean(v):.4f} & {np.std(v):.4f} & {np.max(v):.4f} & {np.min(v):.4f} \\\\")
        
    linhas.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "% ======================================\n"
    ])
    
    with open(ARQUIVO_TXT_FINAL, 'a', encoding='utf-8') as f:
        f.write("\n".join(linhas) + "\n")
    print(f"   => Tabela comparativa de {nome_bonito} salva!")

def simulador_batalha_final(X, d, rodadas=500):
    print("\n" + "="*60)
    print(f"INICIANDO A BATALHA FINAL ({rodadas} RODADAS)")
    print("="*60)
    
    nomes_modelos = ['Perceptron', 'Adaline', 'MLP']
    chaves = ['acc', 'sens', 'spec', 'prec', 'f1']
    nomes = ['Acurácia', 'Sensibilidade', 'Especificidade', 'Precisão', 'F1-Score']
    
    memoria = {mod: {'acc': [], 'sens': [], 'spec': [], 'prec': [], 'f1': [], 'dados_rodada': []} for mod in nomes_modelos}
    
    for r in range(rodadas):
        print(f"   -> Rodada {r+1}/{rodadas}...", end="\r")
        X_train, X_test, d_train, d_test = train_test_split_manual(X, d, test_size=0.2)
        modelos = {
            'Perceptron': Perceptron(taxa_aprendizado=0.01, max_epocas=100),
            'Adaline': Adaline(taxa_aprendizado=0.0001, max_epocas=200, precisao=1e-5),
            'MLP': MLP(p=3, q_list=[15, 15], m=1, taxa_aprendizado=0.05, max_epocas=500)
        }
        for nome, mod in modelos.items():
            hist = mod.fit(X_train, d_train)
            y_pred = mod.predict(X_test)
            acc, sens, spec, prec, f1, conf = calcular_metricas(d_test, y_pred)
            
            memoria[nome]['acc'].append(acc)
            memoria[nome]['sens'].append(sens)
            memoria[nome]['spec'].append(spec)
            memoria[nome]['prec'].append(prec)
            memoria[nome]['f1'].append(f1)
            memoria[nome]['dados_rodada'].append((mod, hist, conf, X_test, d_test))
            
    print("\n\n-> Salvando as 30 imagens verticais...")
    for mod_nome in nomes_modelos:
        for c, n in zip(chaves, nomes):
            v = memoria[mod_nome][c]
            idx_max = np.argmax(v)
            idx_min = np.argmin(v)
            salvar_plot_extremo_vertical(mod_nome, n, "MAIOR", v[idx_max], memoria[mod_nome]['dados_rodada'][idx_max])
            salvar_plot_extremo_vertical(mod_nome, n, "MENOR", v[idx_min], memoria[mod_nome]['dados_rodada'][idx_min])
            
    print("\n-> Gerando Boxplots Puros e Tabelas LaTeX...")
    for c, n in zip(chaves, nomes):
        fig, ax = plt.subplots(figsize=(6, 5))
        dados_box = [memoria['Perceptron'][c], memoria['Adaline'][c], memoria['MLP'][c]]
        bplot = ax.boxplot(dados_box, patch_artist=True, labels=['Perceptron', 'Adaline', 'MLP'])
        ax.set_title(f"Distribuição - {n}", fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        cores = ['#ff9999', '#ffcc99', '#99ccff']
        for patch, color in zip(bplot['boxes'], cores): patch.set_facecolor(color)
        fig.savefig(f"{PASTA_BATALHA_BOXPLOT}/Boxplot_{n}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        salvar_tabela_comparativa_txt(memoria, c, n)

if __name__ == "__main__":
    np.random.seed(42) 
    X, d = carregar_dados('spiral_d.csv')
    simulador_batalha_final(X, d, rodadas=500)