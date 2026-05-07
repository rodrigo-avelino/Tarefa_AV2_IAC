import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# =====================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =====================================================================
DATA_DIR = os.path.join('dados', 'RecFac')
X, y = [], []

print("Lendo diretórios, aplicando Resize (50x50)...")
pastas_individuos = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

for individuo in pastas_individuos:
    caminho_pasta = os.path.join(DATA_DIR, individuo)
    for arquivo in os.listdir(caminho_pasta):
        caminho_imagem = os.path.join(caminho_pasta, arquivo)
        try:
            img = Image.open(caminho_imagem).convert('L')
            img_resized = img.resize((50, 50), Image.Resampling.LANCZOS)
            X.append(np.array(img_resized).flatten())
            y.append(individuo)
        except Exception:
            pass

X = np.array(X)
y = np.array(y)

print(f"Total: {X.shape[0]} imagens | Dimensões: {X.shape[1]} | Classes: {len(np.unique(y))}")

# =====================================================================
# 2. NORMALIZAÇÃO E REDUÇÃO
# =====================================================================
print("Aplicando Z-Score...")
X_scaled = StandardScaler().fit_transform(X)

print("Calculando LDA (Linear Discriminant Analysis)...")
X_lda = LDA(n_components=2).fit_transform(X_scaled, y)

print("Calculando t-SNE (t-Distributed Stochastic Neighbor Embedding)...")
X_tsne = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(X_scaled)

# =====================================================================
# 3. VISUALIZAÇÃO EM COLUNA (VERTICAL)
# =====================================================================
print("Gerando painel vertical...")
palette = sns.color_palette("tab20", len(np.unique(y)))

# Layout 2 linhas x 1 coluna
fig, axes = plt.subplots(2, 1, figsize=(10, 14))

# Título principal
fig.suptitle('Análise Visual de Separabilidade (2500 px² $\\rightarrow$ 2D)', fontsize=18, fontweight='bold')

# Gráfico 1: t-SNE (Aglomerados Topológicos)
sns.scatterplot(ax=axes[0], x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=palette, legend=False, s=100, edgecolor='k', alpha=0.8)
axes[0].set_title('t-SNE (Preservação de Vizinhança Estocástica)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Dimensão Latente 1', fontsize=12)
axes[0].set_ylabel('Dimensão Latente 2', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.5)

# Gráfico 2: LDA (Prova de Separabilidade Linear)
sns.scatterplot(ax=axes[1], x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette=palette, legend='full', s=100, edgecolor='k', alpha=0.8)
axes[1].set_title('LDA (Maximização de Separação Linear)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Discriminante Linear 1', fontsize=12)
axes[1].set_ylabel('Discriminante Linear 2', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.5)

# Ajuste da legenda no gráfico inferior
axes[1].legend(title="Identidades (Classes)", bbox_to_anchor=(1.02, 1.1), loc='upper left', frameon=True)
sns.despine()

# O SEGREDO ESTÁ AQUI: Protege 4% do topo pro título e 15% da direita pra legenda
plt.tight_layout(rect=[0, 0, 0.85, 0.96])

plt.savefig('Visualizacao_Separabilidade_Vertical.png', dpi=300)
print("Imagem salva como 'Visualizacao_Separabilidade_Vertical.png'!")