# 🧠 Inteligência Artificial Computacional - AV2

Este repositório contém o código-fonte integral, os conjuntos de dados e as análises estatísticas referentes ao trabalho da Segunda Avaliação (AV2) da disciplina de Inteligência Artificial Computacional da UNIFOR.

## 🎯 Escopo do Projeto

O trabalho explora a dualidade entre complexidade geométrica e dimensionalidade:
- **Parte I (Espiral):** Desafio de classificação não-linear em baixa dimensão (p=2), onde a profundidade da rede MLP foi essencial.
- **Parte II (Faces):** Reconhecimento facial multiclasse em alta dimensão (p=2501), onde a esparsidade do espaço permitiu a separabilidade linear.

## 📁 Estrutura do Repositório

📦 Tarefa_AV2_IAC
 ┣ 📂 dados/               # Datasets (Spiral e RecFac)
 ┣ 📂 src/                 # Código-fonte (Modelos O.O. e Scripts)
 ┃ ┣ 📜 main.py            # Orquestrador da simulação
 ┃ ┗ 📜 scprit.py          # Script de análise de separabilidade
 ┣ 📂 analises/            # Resultados (Matrizes, Curvas e Boxplots)
 ┣ 📂 relatorio/           # Documentação técnica em PDF
 ┣ 📜 requirements.txt     # Dependências do projeto
 ┗ 📜 README.md            # Documentação principal

## 🛠️ Tecnologias e Requisitos

Os classificadores (Perceptron, Adaline e MLP) foram desenvolvidos **integralmente de forma manual**. As bibliotecas externas foram utilizadas apenas para suporte de infraestrutura:

- **NumPy:** Operações matriciais e Backpropagation.
- **Matplotlib & Seaborn:** Visualização estatística.
- **OpenCV & PIL:** Pré-processamento de imagem.
- **Scikit-Learn:** Auxílio em normalização e projeções (LDA/t-SNE).

## 🚀 Como Executar

1. Clone o repositório.
2. Instale as dependências:
`pip install -r requirements.txt`
3. Execute a simulação principal na pasta src:
`python main.py`

---
*Desenvolvido por Rodrigo Avelino - Engenharia de Computação (UNIFOR).*