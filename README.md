# 🧠 Classificador de Dígitos MNIST com PyTorch

Este é o primeiro projeto de Deep Learning, implementando um classificador de imagens para reconhecer dígitos escritos à mão. O projeto utiliza o dataset **MNIST** e o framework **PyTorch**.

O script principal realiza o carregamento dos dados, define uma arquitetura de Rede Neural, treina o modelo e, finalmente, avalia sua performance.

## 📊 Performance do Modelo (Rede Totalmente Conectada)

O modelo utiliza uma arquitetura simples de Rede Neural Totalmente Conectada (FCN) com três camadas.

| Métrica | Valor |
| :--- | :--- |
| **Arquitetura** | Rede Neural Totalmente Conectada (FCN) |
| **Dataset** | MNIST (60.000 imagens de treino, 10.000 de teste) |
| **Épocas de Treinamento** | 3 |
| **Acurácia (Teste)** | **97.46%** |

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos e Setup

Certifique-se de que o ambiente virtual está ativo e as dependências PyTorch e NumPy estão instaladas.

```bash
# Navegue até o diretório do projeto
cd ~/Projetos/classificador_pytorch/

# Ative o ambiente virtual
source venv_ia/bin/activate

# Instale as dependências
(venv_ia) pip install torch torchvision torchaudio numpy matplotlib