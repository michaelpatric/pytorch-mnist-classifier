import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define as transformações para preparar as imagens
transform = transforms.Compose([
    # Converte para Tensor e normaliza os valores de pixel entre 0 e 1
    transforms.ToTensor(),
    # Normalização (usando média e desvio padrão do MNIST)
    transforms.Normalize((0.1307,), (0.3081,))
])

# 1. Carrega o conjunto de dados de treino e teste
# (Se não existir, o PyTorch fará o download)
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 2. Cria os DataLoaders para iterar sobre os dados em batches
# Batch Size: Número de imagens processadas por vez.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print("Dados MNIST carregados com sucesso!")

# Define a arquitetura da Rede Neural
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Camada 1: 784 (pixels da imagem 28x28) -> 128 neurônios
        self.fc1 = nn.Linear(28 * 28, 128)
        # Camada 2: 128 -> 64 neurônios
        self.fc2 = nn.Linear(128, 64)
        # Camada 3 (Output): 64 -> 10 neurônios (um para cada dígito 0-9)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Achata a imagem de 28x28 para um vetor de 784 pixels
        x = x.view(-1, 28 * 28) 
        
        # Aplica função de ativação ReLU nas camadas escondidas
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output final
        return self.fc3(x)

# Instancia o modelo e define o otimizador e a função de perda
model = Net()
# Função de Perda (Loss Function): Usada para medir o erro
criterion = nn.CrossEntropyLoss()
# Otimizador: Algoritmo para ajustar os pesos (parâmetros) da rede
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Modelo de Rede Neural definido e pronto.")

def train(model, device, train_loader, optimizer, epoch):
    model.train() # Coloca o modelo em modo de treino
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move dados e rótulos para o dispositivo (CPU neste caso)
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad() # Zera gradientes de iteração anterior
        output = model(data)  # Previsão
        loss = criterion(output, target) # Calcula a perda
        loss.backward()       # Backpropagation (calcula gradientes)
        optimizer.step()      # Atualiza os pesos da rede

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
def test(model, device, test_loader):
    model.eval() # Coloca o modelo em modo de avaliação
    test_loss = 0
    correct = 0
    with torch.no_grad(): # Desativa o cálculo de gradientes (economiza memória e acelera)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Soma a perda por batch
            test_loss += criterion(output, target).item() 
            
            # Pega o índice com o valor de probabilidade mais alto (nossa previsão)
            pred = output.argmax(dim=1, keepdim=True)  
            
            # Compara a previsão com o rótulo real (target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nResultado do Teste:')
    print(f'Média da Perda: {test_loss:.4f}, Acertos: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}% de Acerto)\n')


# Define o dispositivo de treino (usaremos CPU, pois WSL2 pode ter limitações de GPU)
device = torch.device("cpu")
model.to(device)

# Loop de Treinamento
num_epochs = 3
print("-" * 50)
print(f"Iniciando treinamento por {num_epochs} épocas...")
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)


# ... Código do Loop de Treinamento ...

print("-" * 50)
print("Treinamento concluído!")

# *************** NOVA LINHA ***************
test(model, device, test_loader)