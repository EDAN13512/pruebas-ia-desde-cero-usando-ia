import torch
import torch.nn as nn
import torch.optim as optim

# 1. Definimos la Arquitectura de la Red (El "Cerebro")
class MiPrimeraRed(nn.Module):
    def __init__(self):
        super().__init__()
        # Capa 1: Entran 2 valores, salen 4 (Capa oculta)
        self.oculta = nn.Linear(2, 4)
        # Capa 2: Entran los 4 de la oculta, sale 1 resultado final
        self.salida = nn.Linear(4, 1)
        # Función de activación: ReLU (hace que la red pueda aprender patrones complejos)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.oculta(x)
        x = self.relu(x)   # La magia no lineal
        x = self.salida(x)
        return x

# Instanciamos la IA
ia = MiPrimeraRed()
print("Arquitectura de tu IA:")
print(ia)

# 2. Los Datos: Vamos a enseñarle la operación matemática de Sumar (x1 + x2)
x_train = torch.tensor([[1.0, 1.0], [2.0, 3.0], [5.0, 5.0], [10.0, -2.0]])
y_train = torch.tensor([[2.0], [5.0], [10.0], [8.0]])

# 3. Herramientas de entrenamiento
criterio = nn.MSELoss() # Calcula el error
optimizador = optim.SGD(ia.parameters(), lr=0.01) # Hace el Backpropagation automático

# 4. Entrenamiento
for epoca in range(500):
    optimizador.zero_grad()      # Limpiar memoria
    prediccion = ia(x_train)     # Predecir
    error = criterio(prediccion, y_train) # Calcular qué tan mal fue
    error.backward()             # Magia matemática
    optimizador.step()           # Ajustar pesos

# 5. La Prueba
print("\n¡Entrenamiento terminado!")
# Le pedimos que sume 7 + 4
prueba = torch.tensor([7.0, 4.0])
resultado = ia(prueba)
print(f"Si le pasamos [7.0, 4.0], la IA predice: {resultado.item():.4f} (Debería ser cercano a 11.0)")