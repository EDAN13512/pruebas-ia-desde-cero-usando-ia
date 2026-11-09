import torch

# 1. Los Datos (La experiencia de la IA)
# Entradas (x) y lo que esperamos que responda (y = el doble de x)
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. La Neurona
# Empezamos con un "peso" (multiplicador) totalmente aleatorio.
# requires_grad=True le dice a PyTorch: "rastrea matemáticamente este valor para poder modificarlo luego".
peso = torch.randn(1, requires_grad=True)

learning_rate = 0.01 # Qué tan rápido aprende (si es muy alto, se vuelve inestable)

print(f"Peso inicial aleatorio: {peso.item():.4f}\n")

# 3. El Bucle de Entrenamiento (La IA aprendiendo)
for epoca in range(100):
    # A. Predicción: La neurona intenta adivinar
    prediccion = x_train * peso
    
    # B. El Castigo: Calculamos qué tan equivocada estuvo (Error Cuadrático Medio)
    error = torch.mean((prediccion - y_train) ** 2)
    
    # C. La Magia (Backpropagation): PyTorch calcula cómo cambiar el peso para reducir el error
    error.backward()
    
    # D. Aprender: Ajustamos el peso y reseteamos la memoria matemática
    with torch.no_grad():
        peso -= learning_rate * peso.grad
        peso.grad.zero_()
        
    if (epoca + 1) % 20 == 0:
        print(f"Época {epoca+1} | Error: {error.item():.4f} | Peso actual: {peso.item():.4f}")

# 4. La Prueba de Fuego
print("\n¡Entrenamiento terminado!")
print(f"El peso final de la neurona es: {peso.item():.4f} (Debería ser casi 2.0)")

prueba = torch.tensor([10.0])
resultado = prueba * peso
print(f"Si le pasamos un 10, la IA predice: {resultado.item():.4f}")