import torch
import torch.nn as nn
import torch.optim as optim

# 1. El Dataset (Nuestra IA va a memorizar y entender esta frase)
texto = "inteligencia artificial"

# 2. Tokenización (Convertir letras a números y viceversa)
letras_unicas = sorted(list(set(texto)))
tamano_vocabulario = len(letras_unicas)

letra_a_numero = {letra: i for i, letra in enumerate(letras_unicas)}
numero_a_letra = {i: letra for i, letra in enumerate(letras_unicas)}

# Vamos a enseñarle a leer de a 3 letras para adivinar la 4ta
# Ejemplo: lee "int", debe adivinar "e". Lee "nte", debe adivinar "l".
tamano_contexto = 4
X, Y = [], []
for i in range(len(texto) - tamano_contexto):
    contexto = texto[i:i+tamano_contexto]
    objetivo = texto[i+tamano_contexto]
    X.append([letra_a_numero[l] for l in contexto])
    Y.append(letra_a_numero[objetivo])

X = torch.tensor(X)
Y = torch.tensor(Y)

# 3. La Arquitectura del Modelo de Lenguaje
class GeneradorTexto(nn.Module):
    def __init__(self, vocab_size, embedding_dim, contexto):
        super().__init__()
        # El Embedding convierte los números de las letras en vectores matemáticos densos
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        # Capa oculta: multiplica el tamaño del contexto por el tamaño del vector
        self.oculta = nn.Linear(contexto * embedding_dim, 32)
        self.relu = nn.ReLU()
        # Capa de salida: debe devolver una probabilidad para cada letra posible de nuestro vocabulario
        self.salida = nn.Linear(32, vocab_size)

    def forward(self, x):
        x = self.emb(x) # Convertir números a vectores
        x = x.view(x.shape[0], -1) # Aplanar los datos para la capa lineal
        x = self.oculta(x)
        x = self.relu(x)
        x = self.salida(x)
        return x

ia = GeneradorTexto(tamano_vocabulario, embedding_dim=10, contexto=tamano_contexto)

# 4. Herramientas para texto
# Usamos CrossEntropyLoss, el estándar de la industria para clasificación y texto
criterio = nn.CrossEntropyLoss()
# Adam es un optimizador más moderno y rápido que SGD
optimizador = optim.Adam(ia.parameters(), lr=0.01)

print("Entrenando la red de texto...")
# 5. Entrenamiento
for epoca in range(300):
    optimizador.zero_grad()
    prediccion = ia(X)
    error = criterio(prediccion, Y)
    error.backward()
    optimizador.step()

print("¡Entrenamiento terminado! Error final:", error.item())

# 6. La Prueba de Fuego: ¡Generar texto nuevo!
# Le damos la semilla inicial "int" y dejamos que escriba sola el resto
semilla = "inte"
texto_generado = semilla

for _ in range(20): # Que genere 20 letras más
    # Convertir el contexto actual a números
    x_prueba = torch.tensor([[letra_a_numero[l] for l in semilla]])
    
    # Predecir
    prediccion = ia(x_prueba)
    
    # Quedarnos con el número que tiene la mayor probabilidad
    indice_ganador = torch.argmax(prediccion).item()
    letra_predicha = numero_a_letra[indice_ganador]
    
    # Añadir al texto final
    texto_generado += letra_predicha
    
    # Actualizar la semilla (descartamos la primera letra, agregamos la nueva)
    semilla = semilla[1:] + letra_predicha

print("\n--- RESULTADO DE LA IA ---")
print(texto_generado)
