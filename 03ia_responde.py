import torch
import torch.nn as nn
import torch.optim as optim

# 1. El Dataset de Entrenamiento (Preguntas y Respuestas)
# 1.0 = Sí | 0.0 = No
datos = [
    ("eres una inteligencia artificial", 1.0),
    ("tienes un cuerpo fisico", 0.0),
    ("puedes procesar datos", 1.0),
    ("necesitas dormir de noche", 0.0),
    ("estamos en una computadora", 1.0),
    ("eres un ser humano", 0.0),
    ("funcionas con codigo python", 1.0),
    ("te gusta comer pizza", 0.0)
]

# 2. Construir el Vocabulario (Ahora por palabras, no por letras)
vocabulario = {"<UNK>": 0} # Token para palabras que la IA no conoce
indice_actual = 1

for frase, _ in datos:
    for palabra in frase.split():
        if palabra not in vocabulario:
            vocabulario[palabra] = indice_actual
            indice_actual += 1

# Función para convertir frases en tensores de números
def frase_a_tensor(frase):
    indices = [vocabulario.get(palabra, 0) for palabra in frase.lower().split()]
    # Si la frase está vacía o son puras palabras desconocidas, evitamos que crashee
    if not indices: indices = [0] 
    return torch.tensor(indices)

# 3. La Arquitectura del Clasificador Binario
class ClasificadorSiNo(nn.Module):
    def __init__(self, tamano_vocab, dim_embedding):
        super().__init__()
        # Convierte cada palabra en un vector de números
        self.embedding = nn.Embedding(tamano_vocab, dim_embedding)
        # Capa lineal para decidir el Sí o No
        self.salida = nn.Linear(dim_embedding, 1)
        # Sigmoide: comprime cualquier número para que caiga exactamente entre 0 y 1
        self.sigmoide = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        # Promediamos el significado de todas las palabras de la frase
        x = x.mean(dim=0) 
        x = self.salida(x)
        x = self.sigmoide(x)
        return x

ia = ClasificadorSiNo(len(vocabulario), dim_embedding=8)

# 4. Herramientas de Entrenamiento
# BCELoss (Binary Cross Entropy) es el estándar para clasificaciones de Sí/No (1 o 0)
criterio = nn.BCELoss()
optimizador = optim.Adam(ia.parameters(), lr=0.05)

print("Entrenando la red para responder Sí o No...")
for epoca in range(300):
    error_total = 0
    for frase, etiqueta in datos:
        optimizador.zero_grad()
        
        tensor_frase = frase_a_tensor(frase)
        tensor_etiqueta = torch.tensor([etiqueta])
        
        prediccion = ia(tensor_frase)
        error = criterio(prediccion, tensor_etiqueta)
        
        error.backward()
        optimizador.step()
        error_total += error.item()

print(f"¡Entrenamiento terminado! Error final: {error_total/len(datos):.4f}\n")

# 5. Modo Interactivo (Chat por teclado)
print("-" * 30)
print("IA lista. Escribe tu pregunta (escribe 'salir' para terminar).")
print("-" * 30)

while True:
    prompt = input("Tú: ")
    if prompt.lower() == 'salir':
        break
        
    tensor_prueba = frase_a_tensor(prompt)
    
    # Desactivamos el cálculo de gradientes porque ya no estamos entrenando
    with torch.no_grad():
        probabilidad = ia(tensor_prueba).item()
        
    # La IA toma la decisión
    if probabilidad >= 0.5:
        respuesta = "IA: Sí"
    else:
        respuesta = "IA: No"
        
    # Te mostramos la "seguridad" de la IA como un extra para que veas qué pasa adentro
    print(f"{respuesta} (Seguridad: {probabilidad*100:.1f}%)")
