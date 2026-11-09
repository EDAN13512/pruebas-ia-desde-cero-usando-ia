import torch
import torch.nn as nn
import torch.optim as optim
import json

# 1. Cargar datos desde el JSON
with open('datos.json', 'r', encoding='utf-8') as f:
    datos_crudos = json.load(f)

# Extraer frases y etiquetas
datos = [(item['frase'], item['respuesta']) for item in datos_crudos]

# 2. Construir el Vocabulario Dinámico
vocabulario = {"<UNK>": 0}
indice_actual = 1

for frase, _ in datos:
    for palabra in frase.lower().split():
        if palabra not in vocabulario:
            vocabulario[palabra] = indice_actual
            indice_actual += 1

def frase_a_tensor(frase):
    indices = [vocabulario.get(palabra, 0) for palabra in frase.lower().split()]
    if not indices: indices = [0]
    # Redimensionamos a (1, longitud_secuencia) para que LSTM lo entienda
    return torch.tensor(indices).unsqueeze(0) 

# 3. La Arquitectura Secuencial (La cura contra la "Bolsa de Palabras")
class ClasificadorLSTM(nn.Module):
    def __init__(self, tamano_vocab, dim_embedding, dim_oculta):
        super().__init__()
        self.embedding = nn.Embedding(tamano_vocab, dim_embedding)
        # La LSTM procesa las palabras en orden y guarda un "estado mental" (memoria)
        self.lstm = nn.LSTM(dim_embedding, dim_oculta, batch_first=True)
        self.salida = nn.Linear(dim_oculta, 1)
        self.sigmoide = nn.Sigmoid()

    def forward(self, x):
        # 1. Convertir números a vectores matemáticos
        x = self.embedding(x) 
        
        # 2. Pasar por la LSTM. 
        # 'out' tiene la salida de cada palabra. 
        # 'hn' tiene el resumen final de toda la frase tras leer la última palabra.
        out, (hn, cn) = self.lstm(x) 
        
        # 3. Usamos solo el resumen final para tomar la decisión
        resumen_final = hn[-1] 
        
        x = self.salida(resumen_final)
        x = self.sigmoide(x)
        return x

# dim_oculta=16 es diminuto y perfecto para 4 GB de RAM.
ia = ClasificadorLSTM(len(vocabulario), dim_embedding=8, dim_oculta=16)

# 4. Entrenamiento
criterio = nn.BCELoss()
optimizador = optim.Adam(ia.parameters(), lr=0.01)

print(f"Vocabulario de {len(vocabulario)} palabras. Entrenando red LSTM...")
for epoca in range(400):
    error_total = 0
    for frase, etiqueta in datos:
        optimizador.zero_grad()
        
        tensor_frase = frase_a_tensor(frase)
        tensor_etiqueta = torch.tensor([[etiqueta]]) # Formato exacto que pide BCELoss
        
        prediccion = ia(tensor_frase)
        error = criterio(prediccion, tensor_etiqueta)
        
        error.backward()
        optimizador.step()
        error_total += error.item()

print(f"¡Entrenamiento terminado! Error final: {error_total/len(datos):.4f}\n")

# 5. Modo Interactivo
print("-" * 30)
print("IA Secuencial lista. (Escribe 'salir' para terminar).")
print("-" * 30)

while True:
    prompt = input("Tú: ")
    if prompt.lower() == 'salir':
        break
        
    tensor_prueba = frase_a_tensor(prompt)
    
    with torch.no_grad():
        probabilidad = ia(tensor_prueba).item()
        
    if probabilidad >= 0.5:
        respuesta = "IA: Sí"
    else:
        respuesta = "IA: No"
        
    print(f"{respuesta} (Seguridad: {probabilidad*100:.1f}%)")