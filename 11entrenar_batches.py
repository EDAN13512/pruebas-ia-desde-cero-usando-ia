import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import re  # IMPORTANTE: Librería nativa de Python para expresiones regulares

# 1. Cargar y Limpiar el Dataset
with open("codigo.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

# Todo a minúsculas para unificar
texto = texto.lower()

# EL FILTRO MÁGICO: 
# Elimina todo excepto letras (a-z), acentos (áéíóú), la ñ, espacios (\s) y los dos puntos (:)
texto = re.sub(r'[^a-záéíóúñ\s:]', '', texto)

# 2. Extraer Vocabulario (El resto de tu código sigue igual a partir de aquí)
palabras = texto.split()
vocab_unico = sorted(list(set(palabras)))

# Creamos los diccionarios
palabra_a_idx = {p: i for i, p in enumerate(vocab_unico)}
idx_a_palabra = {i: p for i, p in enumerate(vocab_unico)}

# ¡GUARDAMOS EL VOCABULARIO! Así el script de chat no necesita leer el texto original
with open('vocab_codigo.json', 'w', encoding='utf-8') as f:
    json.dump({'p_a_i': palabra_a_idx, 'i_a_p': idx_a_palabra}, f)
print(f"Vocabulario de {len(vocab_unico)} palabras guardado en JSON.")

# Convertimos todo el texto a una lista gigante de números
datos_numericos = [palabra_a_idx[p] for p in palabras]

# 3. Hiperparámetros (MicroGPT v2 - Cerebro Expandido)
tam_vocab = len(vocab_unico)
dim_emb = 64        # Antes 16. Le damos 4 veces más capacidad matemática.
tam_contexto = 16   # Antes 5. Ahora podrá leer hasta 16 palabras hacia atrás.
batch_size = 16     # Antes 8. Procesamos más frases al mismo tiempo.

# 4. El Generador de Batches (El escudo de tu RAM)
def obtener_batch():
    # Elegimos 8 posiciones iniciales aleatorias del texto
    indices_aleatorios = [random.randint(0, len(datos_numericos) - tam_contexto - 1) for _ in range(batch_size)]
    
    x_batch, y_batch = [], []
    for i in indices_aleatorios:
        x_batch.append(datos_numericos[i : i + tam_contexto])
        y_batch.append(datos_numericos[i + 1 : i + tam_contexto + 1])
        
    return torch.tensor(x_batch), torch.tensor(y_batch)

# 5. La Arquitectura MicroGPT (Idéntica a la anterior)
class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(tam_vocab, dim_emb)
        self.pos_emb = nn.Embedding(tam_contexto, dim_emb)
        self.atencion_q = nn.Linear(dim_emb, dim_emb, bias=False)
        self.atencion_k = nn.Linear(dim_emb, dim_emb, bias=False)
        self.atencion_v = nn.Linear(dim_emb, dim_emb, bias=False)
        self.red_oculta = nn.Sequential(
            nn.Linear(dim_emb, dim_emb * 4),
            nn.ReLU(),
            nn.Linear(dim_emb * 4, dim_emb)
        )
        self.salida = nn.Linear(dim_emb, tam_vocab)

    def forward(self, x):
        B, T = x.shape
        posiciones = torch.arange(0, T).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(posiciones)
        Q = self.atencion_q(x)
        K = self.atencion_k(x)
        V = self.atencion_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim_emb ** 0.5)
        mascara = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mascara == 0, float('-inf'))
        pesos = F.softmax(scores, dim=-1)
        x_atencion = torch.matmul(pesos, V)
        x = x + self.red_oculta(x_atencion)
        prediccion = self.salida(x)
        return prediccion

# 6. Bucle de Entrenamiento Protegido
modelo = MicroGPT()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)
criterio = nn.CrossEntropyLoss()

print(f"Entrenando en lotes (Batches) de {batch_size} frases...")
# Subimos las épocas a 500 porque ahora no ve todo el texto de golpe, sino en pedacitos
for epoca in range(4000):
    # Generamos un mini-lote al azar (Usa muy poca RAM)
    X_lote, Y_lote = obtener_batch()
    
    optimizador.zero_grad()
    predicciones = modelo(X_lote)
    
    B, T, C = predicciones.shape
    error = criterio(predicciones.view(B * T, C), Y_lote.view(B * T))
    
    error.backward()
    optimizador.step()
    
    if (epoca + 1) % 100 == 0:
        print(f"Época {epoca+1} | Error: {error.item():.4f}")

# 7. Guardar el Modelo
torch.save(modelo.state_dict(), "experto_codigo.pth")
print("\n¡Modelo y Vocabulario guardados de forma segura!")
