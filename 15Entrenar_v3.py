import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import re
from modelo_gpt_v3 import MicroGPTv3

# ==========================================
# CONFIGURACIÓN DEL EXPERTO A ENTRENAR
# ==========================================
ARCHIVO_TXT = "codigo.txt"
ARCHIVO_JSON = "vocab_codigo.json"
ARCHIVO_PTH = "experto_codigo.pth"

CONFIG = {
    "dim_emb": 128,
    "num_heads": 4,
    "num_capas": 3,
    "tam_contexto": 32,
    "dropout": 0.1
}
BATCH_SIZE = 16
EPOCAS = 1500

# 1. Cargar y limpiar conservando ':' y '<>' para el token <fin>
with open(ARCHIVO_TXT, "r", encoding="utf-8") as f:
    texto = f.read().lower()

texto = re.sub(r'[^a-záéíóúñ\s:<>]', '', texto)
palabras = texto.split()

# Aseguramos que <unk> y <fin> siempre existan en el diccionario
vocab_unico = sorted(list(set(palabras + ["<unk>", "<fin>"])))
palabra_a_idx = {p: i for i, p in enumerate(vocab_unico)}
idx_a_palabra = {i: p for i, p in enumerate(vocab_unico)}

with open(ARCHIVO_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "p_a_i": palabra_a_idx,
        "i_a_p": idx_a_palabra,
        "config": CONFIG
    }, f, ensure_ascii=False)

datos_numericos = [palabra_a_idx[p] for p in palabras]
tam_ctx = CONFIG["tam_contexto"]

def obtener_batch():
    max_inicio = len(datos_numericos) - tam_ctx - 1
    indices = [random.randint(0, max_inicio) for _ in range(BATCH_SIZE)]
    x_b = [datos_numericos[i : i + tam_ctx] for i in indices]
    y_b = [datos_numericos[i + 1 : i + tam_ctx + 1] for i in indices]
    return torch.tensor(x_b), torch.tensor(y_b)

# 2. Instanciar MicroGPT v3
modelo = MicroGPTv3(tam_vocab=len(vocab_unico), **CONFIG)
optimizador = optim.AdamW(modelo.parameters(), lr=0.001)
criterio = nn.CrossEntropyLoss()

print(f"Entrenando {ARCHIVO_PTH} | Vocabulario: {len(vocab_unico)} tokens...")
modelo.train()

for epoca in range(EPOCAS):
    X_lote, Y_lote = obtener_batch()
    optimizador.zero_grad()

    predicciones = modelo(X_lote)
    B, T, C = predicciones.shape
    error = criterio(predicciones.view(B * T, C), Y_lote.view(B * T))

    error.backward()
    optimizador.step()

    if (epoca + 1) % 300 == 0:
        print(f"Época {epoca+1}/{EPOCAS} | Error: {error.item():.4f}")

torch.save(modelo.state_dict(), ARCHIVO_PTH)
print(f"¡Cerebro guardado en {ARCHIVO_PTH}!")
