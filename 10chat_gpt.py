import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Copiamos los hiperparámetros y vocabularios (En el futuro, esto se guarda en un JSON)
texto = """en un lugar de la mancha de cuyo nombre no quiero acordarme vivia un hidalgo con un escudo antiguo y un caballo flaco 
este hidalgo pasaba los dias leyendo libros de caballeria hasta que perdio el juicio y decidio convertirse en caballero andante"""
palabras = texto.split()
vocab = sorted(list(set(palabras)))
tam_vocab = len(vocab)
palabra_a_idx = {p: i for i, p in enumerate(vocab)}
idx_a_palabra = {i: p for i, p in enumerate(vocab)}

dim_emb = 16
tam_contexto = 5

# 2. Pegamos la estructura exacta de la clase MicroGPT aquí
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

# 3. La magia: Cargar el cerebro sin entrenar
modelo = MicroGPT()
modelo.load_state_dict(torch.load("micro_gpt_quijote.pth", weights_only=True))
modelo.eval() # Le avisa a PyTorch que el modelo está en modo lectura/generación

# 4. Tu código de generación con Temperatura va aquí abajo
contexto_inicial = "en un lugar de la"
palabras_generadas = contexto_inicial.split()
temperatura = 1.0

print("\nCerebro cargado. Generando texto al instante...")
print(contexto_inicial, end=" ", flush=True)

with torch.no_grad():
    for _ in range(38):
        contexto_actual = palabras_generadas[-tam_contexto:]
        x_entrada = torch.tensor([[palabra_a_idx[p] for p in contexto_actual]])
        predicciones = modelo(x_entrada)
        logits = predicciones[:, -1, :] / temperatura
        probabilidades = F.softmax(logits, dim=-1)
        idx_ganador = torch.multinomial(probabilidades, num_samples=1).item()
        palabra_nueva = idx_a_palabra[idx_ganador]
        print(palabra_nueva, end=" ", flush=True)
        palabras_generadas.append(palabra_nueva)
print("\n")
