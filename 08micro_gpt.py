import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Hiperparámetros ultra ligeros para 4GB RAM
tam_vocab = 50
dim_emb = 16
tam_contexto = 8 # La IA leerá hasta 8 palabras hacia atrás

class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Capa 1: Diccionario de palabras a vectores
        self.token_emb = nn.Embedding(tam_vocab, dim_emb)
        # Capa 2: Positional Encoding (Aprende qué posición ocupa cada palabra)
        self.pos_emb = nn.Embedding(tam_contexto, dim_emb)
        
        # Capa 3: La Auto-Atención que ya construiste
        self.atencion_q = nn.Linear(dim_emb, dim_emb, bias=False)
        self.atencion_k = nn.Linear(dim_emb, dim_emb, bias=False)
        self.atencion_v = nn.Linear(dim_emb, dim_emb, bias=False)
        
        # Capa 4: Red neuronal de salida (para procesar el resultado de la atención)
        self.red_oculta = nn.Sequential(
            nn.Linear(dim_emb, dim_emb * 4),
            nn.ReLU(),
            nn.Linear(dim_emb * 4, dim_emb)
        )
        
        # Capa 5: Proyección final al vocabulario para adivinar la siguiente palabra
        self.salida = nn.Linear(dim_emb, tam_vocab)

    def forward(self, x):
        # x tiene forma (Batch, Longitud_Frase)
        B, T = x.shape
        
        # 1. Sumamos el significado de la palabra + su posición temporal en la frase
        posiciones = torch.arange(0, T).unsqueeze(0) # [0, 1, 2, 3...]
        x = self.token_emb(x) + self.pos_emb(posiciones)
        
        # 2. Mecanismo de Atención
        Q = self.atencion_q(x)
        K = self.atencion_k(x)
        V = self.atencion_v(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim_emb ** 0.5)
        
        # MÁSCARA CASUAL (Causal Masking): Fundamental para que la IA no haga trampa.
        # Le tapamos los ojos para que no pueda ver las palabras del futuro que debe predecir.
        mascara = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mascara == 0, float('-inf'))
        
        pesos = F.softmax(scores, dim=-1)
        x_atencion = torch.matmul(pesos, V)
        
        # 3. Sumamos la atención procesada y lanzamos la predicción al vocabulario
        x = x + self.red_oculta(x_atencion)
        prediccion = self.salida(x)
        
        return prediccion, pesos

print("Construyendo el Micro-GPT...")
modelo = MicroGPT()

# Prueba de laboratorio: Le pasamos una frase ficticia de 4 palabras (índices 1 al 4)
frase_prueba = torch.tensor([[1, 2, 3, 4]]) 
prediccion, matriz_enmascarada = modelo(frase_prueba)

print("\n--- MATRIZ DE ATENCIÓN ENMASCARADA (4x4) ---")
print(torch.round(matriz_enmascarada.squeeze(0) * 1000) / 1000)
print("\nObserva los ceros en la esquina superior derecha.")