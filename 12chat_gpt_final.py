import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# 1. Cargar el Vocabulario desde el JSON
print("Cargando conocimientos...")
with open('vocabulario.json', 'r', encoding='utf-8') as f:
    datos_vocab = json.load(f)

palabra_a_idx = datos_vocab['p_a_i']
# OJO AQUÍ: JSON guarda las claves numéricas como texto ("0", "1"). 
# Las volvemos a convertir a números enteros (int) para que PyTorch no explote.
idx_a_palabra = {int(k): v for k, v in datos_vocab['i_a_p'].items()}

# 2. Hiperparámetros (Deben coincidir con los del entrenamiento)
tam_vocab = len(palabra_a_idx)
dim_emb = 64
tam_contexto = 16

# 3. La Arquitectura MicroGPT (Pegamos la clase intacta)
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

# 4. Cargar el Cerebro Entrenado
modelo = MicroGPT()
modelo.load_state_dict(torch.load("memoria.pth", weights_only=True))
modelo.eval() # Modo generación

# 5. Generación de Texto Interactiva (Chat)
temperatura = 0.8

print(f"\n--- IA LISTA (Vocabulario: {tam_vocab} palabras) ---")
print("Escribe un mensaje para tu IA (o 'salir' para terminar).")
print("Intenta usar palabras que ella conozca, como: hola, quiero, presentar, nombre...")

while True:
    mensaje = input("\nTú: ").lower()
    if mensaje == 'salir': break
    
    # 1. PLANTILLA DE PROMPT: Ensamblamos la estructura a escondidas
    prompt_oculto = f"edan: {mensaje} ia:"
    
    palabras_prompt = [p for p in prompt_oculto.split() if p in palabra_a_idx]
    
    if not palabras_prompt:
        print("IA: (Silencio... no conozco ninguna de esas palabras)")
        continue
        
    palabras_generadas = palabras_prompt.copy()
    
    print("IA:", end=" ", flush=True)
    
    with torch.no_grad():
        # Subimos el límite a 20 palabras para que pueda dar respuestas completas
        for _ in range(20):
            contexto_actual = palabras_generadas[-tam_contexto:] 
            
            x_entrada = torch.tensor([[palabra_a_idx[p] for p in contexto_actual]])
            
            predicciones = modelo(x_entrada)
            logits = predicciones[:, -1, :] / temperatura
            probabilidades = F.softmax(logits, dim=-1)
            
            idx_ganador = torch.multinomial(probabilidades, num_samples=1).item()
            palabra_nueva = idx_a_palabra[idx_ganador]
            
            # 2. TOKEN DE PARADA: Si la IA intenta hablar por ti, le cortamos el micrófono
            if palabra_nueva == "edan:":
                break
                
            print(palabra_nueva, end=" ", flush=True)
            palabras_generadas.append(palabra_nueva)
    print()
