import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. El Dataset: Una historia corta
# En tu modelo final local, podrías cargar un .txt aquí. Por ahora, usamos un texto base.
texto = """en un lugar de la mancha de cuyo nombre no quiero acordarme vivia un hidalgo con un escudo antiguo y un caballo flaco 
este hidalgo pasaba los dias leyendo libros de caballeria hasta que perdio el juicio y decidio convertirse en caballero andante"""

# 2. Preparar el Vocabulario
palabras = texto.split()
vocab = sorted(list(set(palabras)))
tam_vocab = len(vocab)
palabra_a_idx = {p: i for i, p in enumerate(vocab)}
idx_a_palabra = {i: p for i, p in enumerate(vocab)}

# Convertir el texto entero a números
datos_completos = [palabra_a_idx[p] for p in palabras]

# Hiperparámetros de hardware limitado
dim_emb = 16
tam_contexto = 5 # Cuántas palabras mira hacia atrás para predecir la siguiente

# 3. Construir pares de entrenamiento (X = contexto, Y = siguiente palabra)
X, Y = [], []
for i in range(len(datos_completos) - tam_contexto):
    X.append(datos_completos[i : i + tam_contexto])
    Y.append(datos_completos[i + 1 : i + tam_contexto + 1]) # Y es X desplazado 1 paso al futuro

X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# 4. La Arquitectura MicroGPT 
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

# 5. Inicializar y Entrenar
modelo = MicroGPT()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)
criterio = nn.CrossEntropyLoss()

print(f"Entrenando MicroGPT (Vocabulario: {tam_vocab} palabras)...")
for epoca in range(150):
    optimizador.zero_grad()
    
    # Pasamos todas las frases al mismo tiempo
    predicciones = modelo(X_tensor)
    
    # Reacomodamos los tensores para calcular el error (PyTorch exige esta forma 2D)
    B, T, C = predicciones.shape
    predicciones_2d = predicciones.view(B * T, C)
    objetivos_2d = Y_tensor.view(B * T)
    
    error = criterio(predicciones_2d, objetivos_2d)
    error.backward()
    optimizador.step()

print(f"¡Entrenamiento Finalizado! Error final: {error.item():.4f}\n")


# Guardar los pesos entrenados en tu disco de 60 GB
torch.save(modelo.state_dict(), "micro_gpt_quijote.pth")
print("¡Cerebro guardado exitosamente en micro_gpt_quijote.pth!")

# 6. Generación de Texto con Temperatura (Creatividad)
contexto_inicial = "en un lugar de la"
palabras_generadas = contexto_inicial.split()

# Prueba cambiar este valor entre 0.1 y 2.0
temperatura = 1.0 

print(f"--- TEXTO GENERADO (Temperatura: {temperatura}) ---")
print(contexto_inicial, end=" ", flush=True)

with torch.no_grad():
    for _ in range(20):
        contexto_actual = palabras_generadas[-tam_contexto:]
        x_entrada = torch.tensor([[palabra_a_idx[p] for p in contexto_actual]])
        
        predicciones = modelo(x_entrada)
        
        # 1. Tomamos los valores crudos (logits) de la última predicción
        logits = predicciones[:, -1, :] 
        
        # 2. Aplicamos la matemática de la Temperatura
        logits = logits / temperatura
        
        # 3. Convertimos esos números a porcentajes reales (0.0 a 1.0)
        probabilidades = F.softmax(logits, dim=-1)
        
        # 4. La magia: en lugar de argmax, tiramos un dado "cargado" 
        # con las probabilidades usando multinomial
        idx_ganador = torch.multinomial(probabilidades, num_samples=1).item()
        
        palabra_nueva = idx_a_palabra[idx_ganador]
        
        print(palabra_nueva, end=" ", flush=True)
        palabras_generadas.append(palabra_nueva)

print("\n---------------------------------")
