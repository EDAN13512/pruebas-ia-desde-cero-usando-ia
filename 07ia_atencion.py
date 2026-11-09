import torch
import torch.nn as nn
import torch.nn.functional as F

class BloqueAtencion(nn.Module):
    def __init__(self, dim_emb):
        super().__init__()
        self.dim_emb = dim_emb
        
        # Las 3 transformaciones lineales (El cerebro matemático del Transformer)
        # bias=False es estándar aquí para ahorrar cálculos en la CPU
        self.W_q = nn.Linear(dim_emb, dim_emb, bias=False) # Query
        self.W_k = nn.Linear(dim_emb, dim_emb, bias=False) # Key
        self.W_v = nn.Linear(dim_emb, dim_emb, bias=False) # Value

    def forward(self, x):
        # 1. Cada palabra genera su propia Query, Key y Value
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Calculamos los Scores de Atención: Multiplicamos Q por K transpuesto.
        # Esto hace que TODAS las palabras se comparen con TODAS las palabras al mismo tiempo.
        # Lo dividimos por la raíz cuadrada de la dimensión para estabilizar los números.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_emb ** 0.5)
        
        # 3. Softmax convierte esos scores crudos en porcentajes (de 0.0 a 1.0)
        # Asegurando que la atención de cada palabra sume siempre 100%
        pesos_atencion = F.softmax(scores, dim=-1)
        
        # 4. Multiplicamos esos porcentajes por los Values para obtener el contexto final
        salida = torch.matmul(pesos_atencion, V)
        
        return salida, pesos_atencion

# -----------------------------------------
# PRUEBA DE LABORATORIO (La magia en acción)
# -----------------------------------------

# Simulamos una frase de 4 palabras: "la", "pizza", "me", "gusta"
# Le damos vectores aleatorios simulando que ya pasaron por una capa de Embedding
torch.manual_seed(42) # Fijamos la semilla para que te den los mismos números que a mí
dim_embedding = 8 
frase_simulada = torch.rand(1, 4, dim_embedding) # (Batch=1, Palabras=4, Dimension=8)

# Instanciamos nuestra red de atención
atencion = BloqueAtencion(dim_emb=dim_embedding)

# Pasamos la frase por el bloque
with torch.no_grad():
    contexto_final, matriz_atencion = atencion(frase_simulada)

print("--- MATRIZ DE ATENCIÓN (4x4) ---")
# Redondeamos a 4 decimales para que sea legible en la terminal
print(torch.round(matriz_atencion.squeeze(0) * 10000) / 10000)

print("\n¿Qué significa esta matriz?")
print("Fila 1: A qué palabras le presta atención 'la'")
print("Fila 2: A qué palabras le presta atención 'pizza'")
print("Fila 3: A qué palabras le presta atención 'me'")
print("Fila 4: A qué palabras le presta atención 'gusta'")