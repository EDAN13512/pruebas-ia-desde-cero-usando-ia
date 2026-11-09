import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. Definir los "Expertos" disponibles
categorias = {
    0: "Experto_Charla",
    1: "Experto_Mecanica",
    2: "Experto_Codigo"
}

# 2. Dataset de Entrenamiento (Balanceado para evitar sesgos)
datos = [
    # 0 = Charla General
    ("hola como estas", 0),
    ("buenos dias", 0),
    ("hola quiero conversar", 0),
    ("tengo ganas de charlar", 0),
    ("hola tengo una pregunta", 0),
    
    # 1 = Mecánica / Autos
    ("tengo la rueda pinchada", 1),
    ("hola tengo un problema en el motor", 1),
    ("como cambiar el aceite", 1),
    ("mi auto hace un ruido", 1),
    ("tengo los frenos gastados", 1),
    
    # 2 = Código / Programación
    ("tengo un error en python", 2),
    ("hola como hacer un bucle for", 2),
    ("mi codigo no funciona", 2),
    ("tengo un problema con el codigo", 2),
    ("necesito programar una ia", 2)
]

# 3. Construir el Vocabulario
vocabulario = {"<UNK>": 0}
indice = 1
for frase, _ in datos:
    for palabra in frase.lower().split():
        if palabra not in vocabulario:
            vocabulario[palabra] = indice
            indice += 1

def frase_a_tensor(frase):
    indices = [vocabulario.get(palabra, 0) for palabra in frase.lower().split()]
    if not indices: indices = [0]
    return torch.tensor(indices).unsqueeze(0) # (1, longitud)

# 4. Arquitectura del Enrutador (Ultra ligera)
class EnrutadorMoE(nn.Module):
    def __init__(self, tam_vocab, dim_emb, dim_oculta, num_clases):
        super().__init__()
        self.embedding = nn.Embedding(tam_vocab, dim_emb)
        self.lstm = nn.LSTM(dim_emb, dim_oculta, batch_first=True)
        # La salida ahora es igual al número de expertos (3)
        self.salida = nn.Linear(dim_oculta, num_clases)

    def forward(self, x):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        resumen = hn[-1] 
        # Ya no usamos Sigmoide aquí, CrossEntropy maneja los números crudos
        logits = self.salida(resumen)
        return logits

# Instanciamos el Enrutador con 3 clases de salida
enrutador = EnrutadorMoE(len(vocabulario), dim_emb=8, dim_oculta=16, num_clases=3)

# 5. Entrenamiento
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(enrutador.parameters(), lr=0.01)

print("Entrenando el Enrutador LSTM...")
for epoca in range(300):
    error_total = 0
    for frase, etiqueta in datos:
        optimizador.zero_grad()
        
        tensor_frase = frase_a_tensor(frase)
        tensor_etiqueta = torch.tensor([etiqueta]) 
        
        prediccion = enrutador(tensor_frase)
        error = criterio(prediccion, tensor_etiqueta)
        
        error.backward()
        optimizador.step()
        error_total += error.item()

print("¡Enrutador listo!\n")

# 6. Modo Interactivo (Prueba de Enrutamiento)
print("-" * 40)
print("SISTEMA MoE: Escribe tu problema (o 'salir')")
print("-" * 40)

enrutador.eval()
while True:
    prompt = input("\nTú: ")
    if prompt.lower() == 'salir': break
    
    tensor_prueba = frase_a_tensor(prompt)
    
    with torch.no_grad():
        logits = enrutador(tensor_prueba)
        # Convertimos los números crudos en porcentajes que sumen 100%
        probabilidades = F.softmax(logits, dim=-1).squeeze(0)
        
    # Buscamos el experto ganador
    idx_ganador = torch.argmax(probabilidades).item()
    experto_elegido = categorias[idx_ganador]
    
    print("\n--- ANÁLISIS INTERNO DEL ENRUTADOR ---")
    for i in range(3):
        print(f"{categorias[i]}: {probabilidades[i]*100:.1f}%")
        
    print(f"\n[ACCIÓN SISTEMA] -> Cargando {experto_elegido}.pth en RAM...")
    print(f"[ACCIÓN SISTEMA] -> Generando respuesta con {experto_elegido}...")
