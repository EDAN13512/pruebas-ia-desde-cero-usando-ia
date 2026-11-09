import torch
import torch.nn as nn
import torch.optim as optim

# 1. El Dataset (Preguntas y sus respuestas generativas)
datos_crudos = [
    {"prompt": "hola", "respuesta": "hola soy una ia"},
    {"prompt": "quien eres", "respuesta": "soy una inteligencia artificial"},
    {"prompt": "que haces", "respuesta": "aprendo a procesar lenguaje"},
    {"prompt": "eres una maquina", "respuesta": "si funciono con codigo"},
    {"prompt": "te gusta la pizza", "respuesta": "no no puedo comer"}
]

# 2. Construcción del Vocabulario con Tokens Especiales
# <PAD>: relleno, <SOS>: inicio de frase, <EOS>: fin de frase, <UNK>: desconocido
vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
idx_a_palabra = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
idx = 4

for item in datos_crudos:
    for frase in [item["prompt"], item["respuesta"]]:
        for palabra in frase.lower().split():
            if palabra not in vocab:
                vocab[palabra] = idx
                idx_a_palabra[idx] = palabra
                idx += 1

def frase_a_tensor(frase, agregar_sos=False, agregar_eos=False):
    tokens = [vocab.get(p, vocab["<UNK>"]) for p in frase.lower().split()]
    if agregar_sos: tokens = [vocab["<SOS>"]] + tokens
    if agregar_eos: tokens = tokens + [vocab["<EOS>"]]
    return torch.tensor(tokens).unsqueeze(1) # Formato (Longitud, Batch=1)

# 3. Arquitectura Encoder-Decoder (Seq2Seq)
class Encoder(nn.Module):
    def __init__(self, tam_vocab, dim_emb, dim_oculta):
        super().__init__()
        self.emb = nn.Embedding(tam_vocab, dim_emb)
        self.lstm = nn.LSTM(dim_emb, dim_oculta)
        
    def forward(self, x):
        # El Encoder no genera palabras, solo devuelve el "estado mental" final (hn, cn)
        _, (hn, cn) = self.lstm(self.emb(x))
        return hn, cn 

class Decoder(nn.Module):
    def __init__(self, tam_vocab, dim_emb, dim_oculta):
        super().__init__()
        self.emb = nn.Embedding(tam_vocab, dim_emb)
        self.lstm = nn.LSTM(dim_emb, dim_oculta)
        self.salida = nn.Linear(dim_oculta, tam_vocab)
        
    def forward(self, x, hn, cn):
        # Procesa una palabra a la vez, usando el estado mental anterior
        out, (hn, cn) = self.lstm(self.emb(x), (hn, cn))
        prediccion = self.salida(out.squeeze(0))
        return prediccion, hn, cn

dim_emb = 16
dim_oculta = 32
tam_vocab = len(vocab)

encoder = Encoder(tam_vocab, dim_emb, dim_oculta)
decoder = Decoder(tam_vocab, dim_emb, dim_oculta)

# 4. Entrenamiento
criterio = nn.CrossEntropyLoss()
# Optimizamos ambos cerebros al mismo tiempo
optimizador = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)

print(f"Entrenando Seq2Seq (Vocabulario: {tam_vocab} palabras)...")
for epoca in range(250):
    error_total = 0
    for item in datos_crudos:
        optimizador.zero_grad()
        
        # Leemos la pregunta
        x_tensor = frase_a_tensor(item["prompt"])
        y_tensor = frase_a_tensor(item["respuesta"], agregar_eos=True)
        
        # El Encoder comprime la pregunta
        hn, cn = encoder(x_tensor)
        
        # El Decoder arranca recibiendo la señal de inicio <SOS>
        entrada_dec = torch.tensor([[vocab["<SOS>"]]])
        error_frase = 0
        
        # Entrenamos palabra por palabra (Teacher Forcing)
        for i in range(len(y_tensor)):
            prediccion, hn, cn = decoder(entrada_dec, hn, cn)
            objetivo = y_tensor[i]
            error_frase += criterio(prediccion, objetivo)
            # En el entrenamiento, le pasamos la palabra correcta como siguiente paso
            entrada_dec = objetivo.unsqueeze(1)
            
        error_frase.backward()
        optimizador.step()
        error_total += error_frase.item()

print(f"¡Entrenamiento completado! Error final: {error_total/len(datos_crudos):.4f}\n")

# 5. Chat Interactivo (Generación real)
print("-" * 30)
print("IA Generativa lista. (Escribe 'salir' para terminar)")
print("-" * 30)

while True:
    prompt = input("Tú: ")
    if prompt.lower() == 'salir': break
    
    with torch.no_grad():
        x_tensor = frase_a_tensor(prompt)
        hn, cn = encoder(x_tensor) # Comprimir prompt
        
        entrada_dec = torch.tensor([[vocab["<SOS>"]]]) # Arrancar motor de escritura
        respuesta_generada = []
        
        for _ in range(15): # Límite de 15 palabras de respuesta
            prediccion, hn, cn = decoder(entrada_dec, hn, cn)
            
            # Buscar la palabra con mayor probabilidad matemática
            palabra_idx = prediccion.argmax(1).item()
            
            # Si escupe el token de fin, cortamos la generación
            if palabra_idx == vocab["<EOS>"]: 
                break
                
            respuesta_generada.append(idx_a_palabra[palabra_idx])
            
            # Retroalimentamos el decoder con su propia palabra inventada
            entrada_dec = torch.tensor([[palabra_idx]])
            
    print("IA:", " ".join(respuesta_generada))