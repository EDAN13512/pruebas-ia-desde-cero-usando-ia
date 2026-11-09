import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import re

# ==========================================
# 1. EL ENRUTADOR (El Cerebro Central LSTM)
# ==========================================
categorias = {0: "Charla", 1: "Mecanica", 2: "Codigo"}

# Aquí mapeamos a los archivos que acabas de crear
archivos_expertos = {
    0: ("vocab_charla.json", "experto_charla.pth"),
    1: ("vocab_mecanica.json", "experto_mecanica.pth"),
    2: ("vocab_codigo.json", "experto_codigo.pth")
}

datos_router = [
    ("hola como estas", 0), ("buenos dias", 0), ("hola quiero conversar", 0),
    ("tengo ganas de charlar", 0), ("hola tengo una pregunta", 0),
    ("tengo la rueda pinchada", 1), ("hola tengo un problema en el motor", 1),
    ("como cambiar el aceite", 1), ("mi auto hace un ruido", 1), ("tengo los frenos gastados", 1),
    ("tengo un error en python", 2), ("hola como hacer un bucle for", 2),
    ("mi codigo no funciona", 2), ("tengo un problema con el codigo", 2), ("necesito programar una ia", 2)
]

vocab_router = {"<UNK>": 0}
idx = 1
for frase, _ in datos_router:
    for palabra in frase.split():
        if palabra not in vocab_router:
            vocab_router[palabra] = idx
            idx += 1

class EnrutadorMoE(nn.Module):
    def __init__(self, tam_vocab, dim_emb, dim_oculta, num_clases):
        super().__init__()
        self.embedding = nn.Embedding(tam_vocab, dim_emb)
        self.lstm = nn.LSTM(dim_emb, dim_oculta, batch_first=True)
        self.salida = nn.Linear(dim_oculta, num_clases)

    def forward(self, x):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        return self.salida(hn[-1])

# Entrenamos el enrutador velozmente (tarda 1 segundo)
print("Iniciando Sistema MoE (Mezcla de Expertos)...")
print("Entrenando Enrutador central...")
enrutador = EnrutadorMoE(len(vocab_router), dim_emb=8, dim_oculta=16, num_clases=3)
criterio_r = nn.CrossEntropyLoss()
optimizador_r = optim.Adam(enrutador.parameters(), lr=0.01)

for epoca in range(300):
    for frase, etiqueta in datos_router:
        optimizador_r.zero_grad()
        indices = [vocab_router.get(p, 0) for p in frase.split()]
        tensor_x = torch.tensor(indices).unsqueeze(0)
        tensor_y = torch.tensor([etiqueta])
        error = criterio_r(enrutador(tensor_x), tensor_y)
        error.backward()
        optimizador_r.step()

enrutador.eval()

# ==========================================
# 2. LOS EXPERTOS (La Arquitectura MicroGPT)
# ==========================================
dim_emb_gpt = 64
tam_contexto_gpt = 16

class MicroGPT(nn.Module):
    def __init__(self, tam_vocab_dinamico):
        super().__init__()
        self.token_emb = nn.Embedding(tam_vocab_dinamico, dim_emb_gpt)
        self.pos_emb = nn.Embedding(tam_contexto_gpt, dim_emb_gpt)
        self.atencion_q = nn.Linear(dim_emb_gpt, dim_emb_gpt, bias=False)
        self.atencion_k = nn.Linear(dim_emb_gpt, dim_emb_gpt, bias=False)
        self.atencion_v = nn.Linear(dim_emb_gpt, dim_emb_gpt, bias=False)
        self.red_oculta = nn.Sequential(
            nn.Linear(dim_emb_gpt, dim_emb_gpt * 4),
            nn.ReLU(),
            nn.Linear(dim_emb_gpt * 4, dim_emb_gpt)
        )
        self.salida = nn.Linear(dim_emb_gpt, tam_vocab_dinamico)

    def forward(self, x):
        B, T = x.shape
        posiciones = torch.arange(0, T).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(posiciones)
        Q = self.atencion_q(x)
        K = self.atencion_k(x)
        V = self.atencion_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim_emb_gpt ** 0.5)
        mascara = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mascara == 0, float('-inf'))
        pesos = F.softmax(scores, dim=-1)
        x_atencion = torch.matmul(pesos, V)
        x = x + self.red_oculta(x_atencion)
        return self.salida(x)

# EL ESCUDO DE TU RAM: Esta función carga el experto, lo usa y lo destruye
def invocar_experto(categoria_idx, mensaje_usuario):
    archivo_json, archivo_pth = archivos_expertos[categoria_idx]
    
    # 1. Cargar Vocabulario del experto
    with open(archivo_json, 'r', encoding='utf-8') as f:
        datos_vocab = json.load(f)
    palabra_a_idx = datos_vocab['p_a_i']
    idx_a_palabra = {int(k): v for k, v in datos_vocab['i_a_p'].items()}
    
    # 2. Instanciar y Cargar el Cerebro (Se aloja temporalmente en la RAM)
    experto = MicroGPT(len(palabra_a_idx))
    experto.load_state_dict(torch.load(archivo_pth, weights_only=True))
    experto.eval()
    
    # 3. La Plantilla Oculta (Ensamblamos el formato sin que el usuario lo vea)
    prompt_oculto = f"edan: {mensaje_usuario} ia:"
    palabras_generadas = [p for p in prompt_oculto.split() if p in palabra_a_idx]
    
    if not palabras_generadas:
        return "(Silencio... este experto no conoce estas palabras)"
        
    respuesta_final = []
    
    with torch.no_grad():
        for _ in range(20):
            contexto_actual = palabras_generadas[-tam_contexto_gpt:]
            x_entrada = torch.tensor([[palabra_a_idx[p] for p in contexto_actual]])
            
            predicciones = experto(x_entrada)
            logits = predicciones[:, -1, :] / 1.0 # Temperatura 1.0
            probabilidades = F.softmax(logits, dim=-1)
            
            idx_ganador = torch.multinomial(probabilidades, num_samples=1).item()
            palabra_nueva = idx_a_palabra[idx_ganador]
            
            # EL TOKEN DE PARADA
            if palabra_nueva == "edan:":
                break
                
            respuesta_final.append(palabra_nueva)
            palabras_generadas.append(palabra_nueva)
            
    # Al retornar el texto, Python elimina la variable "experto" liberando tu RAM instantáneamente.
    return " ".join(respuesta_final)

# ==========================================
# 3. LA INTERFAZ FINAL
# ==========================================
print("\n" + "="*50)
print(" SISTEMA MoE LISTO (Arquitectura de 3 Cerebros)")
print(" Memoria RAM Protegida (Carga Dinámica Activa)")
print("="*50)

while True:
    prompt = input("\nTú: ").lower()
    if prompt == 'salir': break
    
    # Limpiamos todo menos las letras para el enrutador
    prompt_limpio = re.sub(r'[^a-záéíóúñ\s]', '', prompt)
    
    # 1. Enrutar (Decidir a quién llamar)
    indices = [vocab_router.get(p, 0) for p in prompt_limpio.split()]
    if not indices: indices = [0]
    tensor_x = torch.tensor(indices).unsqueeze(0)
    
    with torch.no_grad():
        logits = enrutador(tensor_x)
        idx_ganador = torch.argmax(logits, dim=-1).item()
        
    nombre_experto = categorias[idx_ganador]
    print(f"\033[90m[Enrutador detecta intención: {nombre_experto}]\033[0m")
    print(f"\033[90m[Cargando Experto_{nombre_experto} en memoria...]\033[0m")
    
    # 2. Generar y Liberar Memoria
    respuesta = invocar_experto(idx_ganador, prompt_limpio)
    
    print(f"\nIA ({nombre_experto}): {respuesta}")
