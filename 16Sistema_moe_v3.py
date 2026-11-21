import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import re
import gc
from modelo_gpt_v3 import MicroGPTv3, generar_respuesta

# ==========================================
# 1. EL ENRUTADOR LSTM AMPLIADO
# ==========================================
categorias = {0: "Charla", 1: "Mecanica", 2: "Codigo"}

archivos_expertos = {
    0: ("vocab_charla.json", "experto_charla.pth"),
    1: ("vocab_mecanica.json", "experto_mecanica.pth"),
    2: ("vocab_codigo.json", "experto_codigo.pth")
}

datos_router = [
    # 0 = Charla
    ("hola como estas", 0), ("buenos dias", 0), ("hola quiero conversar", 0),
    ("tengo ganas de charlar", 0), ("hola tengo una pregunta", 0), ("quien sos vos", 0),
    ("estoy aburrido", 0), ("te gusta la musica", 0), ("estoy cansado", 0),
    ("gracias por la ayuda", 0), ("nos vemos", 0), ("como estuvo tu dia", 0),

    # 1 = Mecánica
    ("se me pincho una rueda", 1), ("tengo la rueda pinchada", 1), ("cambiar cubierta", 1),
    ("hola tengo un problema en el motor", 1), ("el auto no arranca", 1), ("falla la bateria", 1),
    ("como cambiar el aceite", 1), ("mi auto hace un ruido raro", 1), ("tengo los frenos gastados", 1),
    ("el motor recalienta", 1), ("sale humo del capot", 1), ("presion de las cubiertas", 1),

    # 2 = Código
    ("tengo un error en python", 2), ("que es un bucle for", 2), ("como hacer un bucle for", 2),
    ("mi codigo no funciona", 2), ("tengo un problema con el codigo", 2), ("necesito programar una ia", 2),
    ("error de sintaxis", 2), ("como ahorro memoria ram", 2), ("que es una lista en python", 2),
    ("como leo un archivo txt", 2), ("usar pytorch y tensores", 2), ("imprimir variables con print", 2)
]

vocab_router = {"<unk>": 0}
for frase, _ in datos_router:
    for palabra in frase.split():
        if palabra not in vocab_router:
            vocab_router[palabra] = len(vocab_router)

class EnrutadorMoE(nn.Module):
    def __init__(self, tam_vocab, dim_emb, dim_oculta, num_clases):
        super().__init__()
        self.embedding = nn.Embedding(tam_vocab, dim_emb)
        self.lstm = nn.LSTM(dim_emb, dim_oculta, batch_first=True)
        self.salida = nn.Linear(dim_oculta, num_clases)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return self.salida(hn[-1])

print("Entrenando Enrutador central...")
enrutador = EnrutadorMoE(len(vocab_router), dim_emb=16, dim_oculta=24, num_clases=3)
criterio_r = nn.CrossEntropyLoss()
optimizador_r = optim.Adam(enrutador.parameters(), lr=0.01)

for _ in range(300):
    for frase, etiqueta in datos_router:
        optimizador_r.zero_grad()
        indices = [vocab_router.get(p, 0) for p in frase.split()]
        error = criterio_r(enrutador(torch.tensor([indices])), torch.tensor([etiqueta]))
        error.backward()
        optimizador_r.step()

enrutador.eval()

# ==========================================
# 2. CARGA DINÁMICA DE EXPERTOS v3
# ==========================================
def invocar_experto(categoria_idx, mensaje_limpio):
    archivo_json, archivo_pth = archivos_expertos[categoria_idx]

    with open(archivo_json, "r", encoding="utf-8") as f:
        datos_vocab = json.load(f)

    palabra_a_idx = datos_vocab["p_a_i"]
    idx_a_palabra = {int(k): v for k, v in datos_vocab["i_a_p"].items()}
    config = datos_vocab["config"]

    # Instanciamos MicroGPT v3 con su configuración exacta guardada
    experto = MicroGPTv3(tam_vocab=len(palabra_a_idx), **config)
    experto.load_state_dict(torch.load(archivo_pth, weights_only=True))

    # Mapeamos palabras desconocidas a <unk> para conservar siempre el contexto estructural
    prompt_oculto = f"edan: {mensaje_limpio} ia:"
    idx_unk = palabra_a_idx["<unk>"]
    idx_fin = palabra_a_idx["<fin>"]
    tokens_inicio = [palabra_a_idx.get(p, idx_unk) for p in prompt_oculto.split()]

    respuesta = generar_respuesta(
        modelo=experto,
        tokens_inicio=tokens_inicio,
        idx_a_palabra=idx_a_palabra,
        idx_fin=idx_fin,
        max_palabras=35,
        temperatura=0.7,
        top_k=5
    )

    # Corte de seguridad adicional si intentara abrir otro turno
    if "edan:" in respuesta:
        respuesta = respuesta.split("edan:")[0].strip()

    # Liberamos explicitamente los 3.26 MB de RAM antes de volver
    del experto
    gc.collect()

    return respuesta

# ==========================================
# 3. INTERFAZ CONVERSACIONAL
# ==========================================
print("\n" + "=" * 52)
print(" SISTEMA MoE v3 (Multi-Head Attention + Top-K + <fin>)")
print("=" * 52)

while True:
    prompt = input("\nTú: ").lower().strip()
    if prompt == "salir":
        break
    if not prompt:
        continue

    prompt_limpio = re.sub(r'[^a-záéíóúñ\s]', '', prompt)
    indices = [vocab_router.get(p, 0) for p in prompt_limpio.split()] or [0]

    with torch.no_grad():
        logits = enrutador(torch.tensor([indices]))
        probs = F.softmax(logits, dim=-1).squeeze(0)
        idx_ganador = torch.argmax(probs).item()
        confianza = probs[idx_ganador].item() * 100

    nombre_experto = categorias[idx_ganador]
    print(f"\033[90m[Router -> {nombre_experto} ({confianza:.1f}%)]\033[0m")

    respuesta = invocar_experto(idx_ganador, prompt_limpio)
    print(f"IA ({nombre_experto}): {respuesta}")