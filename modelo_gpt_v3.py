import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_emb, num_heads, tam_contexto, dropout=0.1):
        super().__init__()
        assert dim_emb % num_heads == 0, "dim_emb debe ser divisible por num_heads"
        self.num_heads = num_heads
        self.head_dim = dim_emb // num_heads

        # Proyectamos Q, K y V en una sola multiplicación matricial (más rápido en CPU)
        self.qkv = nn.Linear(dim_emb, dim_emb * 3, bias=False)
        self.proyeccion = nn.Linear(dim_emb, dim_emb)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mascara", torch.tril(torch.ones(tam_contexto, tam_contexto)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Dividimos en cabezas paralelas: (Batch, Cabezas, Tiempo, Dim_Cabeza)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mascara[:T, :T] == 0, float('-inf'))
        pesos = F.softmax(scores, dim=-1)
        pesos = self.dropout(pesos)

        salida = torch.matmul(pesos, v)
        salida = salida.transpose(1, 2).contiguous().view(B, T, C)
        return self.proyeccion(salida)


class BloqueTransformer(nn.Module):
    def __init__(self, dim_emb, num_heads, tam_contexto, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_emb)
        self.atencion = MultiHeadAttention(dim_emb, num_heads, tam_contexto, dropout)
        self.norm2 = nn.LayerNorm(dim_emb)
        self.ffn = nn.Sequential(
            nn.Linear(dim_emb, dim_emb * 4),
            nn.GELU(),
            nn.Linear(dim_emb * 4, dim_emb),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.atencion(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MicroGPTv3(nn.Module):
    def __init__(self, tam_vocab, dim_emb=128, num_heads=4, num_capas=3, tam_contexto=32, dropout=0.1):
        super().__init__()
        self.tam_contexto = tam_contexto
        self.token_emb = nn.Embedding(tam_vocab, dim_emb)
        self.pos_emb = nn.Embedding(tam_contexto, dim_emb)
        self.drop = nn.Dropout(dropout)

        self.bloques = nn.Sequential(*[
            BloqueTransformer(dim_emb, num_heads, tam_contexto, dropout)
            for _ in range(num_capas)
        ])

        self.norm_final = nn.LayerNorm(dim_emb)
        self.salida = nn.Linear(dim_emb, tam_vocab)

    def forward(self, x):
        B, T = x.shape
        posiciones = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.drop(self.token_emb(x) + self.pos_emb(posiciones))
        x = self.bloques(x)
        x = self.norm_final(x)
        return self.salida(x)


def generar_respuesta(modelo, tokens_inicio, idx_a_palabra, idx_fin, max_palabras=40, temperatura=0.7, top_k=5):
    modelo.eval()
    generados = list(tokens_inicio)
    respuesta = []

    with torch.no_grad():
        for _ in range(max_palabras):
            contexto = generados[-modelo.tam_contexto:]
            x_entrada = torch.tensor([contexto], dtype=torch.long)

            logits = modelo(x_entrada)[:, -1, :] / temperatura

            # Filtro Top-K: descarta palabras improbables que rompen la fluidez
            if top_k is not None and top_k > 0:
                k_real = min(top_k, logits.size(-1))
                valores_top, _ = torch.topk(logits, k_real)
                umbral = valores_top[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < umbral, float('-inf'))

            probabilidades = F.softmax(logits, dim=-1)
            idx_ganador = torch.multinomial(probabilidades, num_samples=1).item()

            # Detiene la generación limpiamente al llegar a <fin>
            if idx_ganador == idx_fin:
                break

            generados.append(idx_ganador)
            respuesta.append(idx_a_palabra[idx_ganador])

    return " ".join(respuesta)


# Prueba rápida de arquitectura y consumo de memoria
if __name__ == "__main__":
    cerebro_prueba = MicroGPTv3(tam_vocab=1000, dim_emb=128, num_heads=4, num_capas=3, tam_contexto=32)
    total_params = sum(p.numel() for p in cerebro_prueba.parameters())
    peso_mb = (total_params * 4) / (1024 ** 2)
    print(f"MicroGPT v3 listo | Parámetros: {total_params:,} | Peso en RAM: {peso_mb:.2f} MB")