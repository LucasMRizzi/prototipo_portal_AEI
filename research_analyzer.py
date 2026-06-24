import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import pdfplumber

load_dotenv()

# Inicializa o cliente apontando para os endpoints do NVIDIA NIM
client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

def obter_embedding(texto, tipo_input="passage"):
    """
    Obtém embedding usando um ÚNICO modelo fixo da NVIDIA para manter a consistência.
    tipo_input: "passage" para indexar o PDF, "query" para quando fizer perguntas.
    """
    # Escolha UM modelo e use ele para tudo. 
    # O 'nvidia/nv-embedqa-e5-v5' é excelente, rápido e tem 1024 dimensões.
    MODELO_FIXO = "nvidia/nv-embedqa-e5-v5" 
    
    try:
        resultado = client.embeddings.create(
            model=MODELO_FIXO,
            input=[texto],
            encoding_format="float",
            extra_body={"input_type": tipo_input} # A NVIDIA exige isso para saber se é busca ou texto
        )
        return np.array(resultado.data[0].embedding)
        
    except Exception as e:
        # Se der erro, precisamos parar o código para você ver o real motivo (API Key, créditos, etc)
        print(f"\n[ERRO CRÍTICO] Falha na API da NVIDIA com o modelo {MODELO_FIXO}: {e}")
        raise e

def similaridade_cosseno(a, b):
    """Calcula a similaridade de cosseno entre dois vetores."""
    produto_escalar = np.dot(a, b)
    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return produto_escalar / (norma_a * norma_b)

def extrair_texto_pdf(caminho_pdf):
    """Extrai texto de um arquivo PDF usando pdfplumber."""
    try:
        texto_completo = ""
        with pdfplumber.open(caminho_pdf) as pdf:
            for pagina in pdf.pages:
                texto_pagina = pagina.extract_text()
                if texto_pagina:
                    texto_completo += texto_pagina + "\n"
        return texto_completo
    except Exception as e:
        print(f"Erro ao extrair texto do PDF: {e}")
        return None

def limpar_texto(texto):
    """Limpa e normaliza o texto extraído do PDF."""
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', texto)
    texto = texto.strip()
    return texto

def identificar_secoes(texto):
    """Identifica as seções principais de um artigo acadêmico usando busca global por Regex."""
    secoes = {
        'abstract': '',
        'introducao': '',
        'metodologia': '',
        'resultados': '',
        'conclusao': '',
        'referencias': ''
    }

    # Padroes atualizados para capturar o conteúdo entre os títulos no texto corrido
    padroes = {
        'abstract': r'(?i)\b(?:abstract|resumo)\b',
        'introducao': r'(?i)\b(?:introduction|introdução|1\.?\s*introdução)\b',
        'metodologia': r'(?i)\b(?:methodology|methods|metodologia|2\.?\s*metodologia)\b',
        'resultados': r'(?i)\b(?:results|resultados|3\.?\s*resultados)\b',
        'conclusao': r'(?i)\b(?:conclusion|conclusions|conclusão|4\.?\s*conclusão)\b',
        'referencias': r'(?i)\b(?:references|referências|bibliography)\b'
    }

    # Ordena as seções pela ordem em que aparecem no texto para fatiar corretamente
    posicoes = []
    for secao, padrao in padroes.items():
        match = re.search(padrao, texto)
        if match:
            posicoes.append((secao, match.start(), match.end()))
    
    # Ordena por ordem de aparecimento no PDF
    posicoes.sort(key=lambda x: x[1])

    # Fatia o texto entre uma seção e outra
    for i, (secao_atual, inicio, fim) in enumerate(posicoes):
        fim_secao = len(texto)
        if i + 1 < len(posicoes):
            # O fim da seção atual é o início da próxima seção detectada
            fim_secao = posicoes[i+1][1]
        
        # Extrai o conteúdo pertencente àquela seção
        secoes[secao_atual] = texto[fim:fim_secao].strip()

    return secoes

def dividir_em_chunks(texto, tamanho_chunk=1000, sobreposicao=200):
    """
    Divide o texto em chunks baseando-se no número de caracteres.
    1000 caracteres equivalem a aproximadamente 200-250 tokens,
    garantindo que NUNCA ultrapasse o limite de 512 do modelo.
    """
    if not texto:
        return []

    chunks = []
    inicio = 0
    total_caracteres = len(texto)

    while inicio < total_caracteres:
        # Define o fim planejado do chunk
        fim = inicio + tamanho_chunk
        
        # Se não for o fim do texto, tenta quebrar em um espaço para não cortar palavras ao meio
        if fim < total_caracteres:
            posicao_espaco = texto.rfind(' ', inicio, fim)
            if posicao_espaco != -1 and posicao_espaco > inicio:
                fim = posicao_espaco

        chunk = texto[inicio:fim].strip()
        if chunk:
            chunks.append(chunk)
            
        # Avança considerando a sobreposição
        inicio = fim - sobreposicao
        
        # Garante que o loop saia se o avanço travar
        if fim >= total_caracteres or (fim - inicio) <= 0:
            break

    return chunks

def recuperar_chunks_relevantes(chunks, embeddings_chunks, consulta, k=3):
    """Recupera os k chunks mais relevantes para a consulta."""
    if not chunks or not embeddings_chunks:
        return []

    embedding_consulta = obter_embedding(consulta)
    similaridades = []

    for i, emb_chunk in enumerate(embeddings_chunks):
        sim = similaridade_cosseno(embedding_consulta, emb_chunk)
        similaridades.append((i, sim))

    similaridades.sort(key=lambda x: x[1], reverse=True)
    indices_top = [idx for idx, _ in similaridades[:k]]
    return [chunks[i] for i in indices_top]

def gerar_analise_com_nvidia_nim(contexto, prompt_analise):
    """Gera análise usando um LLM do NVIDIA NIM (Llama 3.3)."""
    try:
        prompt_aumentado = f"""Contexto: {contexto}

{prompt_analise}

Forneça uma resposta estruturada e objetiva baseada apenas no contexto acima."""

        # Chamada corrigida para usar o endpoint da NVIDIA via OpenAI SDK
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt_aumentado  # Bug corrigido: de 'prompt' para 'prompt_aumentado'
                }
            ],
            temperature=0.2
        )

        # Bug corrigido: extração correta do texto no formato do SDK OpenAI
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erro ao gerar análise com NVIDIA NIM: {e}")
        return "Erro na análise"
    
def obter_visao_geral_artigo(chunks):
    """Gera um resumo macro do artigo para orientar as análises específicas."""
    if not chunks:
        return "Não foi possível ler o texto."
        
    # Pega os primeiros chunks (Introdução/Resumo) para contextualizar o todo
    contexto_inicial = "\n\n".join(chunks[:4]) 
    
    prompt = """Analise brevemente este início de artigo acadêmico e extraia:
1. O objetivo principal do trabalho.
2. A proposta/solução central que o autor apresenta (ex: arquitetura de microsserviços, etc.).

Seja ultra-conciso (máximo 4 linhas). Use isso apenas como uma sinopse estruturada."""

    return gerar_analise_com_nvidia_nim(contexto_inicial, prompt)

def avaliar_dimensao(chunks, embeddings_chunks, nome_dimensao, prompt_avaliacao, visao_geral=""):
    """Avalia uma dimensão cruzando os trechos específicos com a visão macro do artigo."""
    chunks_relevantes = recuperar_chunks_relevantes(chunks, embeddings_chunks, prompt_avaliacao, k=4)
    contexto_especifico = "\n\n--- TRECHO DO PDF ---\n\n".join(chunks_relevantes)

    if not contexto_especifico.strip():
        return "Contexto insuficiente para avaliação desta dimensão."

    # Criamos um prompt equilibrado: une o rigor técnico à consciência contextual
    prompt_critico = f"""Você é um avaliador acadêmico justo, técnico e analítico.
Você deve avaliar a dimensão: "{nome_dimensao}" focando nos trechos fornecidos, mas sem ignorar o propósito geral do trabalho.

[VISÃO GERAL DO ARTIGO - CONTEXTO MACRO]
{visao_geral}

[DIRETRIZES DE AVALIAÇÃO]
- Analise criticamente os trechos específicos abaixo.
- ATENÇÃO: Lembre-se de que você está lendo apenas FRAGMENTOS do texto. Não acuse o autor de omitir algo (como referências ou conclusões) a menos que o trecho analisado contradiga diretamente a boa prática da dimensão avaliada.
- Seja construtivo e aponte falhas ou méritos reais baseados na proposta técnica descrita na Visão Geral.

Sua resposta deve conter apenas o texto da análise descritiva, sem introduções ou notas."""

    return gerar_analise_com_nvidia_nim(contexto_especifico, prompt_critico)

def gerar_relatorio_markdown(analises, secoes, metadados={}):
    """Gera um relatório descritivo e dissertativo em formato markdown com as análises."""
    relatorio = "# Relatório de Avaliação Crítica do Artigo Acadêmico\n\n"

    # Metadados
    if metadados:
        relatorio += "## Metadados do Documento\n"
        for chave, valor in metadados.items():
            relatorio += f"- **{chave}**: {valor}\n"
        relatorio += "\n---\n\n"

    # Parecer Descritivo por Dimensão
    relatorio += "## Análise Detalhada por Diretriz Acadêmica\n\n"
    for nome_dimensao, analise_texto in analises.items():
        relatorio += f"### {nome_dimensao}\n"
        relatorio += f"{analise_texto}\n\n"

    # Seções identificadas (opcional)
    if any(secoes.values()):
        relatorio += "---\n\n## Estrutura de Conteúdo Identificada\n\n"
        for nome_secao, conteudo in secoes.items():
            if conteudo:
                relatorio += f"### {nome_secao.title()}\n"
                preview = conteudo[:200] + "..." if len(conteudo) > 200 else conteudo
                relatorio += f"{preview}\n\n"

    return relatorio

def gerar_sugestoes_melhoria(dimensao, justificativa):
    """Gera sugestões específicas de melhoria baseadas na dimensão e justificativa."""
    sugestoes_base = {
        'Claridade da Hipótese/Problema': [
            "Defina claramente o problema de pesquisa nas primeiras páginas",
            "Use perguntas de pesquisa ou hipóteses explícitas e mensuráveis"
        ],
        'Originalidade da Abordagem': [
            "Realize uma revisão de literatura mais abrangente para identificar lacunas",
            "Destaque claramente como sua abordagem difere dos trabalhos existentes"
        ],
        'Rigor Metodológico': [
            "Detalhe melhor os procedimentos metodológicos para replicabilidade",
            "Justifique a escolha dos métodos com referências estabelecidas"
        ],
        'Significância dos Resultados': [
            "Interprete os resultados à luz das perguntas de pesquisa iniciais",
            "Discuta as implicações teóricas e práticas dos achados"
        ],
        'Qualidade da Escrita': [
            "Revise o texto para melhorar clareza e coesão",
            "Verifique consistência na terminologia ao longo do texto"
        ]
    }

    sugestoes = sugestoes_base.get(dimensao, ["Revise esta seção cuidadosamente"])
    return "; ".join(sugestoes[:2])

def analisar_artigo(caminho_pdf, contexto_adicional=""):
    """Função principal para analisar um artigo académico em formato PDF."""
    print("Iniciando análise do artigo académico...")

    # 1. Extrai texto do PDF
    print("Extraindo texto do PDF...")
    texto_bruto = extrair_texto_pdf(caminho_pdf)
    if not texto_bruto:
        return "Erro: Não foi possível extrair texto do PDF."

    # 2. Identifica as seções PRIMEIRO (com o texto ainda estruturado)
    print("Identificando seções do artigo...")
    secoes = identificar_secoes(texto_bruto)

    # 3. Agora sim, limpa e normaliza o texto para o pipeline de embeddings
    print("Limpando e normalizando texto para embeddings...")
    texto_limpo = limpar_texto(texto_bruto)

    # 4. Divide em chunks para processamento
    print("Dividindo texto em chunks para processamento...")
    chunks = dividir_em_chunks(texto_limpo, tamanho_chunk=1200, sobreposicao=250)
    if not chunks:
        return "Erro: Não foi possível processar o texto extraído."

    print("Gerando embeddings para os chunks via NVIDIA NIM...")
    embeddings_chunks = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"Processando chunk {i+1}/{len(chunks)}...")
        embedding = obter_embedding(chunk)
        embeddings_chunks.append(embedding)

    dimensoes = {
        'Claridade da Hipótese/Problema': "Qual é a clareza e bem-definição da hipótese, problema de pesquisa ou objetivo apresentado no artigo?",
        'Originalidade da Abordagem': "Quão original ou inovadora é a abordagem, metodologia ou proposta apresentada em relação ao estado da arte existente?",
        'Rigor Metodológico': "Qual é o rigor, adequação e detalhe da metodologia empregada para abordar o problema de pesquisa?",
        'Significância dos Resultados': "Quão significativos, relevantes e impactantes são os resultados apresentados em relação ao campo de estudo?",
        'Qualidade da Escrita': "Qual é a qualidade geral da escrita, clareza, organização e apresentação do artigo?"
    }

    print("Mapeando visão geral do documento para contextualizar a IA...")
    visao_geral = obter_visao_geral_artigo(chunks)
    print(f"Sinopse gerada: {visao_geral}\n")

    print("Avaliando dimensões do artigo via Llama-3.3 no NIM...")
    analises = {}
    for nome_dimensao, prompt_base in dimensoes.items():
        print(f"Analisando criticamente: {nome_dimensao}...")
        prompt_completo = prompt_base
        if contexto_adicional:
            prompt_completo += f" Considere também este contexto: {contexto_adicional}"

        # Enviando a visão_geral para evitar falsas acusações por falta de contexto
        analise = avaliar_dimensao(chunks, embeddings_chunks, nome_dimensao, prompt_completo, visao_geral=visao_geral)
        analises[nome_dimensao] = analise

    print("Gerando relatório final...")
    metadados = {
        'Arquivo Analisado': os.path.basename(caminho_pdf),
        'Total de Chunks Processados': len(chunks),
        'Seções Identificadas': ', '.join([k for k, v in secoes.items() if v])
    }

    relatorio = gerar_relatorio_markdown(analises, secoes, metadados)
    return relatorio

if __name__ == "__main__":
    print("Agente de análise de artigos académicos configurado para NVIDIA NIM pronto para uso.")
    # Exemplo de execução:
    # resultado = analisar_artigo("seu_artigo.pdf", "Este artigo foca em deep learning.")
    # print(resultado)