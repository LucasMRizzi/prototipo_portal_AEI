# 📑 Analisador IA de Artigos Acadêmicos (NVIDIA NIM + FastAPI)

Este projeto é um protótipo de uma funcionalidade desenvolvida para o site do Portal do Ambiente de Empreendedorismo e Inovação da Unesp de Rio Claro, o projeto foi desenvolvido como parte de um PAEG realizado no primeiro semestre de 2026.
O projeto consiste em uma aplicação web completa (API + Interface Gráfica) desenvolvida para realizar análises críticas e profundas de artigos acadêmicos em formato PDF. A inteligência do sistema utiliza os modelos de última geração da **NVIDIA NIM** (endpoints compatíveis com o SDK da OpenAI) para processar embeddings textuais de forma segura e gerar relatórios dissertativos sem vieses de fragmentação (RAG).

A interface utiliza **WebSockets** para exibir o progresso do processamento em tempo real (etapa por etapa), garantindo transparência enquanto os vetores e as análises são gerados.

---

## 🚀 Funcionalidades

* **Upload de PDF:** Extração de texto limpa e normalizada utilizando `pdfplumber`.
* **Mapeamento de Seções:** Identificação automatizada da estrutura do artigo (Abstract, Introdução, Metodologia, etc.) via Regex posicional antes da fragmentação do texto.
* **Pipeline de RAG Otimizado:** Divisão de texto baseada em caracteres com margem de segurança para evitar estouro de limite de tokens da API da NVIDIA.
* **Consciência de Contexto Global:** Geração prévia de uma sinopse macro do artigo para evitar que a IA faça críticas injustas a fragmentos isolados.
* **Interface em Tempo Real:** Comunicação via WebSocket que atualiza o usuário sobre cada etapa do processamento (ex: geração de chunk 1/18).
* **Relatório Dissertativo:** Parecer crítico baseado nas diretrizes de bancas avaliadoras, focado 100% no texto (sem notas ou tabelas repetitivas).

---

## 🛠️ Tecnologias Utilizadas

* **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12+)
* **Servidor ASGI:** [Uvicorn](https://www.uvicorn.org/)
* **Modelos de IA (NVIDIA NIM):**
    * `nvidia/nv-embedqa-e5-v5` (Geração de Embeddings - 1024 dimensões)
    * `meta/llama-3.3-70b-instruct` (LLM para Análise Crítica)
* **Leitura de PDF:** `pdfplumber`
* **Processamento Vetorial:** `numpy`
* **Frontend:** HTML5, CSS3 e JavaScript Vanilla (WebSockets & Fetch API)

---

## 📋 Pré-requisitos

Antes de começar, você precisará:
1. Ter o **Python 3.12 ou superior** instalado na sua máquina.
2. Uma chave de API ativa da NVIDIA (NVIDIA NIM). Você pode obter créditos gratuitos em [NVIDIA Build](https://build.nvidia.com/).

---

## 🔧 Instalação e Configuração

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
   cd seu-repositorio

    Crie e ative o ambiente virtual (venv):
    Bash

    # No Linux/macOS:
    python3 -m venv venv
    source venv/bin/activate

    # No Windows (Prompt de Comando):
    python -m venv venv
    venv\Scripts\activate

    Instale as dependências necessárias:
    Bash

    pip install fastapi uvicorn python-multipart openai numpy pdfplumber python-dotenv

    Configure as Variáveis de Ambiente:
    Crie um arquivo chamado .env na raiz do projeto e adicione a sua chave da NVIDIA:
    Snippet de código

    NVIDIA_API_KEY=nvapi-sua-chave-aqui-xxxxxxxxxxxxxxxxxxxx

## 🖥️ Estrutura do Projeto

Garanta que a estrutura de pastas do seu projeto esteja organizada da seguinte forma:
Plaintext

portalAEI/
├── data/                     # Pasta criada automaticamente para uploads temporários
├── src/                      # Código-fonte isolado da aplicação
│   └── research_analyzer.py  # Módulo com a lógica de IA, RAG e NVIDIA NIM
├── templates/                # Arquivos de visualização (Frontend)
│   └── index.html            # Interface web com WebSocket e botão para PDF
├── venv/                     # Ambiente virtual do Python (dependências locais)
├── .env                      # Variáveis de ambiente (NVIDIA_API_KEY)
├── .gitignore                # Arquivo para o Git ignorar o venv e pycache
├── main.py                   # Ponto de entrada e rotas do servidor FastAPI
├── README.md                 # Documentação do projeto
└── requirements.txt          # Lista de dependências do projeto

## ⚡ Como Executar

Com o ambiente virtual ativo, execute o servidor Uvicorn rodando o arquivo main.py:
Bash

python main.py

O servidor iniciará localmente. Abra o seu navegador e acesse o endereço:
👉 http://127.0.0.1:8000
📖 Como Funciona o Pipeline de Análise

    Upload: O arquivo PDF é enviado à rota /upload/{client_id}.

    Fase de Mapeamento: O texto bruto é extraído e passa pela função identificar_secoes para encontrar os limites lógicos do artigo.

    Chunking de Segurança: O texto limpo é quebrado em blocos de no máximo 1200 caracteres (gerando cerca de 400 tokens), blindando o código contra o erro de limite de 512 tokens do modelo de embedding da NVIDIA.

    Visão Global: O Llama-3.3 lê os primeiros chunks para entender o objetivo macro do autor.

    Busca Semântica (RAG): Para cada critério de avaliação (Rigor Metodológico, Escrita, etc.), o sistema realiza uma busca por similaridade de cosseno e envia os 4 blocos mais relevantes para o LLM gerar a crítica contextualizada.
