from research_analyzer import analisar_artigo

# SUBSTITUA PELO CAMINHO REAL DO SEU PDF
caminho_do_seu_pdf = "artigo2.pdf"


# Executa a análise
relatorio = analisar_artigo(caminho_do_seu_pdf)

print("\n" + "="*50)
print("RELATÓRIO DE ANÁLISE")
print("="*50)
print(relatorio)