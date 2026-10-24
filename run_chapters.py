"""
Script para processamento em lote sequencial de capítulos de Webtoon / Manhwa.
Mantém o modelo neural em memória para máxima velocidade de execução.

Uso:
    python run_chapters.py "caminho/para/capitulo1" "caminho/para/capitulo2"
    ou configure a lista `capitulos` diretamente dentro da função main().
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import glob
import time
import cv2
from cleaner_core import ManhwaCleaner


def processar_capitulo(cleaner: ManhwaCleaner, pasta_capitulo: str):
    pasta_saida = os.path.join(pasta_capitulo, "psd_limpos")
    os.makedirs(pasta_saida, exist_ok=True)
    
    extensoes = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    arquivos = []
    for ext in extensoes:
        arquivos.extend(glob.glob(os.path.join(pasta_capitulo, ext)))
        arquivos.extend(glob.glob(os.path.join(pasta_capitulo, ext.upper())))
        
    # Filtra para não pegar imagens dentro de subpastas (como psd_limpos)
    arquivos = sorted(list(set([a for a in arquivos if os.path.dirname(os.path.abspath(a)) == os.path.abspath(pasta_capitulo)])))
    
    if not arquivos:
        print(f"[!] Nenhuma imagem encontrada em {pasta_capitulo}")
        return
        
    nome_cap = os.path.basename(os.path.normpath(pasta_capitulo))
    print(f"\n=======================================================")
    print(f"[*] Iniciando: {nome_cap} ({len(arquivos)} páginas)")
    print(f"[*] Destino: {pasta_saida}")
    print(f"=======================================================")
    
    sucessos = 0
    t_cap_inicio = time.time()
    
    for i, arq in enumerate(arquivos, 1):
        nome_base = os.path.splitext(os.path.basename(arq))[0] + ".psd"
        destino_psd = os.path.join(pasta_saida, nome_base)
        
        t0 = time.time()
        img = cv2.imread(arq)
        if img is None:
            print(f"  [!] Falha ao abrir {arq}")
            continue
            
        h, w = img.shape[:2]
        cleaned, _ = cleaner.clean_webtoon_strip(img, chunk_h=1600, overlap=300)
        cleaner.export_to_psd(img, cleaned, destino_psd)
        
        duracao = time.time() - t0
        print(f"  [{i:02d}/{len(arquivos):02d}] {os.path.basename(arq)} ({w}x{h}px) -> {nome_base} ({duracao:.2f}s)")
        sucessos += 1
        
    t_total = time.time() - t_cap_inicio
    print(f"\n[DONE] Capítulo {nome_cap} concluído: {sucessos}/{len(arquivos)} páginas em {t_total:.1f}s!")


def main():
    # Se passar pastas como argumentos na linha de comando, usa elas.
    # Ex: python run_chapters.py "E:\Scan\Obra\101" "E:\Scan\Obra\102"
    if len(sys.argv) > 1:
        capitulos = [os.path.abspath(arg) for arg in sys.argv[1:]]
    else:
        # Ou adicione os caminhos das pastas diretamente nesta lista:
        capitulos = [
            # r"C:\Caminho\Para\Seu\Capitulo\01",
            # r"C:\Caminho\Para\Seu\Capitulo\02",
        ]
        
    if not capitulos:
        print("Nenhum capítulo informado.")
        print("Uso: python run_chapters.py <pasta1> [pasta2 ...]")
        print("Ou edite a lista `capitulos` dentro de run_chapters.py.")
        return
        
    print("Iniciando Manhwa Cleaner Agent (v3.1)...")
    cleaner = ManhwaCleaner(inpaint_radius=3, stroke_dilate=2)
    
    t_global = time.time()
    for cap in capitulos:
        if os.path.isdir(cap):
            processar_capitulo(cleaner, cap)
        else:
            print(f"[!] Diretório não encontrado: {cap}")
            
    print(f"\n[★] Processamento concluído em {time.time() - t_global:.1f}s!")


if __name__ == "__main__":
    main()
