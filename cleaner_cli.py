"""
Interface de Linha de Comando (CLI) para o Manhwa Cleaner Agent.
Permite processar páginas individuais ou pastas completas (capítulos inteiros),
com suporte automático ao fatiamento de tiras longas de Webtoon e exportação em .PSD de 2 camadas.
"""

import os
import sys

# Configura encoding de saída para evitar erros no console do Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import glob
import time
import argparse
import cv2
import numpy as np
from cleaner_core import ManhwaCleaner


def processar_tira_webtoon(cleaner: ManhwaCleaner, img_bgr: np.ndarray, chunk_height: int = 2500, overlap: int = 150):
    """
    Fatia tiras verticais gigantescas de webtoon com sobreposição (overlap)
    para evitar estouro de memória e garantir limpeza homogênea.
    """
    h, w = img_bgr.shape[:2]
    
    # Se a imagem for de tamanho comum, processa direto
    if h <= chunk_height:
        return cleaner.clean_image(img_bgr)
        
    print(f"  [Webtoon Strip detectada: {w}x{h}px] Dividindo em blocos de {chunk_height}px...")
    
    cleaned_full = np.zeros_like(img_bgr)
    mask_full = np.zeros((h, w), dtype=np.uint8)
    
    y = 0
    while y < h:
        y_end = min(y + chunk_height, h)
        chunk = img_bgr[y:y_end, :]
        
        chunk_clean, chunk_mask = cleaner.clean_image(chunk)
        
        cleaned_full[y:y_end, :] = chunk_clean
        mask_full[y:y_end, :] = chunk_mask
        
        if y_end == h:
            break
        y += (chunk_height - overlap)
        
    return cleaned_full, mask_full


def processar_arquivo(cleaner: ManhwaCleaner, input_path: str, output_path: str):
    print(f"\n[+] Processando: {os.path.basename(input_path)}")
    inicio = time.time()
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"  [!] Erro: não foi possível abrir a imagem {input_path}")
        return False
        
    cleaned_img, _ = cleaner.clean_webtoon_strip(img, chunk_h=1600, overlap=300)
    
    # Garante extensão .psd
    if not output_path.lower().endswith(".psd"):
        output_path = os.path.splitext(output_path)[0] + ".psd"
        
    cleaner.export_to_psd(img, cleaned_img, output_path)
    duracao = time.time() - inicio
    print(f"  [OK] Salvo em .PSD: {output_path} ({duracao:.2f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Manhwa Cleaner Agent: Limpa balões brancos, degradês e transparentes gerando .PSD com camadas."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Caminho para uma imagem (.png, .jpg, .webp) ou diretório contendo o capítulo."
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Caminho de saída do .PSD ou diretório de destino."
    )
    parser.add_argument(
        "--dilate", type=int, default=2,
        help="Dilatação do traço da letra (padrão: 2). Aumente se sobrarem halos pretos da fonte."
    )
    parser.add_argument(
        "--radius", type=int, default=3,
        help="Raio do inpainting (padrão: 3). Ideal para degradês suaves."
    )
    parser.add_argument(
        "--pad", type=int, default=8,
        help="Pixels extras ao redor da caixa de texto antes de segmentar o balão (padrão: 8)."
    )
    
    args = parser.parse_args()
    
    cleaner = ManhwaCleaner(
        inpaint_radius=args.radius,
        stroke_dilate=args.dilate,
        box_pad=args.pad,
    )
    
    input_path = os.path.abspath(args.input)
    
    if os.path.isfile(input_path):
        if args.output:
            out_file = args.output
        else:
            out_file = os.path.splitext(input_path)[0] + "_clean.psd"
        processar_arquivo(cleaner, input_path, out_file)
        
    elif os.path.isdir(input_path):
        out_dir = args.output if args.output else os.path.join(input_path, "psd_limpos")
        os.makedirs(out_dir, exist_ok=True)
        
        extensoes = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        arquivos = []
        for ext in extensoes:
            arquivos.extend(glob.glob(os.path.join(input_path, ext)))
            arquivos.extend(glob.glob(os.path.join(input_path, ext.upper())))
            
        arquivos = sorted(list(set(arquivos)))
        
        if not arquivos:
            print(f"[!] Nenhuma imagem encontrada em: {input_path}")
            return
            
        print(f"[*] Encontradas {len(arquivos)} páginas para limpar. Destino: {out_dir}")
        sucessos = 0
        for arq in arquivos:
            nome_base = os.path.splitext(os.path.basename(arq))[0] + ".psd"
            destino = os.path.join(out_dir, nome_base)
            if processar_arquivo(cleaner, arq, destino):
                sucessos += 1
                
        print(f"\n[DONE] Concluido! {sucessos}/{len(arquivos)} paginas exportadas em .PSD com sucesso.")
    else:
        print(f"[!] Caminho de entrada inválido: {input_path}")


if __name__ == "__main__":
    main()
