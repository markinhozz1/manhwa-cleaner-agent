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
import json
import time
import traceback
import cv2
from cleaner_core import ManhwaCleaner

PROGRESS_FILENAME = "progress.json"


def _caminho_progresso(pasta_saida: str) -> str:
    return os.path.join(pasta_saida, PROGRESS_FILENAME)


def _carregar_progresso(path: str) -> dict:
    if not os.path.isfile(path):
        return {"pages": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("pages"), dict):
            return data
    except Exception as exc:
        print(f"  [!] Progresso ilegível ({path}): {exc}. Começando um arquivo novo.")
    return {"pages": {}}


def _salvar_progresso(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _tentar_salvar_progresso(path: str, data: dict) -> None:
    try:
        _salvar_progresso(path, data)
    except Exception as exc:
        print(f"  [!] Não foi possível gravar o progresso ({path}): {exc}")


def _psd_concluido(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def processar_capitulo(cleaner: ManhwaCleaner, pasta_capitulo: str) -> list[tuple[str, str]]:
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
        return []
        
    nome_cap = os.path.basename(os.path.normpath(pasta_capitulo))
    print(f"\n=======================================================")
    print(f"[*] Iniciando: {nome_cap} ({len(arquivos)} páginas)")
    print(f"[*] Destino: {pasta_saida}")
    print(f"=======================================================")

    progresso_path = _caminho_progresso(pasta_saida)
    progresso = _carregar_progresso(progresso_path)
    progresso["chapter"] = os.path.abspath(pasta_capitulo)
    progresso["output_dir"] = os.path.abspath(pasta_saida)
    paginas = progresso.setdefault("pages", {})
    
    sucessos = 0
    falhas: list[tuple[str, str]] = []
    t_cap_inicio = time.time()
    
    for i, arq in enumerate(arquivos, 1):
        nome_arquivo = os.path.basename(arq)
        nome_base = os.path.splitext(nome_arquivo)[0] + ".psd"
        destino_psd = os.path.join(pasta_saida, nome_base)
        parcial = destino_psd + ".partial"

        if _psd_concluido(destino_psd):
            print(f"  [{i:02d}/{len(arquivos):02d}] {nome_arquivo} já concluída ({nome_base}), pulando.")
            paginas[nome_arquivo] = {
                "status": "ok",
                "source": arq,
                "output": destino_psd,
                "skipped": True,
            }
            _tentar_salvar_progresso(progresso_path, progresso)
            sucessos += 1
            continue
        
        t0 = time.time()
        try:
            img = cv2.imread(arq)
            if img is None:
                raise RuntimeError(f"não foi possível abrir a imagem: {arq}")

            h, w = img.shape[:2]
            cleaned, _ = cleaner.clean_webtoon_strip(img, chunk_h=1600, overlap=300)
            cleaner.export_to_psd(img, cleaned, parcial)
            os.replace(parcial, destino_psd)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  [!] Erro ao processar {arq}")
            print(tb)
            if os.path.isfile(parcial):
                try:
                    os.remove(parcial)
                except OSError as rm_exc:
                    print(f"  [!] Não foi possível remover {parcial}: {rm_exc}")
            falhas.append((arq, str(exc)))
            paginas[nome_arquivo] = {
                "status": "error",
                "source": arq,
                "error": str(exc),
                "traceback": tb,
            }
            _tentar_salvar_progresso(progresso_path, progresso)
            continue
        
        duracao = time.time() - t0
        print(f"  [{i:02d}/{len(arquivos):02d}] {nome_arquivo} ({w}x{h}px) -> {nome_base} ({duracao:.2f}s)")
        sucessos += 1
        paginas[nome_arquivo] = {
            "status": "ok",
            "source": arq,
            "output": destino_psd,
            "skipped": False,
            "seconds": round(duracao, 2),
        }
        _tentar_salvar_progresso(progresso_path, progresso)
        
    t_total = time.time() - t_cap_inicio
    print(
        f"\n[DONE] Capítulo {nome_cap} concluído: {sucessos}/{len(arquivos)} páginas "
        f"em {t_total:.1f}s. Falhas: {len(falhas)}."
    )
    return falhas


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
    todas_falhas: list[tuple[str, str]] = []
    for cap in capitulos:
        if os.path.isdir(cap):
            try:
                falhas = processar_capitulo(cleaner, cap)
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[!] Erro ao processar o capítulo {cap}")
                print(tb)
                todas_falhas.append((cap, str(exc)))
                continue
            todas_falhas.extend(falhas)
        else:
            print(f"[!] Diretório não encontrado: {cap}")
            
    print(f"\n[★] Processamento concluído em {time.time() - t_global:.1f}s!")
    if todas_falhas:
        print(f"\n[!] Resumo de páginas com falha ({len(todas_falhas)}):")
        for caminho, erro in todas_falhas:
            print(f"    - {caminho}: {erro}")
    else:
        print("\n[OK] Resumo: nenhuma página falhou.")


if __name__ == "__main__":
    main()
