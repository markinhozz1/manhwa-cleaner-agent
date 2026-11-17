"""
Manhwa Balloon & Free Text Cleaner Engine (v3.1)
- Detecção neural de alta precisão (RT-DETR treinado para quadrinhos/webtoons)
- 1. Preservação estrita das bordas dos balões (componentes encostados na extremidade são ignorados)
- 2. Suporte perfeito a balões escuros, degradês e semitransparentes (amostragem perimetral e distância de cor BGR)
- 3. Remoção de falas em inglês fora de balões (text_free) com verificação EasyOCR para proteger SFX coreano e arte
- Exportação em .PSD multicamadas (Original e Editado)
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import easyocr
import pytoshop
from pytoshop import enums
from pytoshop.user import nested_layers
import packbits

pytoshop.codecs.packbits = packbits


def stroke_dilation_params(stroke_dilate: int) -> tuple[int, int]:
    """Converte --dilate em (tamanho do kernel, iterações) para cv2.dilate.

    Com o flag ignorado, a dilatação do balão era uma elipse 5x5 em uma
    iteração. O padrão da CLI é 2, e ``2 * 2 + 1 == 5``, então omitir
    --dilate continua idêntico a esse passe. O diâmetro do kernel acompanha
    o parâmetro (``2 * n + 1``). Iterações: 1 quando n > 0 e 0 para desligar.
    """
    try:
        level = int(stroke_dilate)
    except (TypeError, ValueError):
        level = 2
    if level <= 0:
        return 0, 0
    return (2 * level + 1), 1


def expand_crop(x1: int, y1: int, x2: int, y2: int, pad: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Amplia um retângulo por ``pad`` px, limitado aos limites da imagem."""
    pad = max(0, int(pad))
    return (
        max(0, int(x1) - pad),
        max(0, int(y1) - pad),
        min(int(width), int(x2) + pad),
        min(int(height), int(y2) + pad),
    )


def select_compute_device() -> tuple[str, str]:
    """Escolhe 'cuda', 'mps' ou 'cpu'. Não propaga ausência de torch ou CUDA."""
    try:
        import torch
    except Exception as exc:
        return 'cpu', f'torch indisponível ({type(exc).__name__}: {exc})'

    cuda_note = ''
    try:
        if bool(torch.cuda.is_available()):
            return 'cuda', 'torch.cuda.is_available()'
    except Exception as exc:
        cuda_note = f'consulta CUDA falhou ({exc})'

    mps_note = ''
    try:
        backends = getattr(torch, 'backends', None)
        mps_backend = getattr(backends, 'mps', None) if backends is not None else None
        if mps_backend is not None and bool(mps_backend.is_available()):
            return 'mps', 'torch.backends.mps.is_available()'
    except Exception as exc:
        mps_note = f'consulta MPS falhou ({exc})'

    extras = [note for note in (cuda_note, mps_note) if note]
    if extras:
        return 'cpu', '; '.join(extras)
    return 'cpu', 'CUDA e MPS indisponíveis'


def detector_providers(device: str) -> list[str]:
    """Providers do ONNX Runtime para o dispositivo, com CPU no fim da lista."""
    try:
        available = list(ort.get_available_providers())
    except Exception as exc:
        print(f'  [!] Não foi possível listar os providers do ONNX Runtime ({exc}). Usando CPU.')
        return ['CPUExecutionProvider']

    chosen = None
    if device == 'cuda' and 'CUDAExecutionProvider' in available:
        chosen = 'CUDAExecutionProvider'
    elif device == 'mps' and 'CoreMLExecutionProvider' in available:
        chosen = 'CoreMLExecutionProvider'

    if chosen is None:
        if device in ('cuda', 'mps'):
            listed = ', '.join(available) if available else 'nenhum'
            print(
                f"  [!] ONNX Runtime não oferece GPU para '{device}' "
                f'(providers: {listed}). Detector em CPU.'
            )
        return ['CPUExecutionProvider']
    return [chosen, 'CPUExecutionProvider']


def _open_detector_session(model_path: str, providers: list[str]):
    """Abre a sessão e devolve também os providers que de fato foram aceitos."""
    try:
        return ort.InferenceSession(model_path, providers=providers), list(providers)
    except Exception as exc:
        if providers == ['CPUExecutionProvider']:
            raise
        print(f'  [!] Falha ao abrir o detector em {providers[0]} ({exc}). Recuando para CPU.')
        cpu = ['CPUExecutionProvider']
        return ort.InferenceSession(model_path, providers=cpu), cpu


def _open_ocr_reader(device: str):
    """EasyOCR na GPU quando o dispositivo é cuda/mps; recua para CPU se falhar."""
    if device not in ('cuda', 'mps'):
        return easyocr.Reader(['en'], gpu=False, verbose=False)
    try:
        return easyocr.Reader(['en'], gpu=True, verbose=False)
    except Exception as exc:
        print(f'  [!] EasyOCR não iniciou em {device} ({exc}). Recuando para CPU.')
        return easyocr.Reader(['en'], gpu=False, verbose=False)


class ManhwaCleaner:
    def __init__(self, inpaint_radius: int = 3, stroke_dilate: int = 2, conf_threshold: float = 0.35, box_pad: int = 8):
        self.inpaint_radius = inpaint_radius
        self.stroke_dilate = stroke_dilate
        self.conf_threshold = conf_threshold
        self.box_pad = max(0, int(box_pad))

        self.device, device_reason = select_compute_device()
        print(f'  [*] Dispositivo detectado: {self.device} ({device_reason})')

        print("  [*] Carregando modelo neural Comic Detector (RT-DETR)...")
        model_path = hf_hub_download(
            repo_id='ogkalu/comic-text-and-bubble-detector',
            filename='detector-v4-s_int8.onnx'
        )
        providers = detector_providers(self.device)
        self.session, providers = _open_detector_session(model_path, providers)
        active = providers
        try:
            active = list(self.session.get_providers()) or providers
        except Exception:
            active = providers
        print(f'  [*] Detector RT-DETR em: {active[0]}')

        print("  [*] Inicializando detector de texto em inglês (EasyOCR)...")
        self.ocr_reader = _open_ocr_reader(self.device)
        ocr_device = getattr(self.ocr_reader, 'device', 'cpu')
        print(f'  [*] EasyOCR em: {ocr_device}')
        print("  [OK] Pipeline neural pronto.")

    def detect_elements(self, chunk_bgr: np.ndarray) -> list:
        ch_h, ch_w = chunk_bgr.shape[:2]
        
        resized = cv2.resize(chunk_bgr, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor_img = np.transpose(rgb, (2, 0, 1))[None, ...]
        
        target_sizes = np.array([[ch_w, ch_h]], dtype=np.int64)
        
        outputs = self.session.run(None, {
            'images': tensor_img,
            'orig_target_sizes': target_sizes
        })
        
        labels = outputs[0][0]
        boxes = outputs[1][0]
        scores = outputs[2][0]
        
        detected_items = []
        for label, box, score in zip(labels, boxes, scores):
            # label 1 = text_bubble (fala dentro de balão)
            # label 2 = text_free (fala fora de balão)
            if label in [1, 2] and score >= self.conf_threshold:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(0, min(ch_w - 1, x1))
                y1 = max(0, min(ch_h - 1, y1))
                x2 = max(0, min(ch_w, x2))
                y2 = max(0, min(ch_h, y2))
                
                if x2 > x1 and y2 > y1:
                    detected_items.append((int(label), x1, y1, x2 - x1, y2 - y1, float(score)))
                    
        return detected_items

    def segment_bubble_text(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        Segmenta o texto de um balão:
        - Calcula o fundo a partir das bordas do ROI (preserva balões escuros, brancos e degradês).
        - Elimina qualquer componente que encoste nas bordas para NUNCA cortar o contorno do balão.
        """
        roi_h, roi_w = roi_bgr.shape[:2]
        if roi_h < 6 or roi_w < 6:
            return np.zeros((roi_h, roi_w), dtype=np.uint8)
            
        # Amostra a borda externa (3px) para encontrar a cor real do fundo
        top = roi_bgr[:3, :]
        bot = roi_bgr[-3:, :]
        left = roi_bgr[:, :3]
        right = roi_bgr[:, -3:]
        perimeter_bgr = np.concatenate([
            top.reshape(-1, 3), bot.reshape(-1, 3),
            left.reshape(-1, 3), right.reshape(-1, 3)
        ])
        
        bg_bgr = np.median(perimeter_bgr, axis=0)
        bg_lum = 0.299 * bg_bgr[2] + 0.587 * bg_bgr[1] + 0.114 * bg_bgr[0]
        
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(roi_bgr, bg_bgr.astype(np.uint8))
        dist = np.max(diff, axis=2)
        
        if bg_lum < 100:
            # Balão escuro (roxo, preto, etc): texto é mais claro que o fundo
            raw_mask = ((gray > (bg_lum + 18)) | (dist > 22)).astype(np.uint8) * 255
        else:
            # Balão claro (branco, pastel, etc): texto é mais escuro que o fundo
            raw_mask = ((gray < (bg_lum - 20)) | (dist > 25)).astype(np.uint8) * 255
            
        # Filtra componentes conexos para proteger as bordas do balão
        num_labels, lmap, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
        clean_mask = np.zeros_like(gray)
        
        for i in range(1, num_labels):
            cx = stats[i, cv2.CC_STAT_LEFT]
            cy = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Contorno do balão encosta na borda deste recorte (o chamador
            # amplia a caixa antes). Letra que só tocava a caixa do detector
            # fica para dentro e é removida.
            touches_edge = (cx == 0) or (cy == 0) or (cx + cw >= roi_w) or (cy + ch >= roi_h)
            if not touches_edge and area >= 4:
                clean_mask[lmap == i] = 255

        # Dilatação do traço. --dilate chega em self.stroke_dilate; o padrão
        # 2 continua sendo elipse 5x5 em 1 iteração.
        kernel_size, iterations = stroke_dilation_params(self.stroke_dilate)
        if kernel_size > 0 and iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            clean_mask = cv2.dilate(clean_mask, kernel, iterations=iterations)
        return clean_mask

    def segment_free_text(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        Segmenta falas em inglês FORA de balões (text_free).
        Usa EasyOCR para validar que é realmente uma fala em inglês
        e não arte desenhada ou onomatopeias coreanas.
        """
        roi_h, roi_w = roi_bgr.shape[:2]
        if roi_h < 10 or roi_w < 10:
            return np.zeros((roi_h, roi_w), dtype=np.uint8)
            
        ocr_results = self.ocr_reader.readtext(roi_bgr)
        has_english = False
        for _, text, conf in ocr_results:
            alpha_count = sum(1 for c in text if c.isalpha())
            if alpha_count >= 2 and conf >= 0.20:
                has_english = True
                break
                
        if not has_english:
            # Não é texto em inglês -> Preserva intacto!
            return np.zeros((roi_h, roi_w), dtype=np.uint8)
            
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Falas livres em quadrinhos costumam ser brancas com contorno preto
        white_core = (gray > 190).astype(np.uint8) * 255
        if np.count_nonzero(white_core) > 20:
            # Dilata para cobrir o contorno preto das letras
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            return cv2.dilate(white_core, kernel, iterations=1)
        else:
            # Letras escuras
            dark_core = (gray < 75).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            return cv2.dilate(dark_core, kernel, iterations=1)

    def clean_webtoon_strip(self, img_bgr: np.ndarray, chunk_h: int = 1600, overlap: int = 300) -> tuple[np.ndarray, np.ndarray]:
        """
        Processa tiras longas de webtoon em blocos com inpainting inteligente.
        """
        h, w = img_bgr.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        
        y = 0
        step = chunk_h - overlap
        
        while y < h:
            y_end = min(y + chunk_h, h)
            chunk = img_bgr[y:y_end, :]
            
            items = self.detect_elements(chunk)
            
            for (label, bx, by, bw, bh, score) in items:
                gx1 = bx
                gy1 = y + by
                gx2 = min(w, bx + bw)
                gy2 = min(h, gy1 + bh)

                # Balão: amplia o recorte antes de segmentar. A caixa do
                # detector costuma encostar na letra, e o filtro de borda
                # a tratava como contorno. O pad fica limitado à imagem; a
                # máscara (do recorte ampliado) volta para essas coordenadas.
                # Texto livre não usa esse filtro e mantém a caixa original.
                pad = self.box_pad if label == 1 else 0
                ex1, ey1, ex2, ey2 = expand_crop(gx1, gy1, gx2, gy2, pad, w, h)

                roi = img_bgr[ey1:ey2, ex1:ex2]
                if roi.size == 0:
                    continue

                if label == 1:
                    stroke_mask = self.segment_bubble_text(roi)
                else:
                    stroke_mask = self.segment_free_text(roi)

                full_mask[ey1:ey2, ex1:ex2] = cv2.bitwise_or(full_mask[ey1:ey2, ex1:ex2], stroke_mask)
                
            if y_end == h:
                break
            y += step
            
        if np.any(full_mask):
            cleaned_bgr = cv2.inpaint(img_bgr, full_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        else:
            cleaned_bgr = img_bgr.copy()
            
        return cleaned_bgr, full_mask

    def export_to_psd(self, original_bgr: np.ndarray, cleaned_bgr: np.ndarray, output_path: str):
        """
        Grava o arquivo .PSD com camadas 'Original' e 'Editado'.
        """
        h, w = original_bgr.shape[:2]
        
        orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        clean_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        
        channels_orig = {
            0: orig_rgb[:, :, 0],
            1: orig_rgb[:, :, 1],
            2: orig_rgb[:, :, 2],
            -1: alpha
        }
        
        channels_clean = {
            0: clean_rgb[:, :, 0],
            1: clean_rgb[:, :, 1],
            2: clean_rgb[:, :, 2],
            -1: alpha
        }
        
        layer_orig = nested_layers.Image(
            name='Original',
            visible=True,
            opacity=255,
            top=0, left=0, bottom=h, right=w,
            channels=channels_orig
        )
        
        layer_clean = nested_layers.Image(
            name='Editado',
            visible=True,
            opacity=255,
            top=0, left=0, bottom=h, right=w,
            channels=channels_clean
        )
        
        # Ordem visual no Photoshop (topo para base): Editado em cima, Original em baixo
        layout = [layer_clean, layer_orig]
        psd_obj = nested_layers.nested_layers_to_psd(
            layout, color_mode=3, compression=enums.Compression.raw
        )
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'wb') as fd:
            psd_obj.write(fd)
            
        return output_path
