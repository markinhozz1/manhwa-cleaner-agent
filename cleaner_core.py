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


class ManhwaCleaner:
    def __init__(self, inpaint_radius: int = 3, stroke_dilate: int = 2, conf_threshold: float = 0.35):
        self.inpaint_radius = inpaint_radius
        self.stroke_dilate = stroke_dilate
        self.conf_threshold = conf_threshold
        
        print("  [*] Carregando modelo neural Comic Detector (RT-DETR)...")
        model_path = hf_hub_download(
            repo_id='ogkalu/comic-text-and-bubble-detector',
            filename='detector-v4-s_int8.onnx'
        )
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        print("  [*] Inicializando detector de texto em inglês (EasyOCR)...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
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
            
            # Se encosta na borda do retângulo, faz parte da linha/borda do balão, NÃO é texto!
            touches_edge = (cx == 0) or (cy == 0) or (cx + cw >= roi_w) or (cy + ch >= roi_h)
            if not touches_edge and area >= 4:
                clean_mask[lmap == i] = 255
                
        # Dilatação suave para cobrir antialiasing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)
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
                
                roi = img_bgr[gy1:gy2, gx1:gx2]
                if roi.size == 0:
                    continue
                    
                if label == 1:
                    stroke_mask = self.segment_bubble_text(roi)
                else:
                    stroke_mask = self.segment_free_text(roi)
                    
                full_mask[gy1:gy2, gx1:gx2] = cv2.bitwise_or(full_mask[gy1:gy2, gx1:gx2], stroke_mask)
                
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
