from PIL import Image
import requests
from io import BytesIO

IMAGE_FETCH_TIMEOUT = 5


def get_captions(item, dataset_name: str = "") -> list:
    """
    データセット別にキャプションを取得する共通関数

    対応:
      STAIR-Captions : item["annotations"]["caption"]
      DEJIMA         : item["caption"]
      COCO-Caption   : item["answer"]
      GAIA           : item["captions"]
      cc3m-wds       : item["txt"]
    """
    # STAIR-Captions
    annotations = item.get("annotations")
    if annotations and isinstance(annotations, dict):
        caps = annotations.get("caption", [])
        if caps:
            return [str(c).strip() for c in caps if str(c).strip()]

    # COCO-Caption (answer フィールド)
    answer = item.get("answer")
    if answer:
        if isinstance(answer, list):
            return [str(c).strip() for c in answer if str(c).strip()]
        if isinstance(answer, str) and answer.strip():
            return [answer.strip()]

    # cc3m-wds
    txt = item.get("txt")
    if txt and isinstance(txt, str) and txt.strip():
        return [txt.strip()]

    # DEJIMA / GAIA / 汎用フォールバック
    for key in ["caption", "captions", "text", "description"]:
        val = item.get(key)
        if val is None:
            continue
        if isinstance(val, list) and val:
            return [str(c).strip() for c in val if str(c).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip()]

    return [""]


def get_image(item, image_size: int):
    """
    データセット別に画像を取得する

    Returns:
        (PIL.Image, success: bool)
        success=False の場合は黒画像を返す（学習時にスキップ推奨）
    """
    # image / jpg フィールド（埋め込み済み）
    for key in ["image", "jpg"]:
        img = item.get(key)
        if img is not None:
            try:
                if isinstance(img, Image.Image):
                    return img.convert("RGB"), True
                if isinstance(img, bytes):
                    return Image.open(BytesIO(img)).convert("RGB"), True
            except Exception:
                pass

    # URLから取得
    for key in ["url", "coco_url", "image_src", "flickr_url"]:
        url = item.get(key)
        if url and isinstance(url, str) and url.startswith("http"):
            try:
                resp = requests.get(url, timeout=IMAGE_FETCH_TIMEOUT)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return img, True
            except Exception:
                continue

    # 全て失敗 → 黒画像 + 失敗フラグ
    return Image.new("RGB", (image_size, image_size), color=0), False