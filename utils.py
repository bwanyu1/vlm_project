def get_captions(item) -> list:
    """
    STAIR-Captionsのデータ構造からキャプションリストを取得する

    データ構造:
      item["annotations"]["caption"] = ['キャプション1', 'キャプション2', ...]
    """
    # STAIR-Captions形式
    annotations = item.get("annotations")
    if annotations and isinstance(annotations, dict):
        caps = annotations.get("caption", [])
        if caps:
            return [str(c).strip() for c in caps if str(c).strip()]

    # フォールバック（他のデータセット用）
    for key in ["captions", "caption", "text", "description"]:
        val = item.get(key)
        if val is None:
            continue
        if isinstance(val, list) and val:
            return [str(c).strip() for c in val if str(c).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip()]

    return [""]