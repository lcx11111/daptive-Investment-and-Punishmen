from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "5_6_source.pptx"
OUT = ROOT / "5_6_result_figures_revised.pptx"
FIG = ROOT / "figures"

REPLACEMENTS = {
    23: {
        "media": "ppt/media/image47.png",
        "image": FIG / "model_F_investment_q_profiles.png",
        "pos": {"x": "751205", "y": "1111250", "cx": "6026150", "cy": "4420870"},
    },
    24: {
        "media": "ppt/media/image48.png",
        "image": FIG / "model_F_punishment_q_policy_map.png",
        "pos": {"x": "3500000", "y": "840000", "cx": "5300000", "cy": "5300000"},
    },
    25: {
        "media": "ppt/media/image49.png",
        "image": FIG / "model_F_punishment_q_bar_facets.png",
        "pos": {"x": "1610000", "y": "298450", "cx": "8970000", "cy": "6197600"},
    },
}

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def update_slide_geometry(xml_bytes, pos):
    root = ET.fromstring(xml_bytes)
    pic = root.find(".//p:pic", NS)
    if pic is None:
        return xml_bytes
    xfrm = pic.find(".//a:xfrm", NS)
    if xfrm is None:
        return xml_bytes
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    if off is not None:
        off.set("x", pos["x"])
        off.set("y", pos["y"])
    if ext is not None:
        ext.set("cx", pos["cx"])
        ext.set("cy", pos["cy"])
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main():
    missing = [str(cfg["image"]) for cfg in REPLACEMENTS.values() if not cfg["image"].exists()]
    if missing:
        raise FileNotFoundError("Missing replacement figures: " + "; ".join(missing))

    slide_xml_names = {f"ppt/slides/slide{slide_no}.xml": cfg for slide_no, cfg in REPLACEMENTS.items()}
    media_names = {cfg["media"]: cfg for cfg in REPLACEMENTS.values()}

    with ZipFile(SRC, "r") as zin, ZipFile(OUT, "w", ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in media_names:
                data = media_names[item.filename]["image"].read_bytes()
            elif item.filename in slide_xml_names:
                data = update_slide_geometry(data, slide_xml_names[item.filename]["pos"])
            zout.writestr(item, data)

    print(OUT)


if __name__ == "__main__":
    main()
