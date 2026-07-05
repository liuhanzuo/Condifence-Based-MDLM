"""
把 observations.md 导出为 PDF，图片以 base64 内嵌（确保不丢图）。
markdown -> 自包含 HTML -> Chromium 打印 PDF。

运行： python exp/export_pdf.py --md observations.md --out observations.pdf
"""
import argparse
import base64
import os
import re
import mimetypes

import markdown


def embed_images(html, md_dir):
    """把 <img src="rel/path"> 的本地图片转成 base64 data URI。"""
    def repl(m):
        src = m.group(1)
        if src.startswith("data:") or src.startswith("http"):
            return m.group(0)
        # 相对 md 文件解析
        path = os.path.normpath(os.path.join(md_dir, src))
        if not os.path.exists(path):
            # 尝试去掉开头 ../
            alt = os.path.normpath(os.path.join(md_dir, src.lstrip("./").replace("../", "")))
            path = alt if os.path.exists(alt) else path
        if not os.path.exists(path):
            print(f"  [WARN] image not found: {src} -> {path}")
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        print(f"  embedded: {src} ({os.path.getsize(path)//1024} KB)")
        return f'<img src="data:{mime};base64,{b64}"'

    return re.sub(r'<img\s+src="([^"]+)"', repl, html)


CSS = """
<style>
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Microsoft YaHei","Segoe UI",Arial,sans-serif;
       font-size: 12px; line-height: 1.6; color: #222; }
h1 { font-size: 22px; border-bottom: 3px solid #3b7; padding-bottom: 6px; }
h2 { font-size: 17px; color: #1a6; border-bottom: 1px solid #ddd;
     padding-bottom: 4px; margin-top: 24px; page-break-after: avoid; }
h3 { font-size: 14px; color: #333; }
blockquote { border-left: 4px solid #3b7; background: #f6fbf8;
             margin: 8px 0; padding: 6px 12px; color: #444; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas,monospace; font-size: 11px; }
pre { background: #f7f7f7; padding: 10px; border-radius: 5px; overflow-x: auto; }
img { max-width: 100%; display: block; margin: 10px auto;
      border: 1px solid #eee; border-radius: 4px; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 11px; }
th,td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
th { background: #eef7f1; }
hr { border: none; border-top: 1px solid #ddd; margin: 18px 0; }
strong { color: #b1400a; }
</style>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="observations.md")
    ap.add_argument("--out", default="observations.pdf")
    args = ap.parse_args()

    md_path = os.path.abspath(args.md)
    md_dir = os.path.dirname(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{CSS}</head><body>{html_body}</body></html>"
    print("embedding images...")
    html = embed_images(html, md_dir)

    tmp_html = os.path.join(md_dir, "_export_tmp.html")
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(html)

    from playwright.sync_api import sync_playwright
    out_path = os.path.abspath(args.out)
    print("rendering PDF via Chromium...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("file:///" + tmp_html.replace("\\", "/"))
        page.wait_for_timeout(500)
        page.pdf(path=out_path, format="A4", print_background=True,
                 margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
        browser.close()
    os.remove(tmp_html)
    print(f"Saved PDF -> {out_path} ({os.path.getsize(out_path)//1024} KB)")


if __name__ == "__main__":
    main()
