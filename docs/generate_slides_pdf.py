import subprocess
import sys
from pathlib import Path


def generate_slides_pdf(
    input_md="docs/slides.md",
    output_pdf="docs/slides.pdf",
    output_html="docs/slides.html",
):
    """
    Compiles Marp presentation markdown into presentation-ready PDF and HTML slides.
    """
    root_dir = Path(__file__).resolve().parent.parent
    md_path = root_dir / input_md
    pdf_path = root_dir / output_pdf
    html_path = root_dir / output_html

    if not md_path.exists():
        print(f"Error: Slide markdown file not found at {md_path}")
        sys.exit(1)

    print(f"Compiling presentation slides from: {md_path}")

    # 1. Compile PDF
    cmd_pdf = [
        "npx",
        "-y",
        "@marp-team/marp-cli",
        str(md_path),
        "--pdf",
        "--allow-local-files",
        "-o",
        str(pdf_path),
    ]
    print(f"Running: {' '.join(cmd_pdf)}")
    res_pdf = subprocess.run(cmd_pdf, cwd=root_dir)
    if res_pdf.returncode != 0:
        print("Failed to generate PDF slides.")
        sys.exit(res_pdf.returncode)
    print(f"Successfully generated PDF slides at: {pdf_path}")

    # 2. Compile HTML
    cmd_html = [
        "npx",
        "-y",
        "@marp-team/marp-cli",
        str(md_path),
        "-o",
        str(html_path),
    ]
    print(f"Running: {' '.join(cmd_html)}")
    res_html = subprocess.run(cmd_html, cwd=root_dir)
    if res_html.returncode == 0:
        print(f"Successfully generated HTML interactive presentation at: {html_path}")


if __name__ == "__main__":
    generate_slides_pdf()
