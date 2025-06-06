from pathlib import Path
import typer
from src.utils.color_quantization import quantize_image
from src.utils.mask_builder import masks_from_quantized
from src.utils.svg_exporter import masks_to_svgs
from src.utils.utils import ensure_out_dir

app = typer.Typer(help="Generate colour-layer SVG stencils from pixel art PNGs.")


@app.command()
def make_stencils(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Source PNG"),
    output_dir: Path = typer.Option(
        Path("./stencils"), "--output-dir", "-o", help="Where SVGs are written"
    ),
    max_colors: int = typer.Option(
        4, "--max-colors", "-n", min=2, max=12, help="Palette size cap"
    ),
    scale: int = typer.Option(
        10, "--scale", "-s", help="Multiply pixel size in final SVG"
    ),
):
    """Convert *input_path* PNG into ≤ *max_colors* SVG layers in *output_dir*."""
    ensure_out_dir(output_dir)

    img_quant, palette = quantize_image(input_path, max_colors)
    masks = masks_from_quantized(img_quant, palette)
    masks_to_svgs(masks, palette, output_dir, scale)
    typer.secho(f"✅  Wrote {len(palette)} SVG layer(s) to «{output_dir}»", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    app()
