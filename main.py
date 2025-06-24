from pathlib import Path
import typer
from src.utils.color_quantizer import quantize_image
from src.utils.mask_builder import masks_from_quantized
from src.utils.svg_exporter import masks_to_svgs
from src.utils.utils import ensure_out_dir


def make_stencils(
    *,
    input_path: Path,
    output_dir: Path,
    max_colors: int,
    scale: int,
):
    """Convert *input_path* PNG into ≤ *max_colors* SVG layers in *output_dir*."""
    ensure_out_dir(output_dir)

    typer.secho(f"Quantizing image", fg=typer.colors.WHITE)
    img_quant, palette = quantize_image(input_path, max_colors)
    typer.secho(f"Creating masks", fg=typer.colors.WHITE)
    masks = masks_from_quantized(img_quant)
    typer.secho(f"Creating SVGS", fg=typer.colors.WHITE)
    masks_to_svgs(masks, palette, output_dir, scale)
    typer.secho(f"✅  Wrote {len(palette)} SVG layer(s) to «{output_dir}»", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    input_dir = Path("./art/pokemon-gold/test")

    total = 0
    for i, input_path in enumerate(input_dir.glob("*.png")):
        typer.secho(f"Making stencils for #{i+1}")
        file_stem = input_path.stem
        output_dir = Path(f"./art/pokemon-gold/svgs/{file_stem}")
        make_stencils(
            input_path=Path(input_path),
            output_dir=output_dir,
            max_colors=4,
            scale=10,
        )
        total += 1

    typer.secho(f"✅ Converted {total} PNGS -> SVGs", fg=typer.colors.GREEN, bold=True)
