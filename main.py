from pathlib import Path
import typer
from src.utils.color_quantizer import quantize_image
from src.utils.mask_builder import masks_from_quantized
from src.utils.svg_exporter import masks_to_svgs
from src.utils.utils import ensure_out_dir
from src.models import POKEMON_NAMES


def make_stencils(
    *,
    input_path: Path,
    output_dir: Path,
    base_filename: str,
    max_colors: int,
    scale: int,
):
    """Convert *input_path* PNG into ≤ *max_colors* SVG layers in *output_dir*."""
    ensure_out_dir(output_dir)

    typer.secho(f"Quantizing image", fg=typer.colors.WHITE)
    label_map, rgba_palette = quantize_image(input_path, max_colors)
    typer.secho(f"Creating masks", fg=typer.colors.WHITE)
    masks = masks_from_quantized(label_map, rgba_palette)
    typer.secho(f"Creating SVGS", fg=typer.colors.WHITE)
    masks_to_svgs(masks, rgba_palette, output_dir, scale, base_filename)
    typer.secho(f"✅  Wrote {len(rgba_palette)} SVG layer(s) to «{output_dir}»", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    input_dir = Path("./art/pokemon-gold/test")

    total = 0
    for i, input_path in enumerate(input_dir.glob("*.png")):
        typer.secho(f"Making stencils for #{i+1}")
        file_stem = input_path.stem
        num = int(file_stem.split("_")[-1])
        name = POKEMON_NAMES.get(num, f"pokemon{num}")
        base_filename = f"{num:03}_{name}"
        output_dir = Path(f"./art/pokemon-gold/svgs/{base_filename}")
        make_stencils(
            input_path=Path(input_path),
            output_dir=output_dir,
            base_filename=base_filename,
            max_colors=4,
            scale=10,
        )
        total += 1

    typer.secho(f"✅ Converted {total} PNGS -> SVGs", fg=typer.colors.GREEN, bold=True)

