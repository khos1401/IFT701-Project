from dataclasses import dataclass
from typing import Optional, Tuple, Union
from io import BytesIO

from PIL import Image, ImageOps  # pip install pillow


@dataclass
class ImageOpsConfig:
    # One of these two ways to shrink:
    max_size: Optional[Tuple[int, int]] = None  # e.g. (1024, 1024) keeps aspect, fits within box
    scale: Optional[float] = None               # e.g. 0.5 to halve width/height

    # One of these two ways to crop:
    crop_box: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom) in pixels
    center_crop_size: Optional[Tuple[int, int]] = None    # e.g. (512, 512)


class JpgToPngProcessor:
    """
    Load a JPG, reduce its resolution, optionally crop, and output PNG.
    """

    def __init__(self, jpg_path: str):
        self.jpg_path = jpg_path

    def process(
        self,
        config: ImageOpsConfig,
        output_path: Optional[str] = None,
        return_bytes: bool = True,
    ) -> Union[bytes, None]:
        """
        Args:
            config: ImageOpsConfig with resize/crop settings.
            output_path: If provided, saves PNG here (e.g., 'out.png').
            return_bytes: If True, returns PNG bytes. If False and output_path
                          is given, returns None.

        Returns:
            PNG bytes if return_bytes=True, else None (after saving).
        """
        # 1) Load and normalize orientation
        with Image.open(self.jpg_path) as im:
            # Ensure RGB (some JPGs can be L/CMYK) and apply EXIF orientation
            im = ImageOps.exif_transpose(im.convert("RGB"))

            # 2) Reduce resolution
            im = self._resize(im, config)

            # 3) Crop
            im = self._crop(im, config)

            # 4) Encode to PNG (save or return bytes)
            if output_path:
                im.save(output_path, format="PNG", optimize=True)
                if not return_bytes:
                    return None

            buf = BytesIO()
            im.save(buf, format="PNG", optimize=True)
            return buf.getvalue()

    # ---------- helpers ----------

    def _resize(self, im: Image.Image, config: ImageOpsConfig) -> Image.Image:
        if config.scale is not None and config.max_size is not None:
            raise ValueError("Specify either scale or max_size, not both.")

        if config.scale is not None:
            if config.scale <= 0:
                raise ValueError("scale must be > 0.")
            w, h = im.size
            new_size = (max(1, int(w * config.scale)), max(1, int(h * config.scale)))
            return im.resize(new_size, Image.Resampling.LANCZOS)

        if config.max_size is not None:
            # thumbnail modifies in-place, keeping aspect and fitting within box
            im_copy = im.copy()
            im_copy.thumbnail(config.max_size, Image.Resampling.LANCZOS)
            return im_copy

        # No resize requested
        return im

    def _crop(self, im: Image.Image, config: ImageOpsConfig) -> Image.Image:
        if config.crop_box and config.center_crop_size:
            raise ValueError("Specify either crop_box or center_crop_size, not both.")

        if config.crop_box:
            l, t, r, b = config.crop_box
            # Clamp to image bounds
            l = max(0, min(l, im.width))
            t = max(0, min(t, im.height))
            r = max(l, min(r, im.width))
            b = max(t, min(b, im.height))
            if r <= l or b <= t:
                raise ValueError("Invalid crop_box after clamping.")
            return im.crop((l, t, r, b))

        if config.center_crop_size:
            cw, ch = config.center_crop_size
            if cw <= 0 or ch <= 0:
                raise ValueError("center_crop_size must be positive.")
            x_c = im.width // 2
            y_c = im.height // 2
            l = max(0, x_c - cw // 2)
            t = max(0, y_c - ch // 2)
            r = min(im.width, l + cw)
            b = min(im.height, t + ch)

            # If requested crop is larger than image, pad by resizing first or error
            if r - l <= 0 or b - t <= 0:
                raise ValueError("center_crop_size is larger than the image.")
            return im.crop((l, t, r, b))

        # No crop requested
        return im

