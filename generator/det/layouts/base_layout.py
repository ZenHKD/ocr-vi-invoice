"""
Base layout classes and utilities for invoice generation.
All text is randomized using vocabulary from model.rec.vocab
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Import vocabulary for text randomization
import sys
import os
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from model.rec.vocab import VOCAB


class LayoutType(Enum):
    SUPERMARKET_THERMAL = "supermarket_thermal"
    FORMAL_VAT = "formal_vat"
    RESTAURANT_BILL = "restaurant_bill"
    TRADITIONAL_MARKET = "traditional_market"
    MODERN_POS = "modern_pos"
    CAFE_MINIMAL = "cafe_minimal"
    HANDWRITTEN = "handwritten"
    DELIVERY_RECEIPT = "delivery_receipt"
    HOTEL_BILL = "hotel_bill"
    UTILITY_BILL = "utility_bill"
    ECOMMERCE_RECEIPT = "ecommerce_receipt"
    TAXI_RECEIPT = "taxi_receipt"


@dataclass
class LayoutConfig:
    """Configuration for a layout."""
    layout_type: LayoutType
    width_range: Tuple[int, int]
    height_range: Tuple[int, int]
    margin: int = 20
    line_spacing: float = 1.2
    font_sizes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    features: Dict[str, bool] = field(default_factory=dict)


# Text randomization utilities
def generate_random_text(min_length: int = 5, max_length: int = 20) -> str:
    """Generate random text using vocabulary characters."""
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(VOCAB) for _ in range(length))


def generate_random_number_string(length: int = 6) -> str:
    """Generate random number as string."""
    return ''.join(random.choice('0123456789') for _ in range(length))


def generate_random_phone() -> str:
    """Generate random phone number format."""
    return f"0{generate_random_number_string(9)}"


def generate_random_label(min_len: int = 2, max_len: int = 8) -> str:
    """Generate random label text."""
    return generate_random_text(min_len, max_len)


class FontManager:
    """Manage fonts for rendering text."""

    _font_paths = {}
    _font_cache = {}
    _font_dir = Path(__file__).parent.parent.parent.parent / "synthetic_data" / "fonts"

    @classmethod
    def _scan_fonts(cls):
        """Dynamically scan for fonts if not already loaded."""
        if cls._font_paths:
            return

        patterns = {
            "sans": ["formal/*/*-Regular.ttf", "formal/*/*-Medium.ttf"],
            "sans_bold": ["formal/*/*-Bold.ttf"],
            "serif": ["formal/*/*-Regular.ttf", "formal/*/*-Medium.ttf"],
            "serif_bold": ["formal/*/*-Bold.ttf"],
            "mono": ["thermal_printer/*/*-Regular.ttf", "dot_matrix/*/*-Regular.ttf"],
            "handwritten": ["handwritten/*/*-Regular.ttf"],
            "thermal": ["thermal_printer/*/*-Regular.ttf", "dot_matrix/*/*-Regular.ttf"],
        }

        base_dir = Path(__file__).parent.parent.parent / "synthetic_data" / "fonts"
        if not base_dir.exists():
            # Try alternative path
            base_dir = project_root / "synthetic_data" / "fonts"

        for category, glob_patterns in patterns.items():
            cls._font_paths[category] = []
            for pattern in glob_patterns:
                found = list(base_dir.glob(pattern))
                valid_fonts = []
                for p in found:
                    if cls._font_supports_vietnamese(str(p)):
                        valid_fonts.append(str(p))
                cls._font_paths[category].extend(valid_fonts)

            if not cls._font_paths[category]:
                cat_dir = base_dir / category.split("_")[0]
                if cat_dir.exists():
                    candidates = list(cat_dir.glob("**/*.ttf"))
                    valid_fallbacks = [str(p) for p in candidates if cls._font_supports_vietnamese(str(p))]
                    cls._font_paths[category].extend(valid_fallbacks)

    @classmethod
    def _font_supports_vietnamese(cls, font_path: str) -> bool:
        """Check if font supports basic Vietnamese characters."""
        try:
            font = ImageFont.truetype(font_path, 20)
            test_chars = ["ế", "ộ", "ơ", "ư", "ắ", "ậ", "đ"]
            
            import fontTools.ttLib
            tt = fontTools.ttLib.TTFont(font_path)
            cmap = tt['cmap']
            tables = cmap.getBestCmap()
            
            if not tables:
                return False
                
            for char in test_chars:
                if ord(char) not in tables:
                    return False
                    
            return True
        except Exception:
            return False

    @classmethod
    def get_font(cls, family: str = "sans", size: int = 14, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get a font instance."""
        cls._scan_fonts()
        
        key = f"{family}_{size}_{bold}"
        
        if key in cls._font_cache:
            return cls._font_cache[key]
            
        paths = cls._font_paths.get(family, [])
        if not paths:
            if family.endswith("_bold"):
                fallback = family.replace("_bold", "")
                paths = cls._font_paths.get(fallback, [])

        if not paths:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size)
            except:
                font = ImageFont.load_default()
            cls._font_cache[key] = font
            return font
            
        font_path = random.choice(paths)
        
        try:
            font = ImageFont.truetype(font_path, size)
            cls._font_cache[key] = font
            return font
        except Exception as e:
            print(f"Error loading font {font_path}: {e}")
            font = ImageFont.load_default()
            cls._font_cache[key] = font
            return font

    @classmethod
    def get_random_font(cls, size: int = 14, style: str = "any") -> ImageFont.FreeTypeFont:
        """Get a random font."""
        if style == "any":
            family = random.choice(["sans", "serif", "mono"])
        else:
            family = style
        return cls.get_font(family, size)


class BaseLayout:
    """Base class for invoice layouts."""

    def __init__(self, config: LayoutConfig):
        self.config = config
        self.width = random.randint(*config.width_range)
        self.height = random.randint(*config.height_range)
        self.margin = config.margin
        self.y_cursor = self.margin
        self.img = None
        self.draw = None
        self.rendered_text = []
        self.currency_format_style = "standard"

    def _init_canvas(self, bg_color: Tuple[int, int, int] = (255, 255, 255)):
        """Initialize the canvas."""
        self.img = Image.new("RGB", (self.width, self.height), bg_color)
        self.draw = ImageDraw.Draw(self.img)
        self.rendered_text = []
        
        if random.random() < 0.5:
            self.currency_format_style = "none"
        else:
            self.currency_format_style = random.choice(["standard", "symbol", "symbol_clean"])
            
    def _format_currency(self, value: float) -> str:
        """Format currency with random style (number only, or with đ/₫ symbol)."""
        val_int = int(value)
        val_str = f"{val_int:,}"
        
        if self.currency_format_style == "none":
            return val_str
        elif self.currency_format_style == "standard":
            return f"{val_str}đ"
        elif self.currency_format_style == "symbol":
            return f"{val_str}₫"
        elif self.currency_format_style == "symbol_clean":
            return f"{val_str} ₫"
        else:
            return f"{val_str}đ"

    def _text_size(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        """Get text dimensions."""
        if font is None:
            return (len(text) * 8, 16)
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            return (len(text) * 8, 16)

    def _draw_text(self, text: str, x: int, y: int, font: ImageFont.FreeTypeFont,
                   color: Tuple[int, int, int] = (0, 0, 0), max_width: Optional[int] = None):
        """Draw text with optional truncation and track polygon annotation."""
        text = text.replace('\n', ' ')
        if max_width:
            while self._text_size(text, font)[0] > max_width and len(text) > 3:
                text = text[:-4] + "..."
        self.draw.text((x, y), text, font=font, fill=color, anchor="lt")
        
        if text.strip():
            w, h = self._text_size(text, font)
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            self.rendered_text.append({
                "text": text.strip(),
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            })

    def _draw_line(self, y: int, style: str = "solid", color: Tuple[int, int, int] = (0, 0, 0)):
        """Draw a horizontal line."""
        x1, x2 = self.margin, self.width - self.margin

        if style == "solid":
            self.draw.line((x1, y, x2, y), fill=color, width=1)
        elif style == "dashed":
            dash_len = 10
            for x in range(x1, x2, dash_len * 2):
                self.draw.line((x, y, min(x + dash_len, x2), y), fill=color, width=1)
        elif style == "dotted":
            for x in range(x1, x2, 5):
                self.draw.point((x, y), fill=color)
        elif style == "double":
            self.draw.line((x1, y - 2, x2, y - 2), fill=color, width=1)
            self.draw.line((x1, y + 2, x2, y + 2), fill=color, width=1)

    def _draw_table_with_borders(self, headers: List[str], rows: List[List[str]],
                                  col_widths: List[int], font: ImageFont.FreeTypeFont,
                                  header_font: ImageFont.FreeTypeFont = None,
                                  color: Tuple[int, int, int] = (0, 0, 0),
                                  border_color: Tuple[int, int, int] = (0, 0, 0),
                                  border_width: int = 1,
                                  cell_padding: int = 4,
                                  draw_outer_border: bool = True,
                                  draw_vertical_lines: bool = True,
                                  draw_horizontal_lines: bool = True):
        """
        Draw a table WITH visible border lines. Borders are NOT annotated (visual only).
        Only text cells are annotated as detection targets.
        
        This teaches the model that table borders (|, _, grid lines) are NOT text.
        """
        if header_font is None:
            header_font = font
        
        x_start = self.margin
        table_width = sum(col_widths)
        _, row_height_ref = self._text_size("Ag", font)
        row_height = row_height_ref + cell_padding * 2
        
        total_rows = 1 + len(rows)  # header + data
        table_height = total_rows * row_height
        
        y_top = self.y_cursor
        
        # === Draw outer border (visual only, NOT annotated) ===
        if draw_outer_border:
            self.draw.rectangle(
                [x_start, y_top, x_start + table_width, y_top + table_height],
                outline=border_color, width=border_width
            )
        
        # === Draw header row ===
        x = x_start
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            # Draw text (annotated)
            text_x = x + cell_padding
            text_y = y_top + cell_padding
            self._draw_text(header, text_x, text_y, header_font, color)
            
            # Draw vertical line AFTER this column (visual only)
            if draw_vertical_lines and i < len(headers) - 1:
                self.draw.line(
                    [x + width, y_top, x + width, y_top + table_height],
                    fill=border_color, width=border_width
                )
            x += width
        
        # Draw horizontal line under header
        header_bottom_y = y_top + row_height
        if draw_horizontal_lines:
            self.draw.line(
                [x_start, header_bottom_y, x_start + table_width, header_bottom_y],
                fill=border_color, width=border_width
            )
        
        # === Draw data rows ===
        for row_idx, row_data in enumerate(rows):
            row_y = header_bottom_y + row_idx * row_height
            
            x = x_start
            for col_idx, (cell_text, width) in enumerate(zip(row_data, col_widths)):
                text_x = x + cell_padding
                text_y = row_y + cell_padding
                self._draw_text(cell_text, text_x, text_y, font, color, max_width=width - cell_padding * 2)
                x += width
            
            # Draw horizontal line under this row (visual only)
            if draw_horizontal_lines and row_idx < len(rows) - 1:
                line_y = row_y + row_height
                self.draw.line(
                    [x_start, line_y, x_start + table_width, line_y],
                    fill=border_color, width=border_width
                )
        
        # Advance y cursor past the table
        self.y_cursor = y_top + table_height + 5

    def _advance_y(self, amount: int = None, font: ImageFont.FreeTypeFont = None):
        """Advance the y cursor."""
        if amount:
            self.y_cursor += amount
        elif font:
            self.y_cursor += int(self._text_size("A", font)[1] * self.config.line_spacing)
        else:
            self.y_cursor += 20

    def render(self, data: Dict) -> Image.Image:
        """Render the invoice. Override in subclasses."""
        raise NotImplementedError

    def get_ocr_annotations(self) -> List[Dict]:
        """Get all rendered text with polygons for OCR ground truth."""
        return self.rendered_text

    def get_ocr_text(self) -> List[str]:
        """Get just text strings (legacy compatibility)."""
        return [item["text"] for item in self.rendered_text]

    def _crop_to_content(self) -> Image.Image:
        """Crop image to actual content."""
        return self.img.crop((0, 0, self.width, min(self.y_cursor + 20, self.height)))
