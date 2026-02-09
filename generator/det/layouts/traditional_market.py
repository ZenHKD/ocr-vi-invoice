"""
Traditional market receipt layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class TraditionalMarketLayout(BaseLayout):
    """Traditional market receipt - minimal, often handwritten style."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.TRADITIONAL_MARKET,
            width_range=(300, 450),
            height_range=(400, 700),
            margin=15,
            line_spacing=1.7,
            font_sizes={
                "body": (14, 20),
                "small": (10, 14),
            },
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render traditional market receipt."""
        bg = random.choice([
            (255, 255, 245),
            (250, 248, 240),
            (255, 255, 255),
        ])
        self._init_canvas(bg)

        body_font = FontManager.get_font("handwritten", random.randint(*self.config.font_sizes["body"]))
        small_font = FontManager.get_font("handwritten", random.randint(*self.config.font_sizes["small"]))

        black = (0, 0, 0)
        gray = (80, 80, 80)

        # Simple header
        store_name = generate_random_text(5, 15)
        self._draw_text(store_name, self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        # Date
        date = data.get("date", generate_random_number_string(8))
        self._draw_text(date, self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(10)

        # Items - very simple format
        items = data.get("items", [])

        for item in items:
            desc = generate_random_text(5, 15)
            qty = item.get("qty", random.randint(1, 10))
            total = item.get("total", random.randint(5000, 50000))

            price_str = self._format_currency(total)
            # Separate qty
            # Original: line = f"{desc} ({qty}): {price_str}"
            
            # Draw (qty) as a single block separate from desc and price
            self._draw_text(desc, self.margin, self.y_cursor, body_font, black)
            w_desc, _ = self._text_size(desc, body_font)
            
            # Gap after description
            gap1 = 12
            x = self.margin + w_desc + gap1
            
            # Single block for (qty)
            s_qty_block = f"({qty})"
            self._draw_text(s_qty_block, x, self.y_cursor, body_font, black)
            w_block, _ = self._text_size(s_qty_block, body_font)
            x += w_block
            
            gap2 = 5
            x += gap2
            self._draw_text(":", x, self.y_cursor, body_font, black)
            w_colon, _ = self._text_size(":", body_font)
            x += w_colon
            
            # Gap before price
            gap3 = 10
            x += gap3
            self._draw_text(price_str, x, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

        # Simple line
        self._advance_y(10)
        self.draw.line(
            (self.margin, self.y_cursor, self.width - self.margin, self.y_cursor),
            fill=black, width=1
        )
        self._advance_y(10)

        # Total
        grand = data.get("grand_total", sum(item.get("total", 0) for item in items))
        total_label = generate_random_text(4, 8)
        total_str = self._format_currency(grand)
        total_text = f"{total_label}: {total_str}"
        self._draw_text(total_text, self.margin, self.y_cursor, body_font, black)

        return self.img