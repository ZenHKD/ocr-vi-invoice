"""
Cafe minimal receipt layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text
)


class CafeMinimalLayout(BaseLayout):
    """Modern minimalist cafe receipt."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.CAFE_MINIMAL,
            width_range=(300, 400),
            height_range=(400, 700),
            margin=25,
            line_spacing=1.7,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render minimal cafe receipt."""
        self._init_canvas((255, 255, 255))

        title_font = FontManager.get_font("sans", 22)
        body_font = FontManager.get_font("sans", 13)
        small_font = FontManager.get_font("sans", 10)

        black = (0, 0, 0)
        gray = (120, 120, 120)

        # Centered store name
        store = generate_random_text(5, 15).upper()
        tw, _ = self._text_size(store, title_font)
        self._draw_text(store, (self.width - tw) // 2, self.y_cursor, title_font, black)
        self._advance_y(font=title_font)
        self._advance_y(20)

        # Simple line
        self.draw.line(
            (self.width // 4, self.y_cursor, 3 * self.width // 4, self.y_cursor),
            fill=gray, width=1
        )
        self._advance_y(20)

        # Items
        items = data.get("items", [])
        for item in items:
            desc = generate_random_text(5, 15)
            qty = item.get("qty", random.randint(1, 3))
            total = item.get("total", random.randint(20000, 100000))

            # Separate qty for standalone detection
            # Original: self._draw_text(f"{qty}x {desc}", self.margin, self.y_cursor, body_font, black)
            s_qty = f"{qty}"
            s_rest = f"x {desc}"
            
            w_qty, _ = self._text_size(s_qty, body_font)
            self._draw_text(s_qty, self.margin, self.y_cursor, body_font, black)
            # Increase gap
            self._draw_text(s_rest, self.margin + w_qty + 12, self.y_cursor, body_font, black)

            price_str = self._format_currency(total)
            tw, _ = self._text_size(price_str, body_font)
            self._draw_text(price_str, self.width - self.margin - tw, self.y_cursor, body_font, black)

            self._advance_y(font=body_font)

        self._advance_y(15)
        self.draw.line(
            (self.margin, self.y_cursor, self.width - self.margin, self.y_cursor),
            fill=gray, width=1
        )
        self._advance_y(15)

        # Total
        grand = data.get("grand_total", sum(item.get("total", 0) for item in items))
        total_label = generate_random_text(4, 8).upper()
        self._draw_text(total_label, self.margin, self.y_cursor, title_font, black)
        total_str = self._format_currency(grand)
        tw, _ = self._text_size(total_str, title_font)
        self._draw_text(total_str, self.width - self.margin - tw, self.y_cursor, title_font, black)

        self._advance_y(font=title_font)
        self._advance_y(30)

        # Thank you centered
        thanks = f"{generate_random_text(5, 12)} • {generate_random_text(5, 12)}"
        tw, _ = self._text_size(thanks, small_font)
        self._draw_text(thanks, (self.width - tw) // 2, self.y_cursor, small_font, gray)

        return self.img