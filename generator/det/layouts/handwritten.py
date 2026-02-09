"""
Handwritten receipt layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class HandwrittenLayout(BaseLayout):
    """Simulated handwritten receipt/note."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.HANDWRITTEN,
            width_range=(400, 600),
            height_range=(500, 800),
            margin=30,
            line_spacing=2.1,
            font_sizes={
                "body": (16, 24),
            },
            features={
                "lined_paper": random.random() < 0.5,
                "crossed_out": random.random() < 0.2,
            }
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render handwritten receipt."""
        bg = random.choice([
            (255, 255, 240),
            (255, 250, 230),
            (245, 245, 245),
            (255, 255, 255),
        ])
        self._init_canvas(bg)

        # Draw lines if lined paper
        if self.config.features.get("lined_paper"):
            line_color = (200, 200, 220)
            for y in range(50, self.height, 30):
                self.draw.line((20, y, self.width - 20, y), fill=line_color, width=1)

        font = FontManager.get_font("handwritten", random.randint(*self.config.font_sizes["body"]))
        ink_color = random.choice([
            (0, 0, 100),
            (0, 0, 0),
            (50, 50, 50),
        ])

        def jitter():
            return random.randint(-3, 3)

        # Header
        store = generate_random_text(5, 15)
        self._draw_text(store, self.margin + jitter(), self.y_cursor + jitter(), font, ink_color)
        self._advance_y(font=font)

        # Date
        date_label = generate_random_text(3, 6)
        date = data.get("date", generate_random_number_string(8))
        self._draw_text(f"{date_label}: {date}", self.margin + jitter(), self.y_cursor + jitter(), font, ink_color)
        self._advance_y(font=font)

        self._advance_y(10)

        # Items with casual formatting
        items = data.get("items", [])

        for item in items:
            desc = generate_random_text(5, 15)
            qty = item.get("qty", random.randint(1, 5))
            total = item.get("total", random.randint(10000, 100000))

            t_str = self._format_currency(total)
            # Separate qty for standalone detection
            # Original: line = f"{desc} x{qty} = {t_str}"
            
            x_pos = self.margin + jitter()
            y_pos = self.y_cursor + jitter()
            
            gap = 10 # Explicit spacing gap
            
            # Draw desc
            self._draw_text(desc, x_pos, y_pos, font, ink_color)
            w_desc, _ = self._text_size(desc, font)
            
            # Draw " x" after gap
            self._draw_text(" x", x_pos + w_desc + gap, y_pos, font, ink_color)
            w_x, _ = self._text_size(" x", font)
            
            # Draw qty standalone after gap
            s_qty = f"{qty}"
            self._draw_text(s_qty, x_pos + w_desc + gap + w_x + gap, y_pos, font, ink_color)
            w_qty, _ = self._text_size(s_qty, font)
            
            # Draw rest
            s_tail = f" = {t_str}"
            self._draw_text(s_tail, x_pos + w_desc + gap + w_x + gap + w_qty, y_pos, font, ink_color)
            self._advance_y(font=font)

        # Underline before total
        self._advance_y(5)
        self.draw.line(
            (self.margin, self.y_cursor, self.width - self.margin, self.y_cursor),
            fill=ink_color, width=2
        )
        self._advance_y(10)

        # Grand total
        grand = data.get("grand_total", sum(item.get("total", 0) for item in items))
        total_label = generate_random_text(4, 8)
        total_text = f"{total_label}: {self._format_currency(grand)}"

        self._draw_text(total_text, self.margin + jitter(), self.y_cursor + jitter(), font, ink_color)

        return self.img