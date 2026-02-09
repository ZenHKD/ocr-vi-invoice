"""
Formal VAT invoice layout with fully randomized text.
"""

import random
from typing import Dict, List
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class FormalVATLayout(BaseLayout):
    """Formal VAT invoice with structured layout."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.FORMAL_VAT,
            width_range=(800, 1000),
            height_range=(1000, 1400),
            margin=50,
            line_spacing=2.0,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render formal VAT invoice."""
        self._init_canvas((255, 255, 255))

        title_font = FontManager.get_font("serif", 32, bold=True)
        header_font = FontManager.get_font("sans", 20, bold=True)
        body_font = FontManager.get_font("sans", 16)
        small_font = FontManager.get_font("sans", 14)

        black = (0, 0, 0)
        dark_gray = (60, 60, 60)
        gray = (120, 120, 120)
        red = (180, 0, 0)

        # Invoice title
        invoice_title = generate_random_text(10, 20).upper()
        tw, _ = self._text_size(invoice_title, title_font)
        self._draw_text(invoice_title, (self.width - tw) // 2, self.y_cursor, title_font, red)
        self._advance_y(font=title_font)
        self._advance_y(5)

        # Invoice number
        inv_label = generate_random_text(8, 15)
        inv_num = f"{generate_random_text(2, 4).upper()}-{generate_random_number_string(8)}"
        tw, _ = self._text_size(f"{inv_label}: {inv_num}", body_font)
        self._draw_text(f"{inv_label}: {inv_num}", (self.width - tw) // 2, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._advance_y(20)

        # Company info (left) and customer info (right)
        company_label = generate_random_text(8, 15)
        company = generate_random_text(8, 20)
        tax_id = generate_random_number_string(10)
        addr1 = generate_random_text(20, 35)

        self._draw_text(f"{company_label}:", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(company, self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._draw_text(addr1, self.margin, self.y_cursor, small_font, dark_gray)
        self._advance_y(font=small_font)
        tax_label = generate_random_text(6, 12)
        self._draw_text(f"{tax_label}: {tax_id}", self.margin, self.y_cursor, small_font, dark_gray)
        self._advance_y(font=small_font)

        self._advance_y(15)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(15)

        # Date
        date_label = generate_random_text(5, 10)
        date = data.get("date", generate_random_number_string(8))
        self._draw_text(f"{date_label}: {date}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._advance_y(15)

        # Items table header
        col_widths = [80, self.width - 2 * self.margin - 80 - 120 - 120, 120, 120]
        headers = [
            generate_random_text(3, 6),
            generate_random_text(8, 15),
            generate_random_text(6, 12),
            generate_random_text(6, 12)
        ]

        x = self.margin
        for header, width in zip(headers, col_widths):
            self._draw_text(header, x, self.y_cursor, header_font, black)
            x += width
        self._advance_y(font=header_font)

        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(5)

        # Items
        items = data.get("items", [])
        subtotal = 0

        for idx, item in enumerate(items):
            desc = generate_random_text(10, 30)
            qty = item.get("qty", random.randint(1, 10))
            unit_price = item.get("unit", random.randint(50000, 500000))
            total = item.get("total", qty * unit_price)
            subtotal += total

            x = self.margin
            self._draw_text(f"{idx + 1}", x, self.y_cursor, body_font, dark_gray)
            x += col_widths[0]
            self._draw_text(desc, x, self.y_cursor, body_font, black)
            x += col_widths[1]
            self._draw_text(f"{qty}", x, self.y_cursor, body_font, dark_gray)
            x += col_widths[2]
            self._draw_text(self._format_currency(total), x, self.y_cursor, body_font, black)

            self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(15)

        # Totals
        vat_rate = data.get("vat_rate", 10)
        vat = int(subtotal * vat_rate / 100)
        grand = subtotal + vat

        def draw_total_line(label, value, font=body_font, color=black):
            self._draw_text(f"{label}:", self.width - self.margin - 350, self.y_cursor, font, color)
            val_str = self._format_currency(value)
            tw, _ = self._text_size(val_str, font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, font, black)
            self._advance_y(font=font)

        subtotal_label = generate_random_text(7, 14)
        vat_label = generate_random_text(6, 12)
        total_label = generate_random_text(6, 12)

        draw_total_line(subtotal_label, subtotal, body_font, gray)
        draw_total_line(f"{vat_label} ({vat_rate}%)", vat, body_font, gray)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="double", color=black)
        self._advance_y(10)

        draw_total_line(total_label.upper(), grand, header_font, red)

        return self.img
