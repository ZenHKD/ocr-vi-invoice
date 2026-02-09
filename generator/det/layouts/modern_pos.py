"""
Modern POS receipt layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class ModernPOSLayout(BaseLayout):
    """Modern digital POS receipt layout (Square/Shopify style)."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.MODERN_POS,
            width_range=(320, 400),
            height_range=(450, 750),
            margin=25,
            line_spacing=1.7,
            font_sizes={
                "header": (16, 22),
                "body": (11, 14),
                "footer": (9, 11),
            },
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render modern POS receipt."""
        self._init_canvas((255, 255, 255))

        header_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["header"]), bold=True)
        body_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["body"]))
        small_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["footer"]))

        black = (40, 40, 40)
        gray = (140, 140, 140)
        accent = random.choice([(0, 122, 255), (52, 199, 89), (255, 149, 0)])

        # === HEADER ===
        store_name = generate_random_text(5, 12).upper()
        tw, _ = self._text_size(store_name, header_font)
        self._draw_text(store_name, (self.width - tw) // 2, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)

        # Tagline
        if random.random() < 0.7:
            tagline = generate_random_text(10, 25)
            tw, _ = self._text_size(tagline, small_font)
            self._draw_text(tagline, (self.width - tw) // 2, self.y_cursor, small_font, gray)
            self._advance_y(font=small_font)

        self._advance_y(15)

        # Order info
        order_label = generate_random_text(4, 8)
        inv_num = f"#{generate_random_number_string(4)}"
        date = data.get("date", generate_random_number_string(8))

        self._draw_text(f"{order_label} {inv_num}", self.margin, self.y_cursor, body_font, black)
        tw, _ = self._text_size(date, small_font)
        self._draw_text(date, self.width - self.margin - tw, self.y_cursor, small_font, gray)
        self._advance_y(font=body_font)

        self._advance_y(10)
        self.draw.line((self.margin, self.y_cursor, self.width - self.margin, self.y_cursor), fill=gray, width=1)
        self._advance_y(15)

        # === ITEMS ===
        items = data.get("items", [])

        for item in items:
            desc = generate_random_text(5, 20)
            qty = item.get("qty", random.randint(1, 5))
            total = item.get("total", random.randint(10000, 500000))

            if qty == 1:
                self._draw_text(desc, self.margin, self.y_cursor, body_font, black)
            else:
                s_qty = f"{qty}"
                s_rest = f" × {desc}"
                
                w_qty, _ = self._text_size(s_qty, body_font)
                self._draw_text(s_qty, self.margin, self.y_cursor, body_font, black)
                self._draw_text(s_rest, self.margin + w_qty + 12, self.y_cursor, body_font, black)

            price_str = self._format_currency(total)
            tw, _ = self._text_size(price_str, body_font)
            self._draw_text(price_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)
            self._advance_y(3)

        self._advance_y(10)
        self.draw.line((self.margin, self.y_cursor, self.width - self.margin, self.y_cursor), fill=gray, width=1)
        self._advance_y(15)

        # === TOTALS ===
        subtotal = data.get("subtotal", sum(item.get("total", 0) for item in items))
        vat = data.get("vat", int(subtotal * 0.1))
        grand = data.get("grand_total", subtotal + vat)

        def draw_summary_line(label, value, bold=False):
            font = header_font if bold else body_font
            color = black if bold else gray
            self._draw_text(label, self.margin, self.y_cursor, font, color)
            val_str = self._format_currency(value)
            tw, _ = self._text_size(val_str, font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, font, black)
            self._advance_y(font=font)

        subtotal_label = generate_random_text(5, 10)
        draw_summary_line(subtotal_label, subtotal)
        if vat > 0:
            tax_label = generate_random_text(3, 6)
            draw_summary_line(tax_label, vat)

        self._advance_y(5)
        total_label = generate_random_text(4, 8)
        draw_summary_line(total_label, grand, bold=True)

        # Payment method
        self._advance_y(15)
        payment_label = generate_random_text(6, 12)
        payment = generate_random_text(4, 10)
        self._draw_text(f"{payment_label} {payment}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        # QR Code placeholder
        if random.random() < 0.4:
            self._advance_y(20)
            qr_size = 60
            qr_x = (self.width - qr_size) // 2
            self.draw.rectangle(
                (qr_x, self.y_cursor, qr_x + qr_size, self.y_cursor + qr_size),
                outline=gray, width=1
            )
            for _ in range(20):
                x = qr_x + random.randint(5, qr_size - 5)
                y = self.y_cursor + random.randint(5, qr_size - 5)
                self.draw.rectangle((x, y, x + 3, y + 3), fill=black)
            self._advance_y(qr_size + 10)

        return self.img