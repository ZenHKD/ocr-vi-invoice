"""
Supermarket thermal receipt layout with fully randomized text.
"""

import random
from typing import Dict, Tuple
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string, generate_random_phone
)


class ThermalReceiptLayout(BaseLayout):
    """Supermarket thermal receipt layout."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.SUPERMARKET_THERMAL,
            width_range=(280, 380),
            height_range=(600, 1400),
            margin=10,
            line_spacing=1.8,
            font_sizes={
            "header": (14, 20),
            "body": (10, 14),
            "footer": (8, 12),
            },
            features={
            "barcode": True,
            "qr_code": random.random() < 0.3,
            "logo_placeholder": random.random() < 0.4,
            }
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render thermal receipt."""
        bg = random.choice([
            (255, 255, 255),
            (252, 250, 245),
            (248, 248, 248),
            (255, 253, 245),
        ])
        self._init_canvas(bg)

        header_size = random.randint(*self.config.font_sizes["header"])
        body_size = random.randint(*self.config.font_sizes["body"])
        footer_size = random.randint(*self.config.font_sizes["footer"])

        header_font = FontManager.get_font("mono", header_size)
        body_font = FontManager.get_font("mono", body_size)
        footer_font = FontManager.get_font("mono", footer_size)

        text_color = random.choice([
            (0, 0, 0),
            (30, 30, 30),
            (50, 50, 50),
            (20, 10, 10),
        ])

        # === HEADER ===
        store_name = generate_random_text(5, 15)
        tw, th = self._text_size(store_name, header_font)
        self._draw_text(store_name, (self.width - tw) // 2, self.y_cursor, header_font, text_color)
        self._advance_y(font=header_font)

        # Address
        address = generate_random_text(20, 40)
        words = address.split()
        if not words:
            words = [generate_random_text(5, 10) for _  in range(random.randint(3, 6))]
        
        lines = []
        current = ""
        for word in words:
            test = current + " " + word if current else word
            if self._text_size(test, footer_font)[0] < self.width - 2 * self.margin:
                current = test
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        for line in lines[:2]:
            tw, _ = self._text_size(line, footer_font)
            self._draw_text(line, (self.width - tw) // 2, self.y_cursor, footer_font, text_color)
            self._advance_y(font=footer_font)

        # Phone
        phone = generate_random_phone()
        tw, _ = self._text_size(phone, footer_font)
        self._draw_text(phone, (self.width - tw) // 2, self.y_cursor, footer_font, text_color)
        self._advance_y(font=footer_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="dashed", color=text_color)
        self._advance_y(10)

        # Invoice number and date
        inv_label = generate_random_text(2, 4)
        inv_num = generate_random_number_string(6)
        date = data.get("date", generate_random_number_string(8))

        self._draw_text(f"{inv_label}: {inv_num}", self.margin, self.y_cursor, footer_font, text_color)
        date_w, _ = self._text_size(date, footer_font)
        self._draw_text(date, self.width - self.margin - date_w, self.y_cursor, footer_font, text_color)
        self._advance_y(font=footer_font)

        self._draw_line(self.y_cursor, style="dashed", color=text_color)
        self._advance_y(8)

        # === ITEMS ===
        items = data.get("items", [])
        for item in items:
            desc = generate_random_text(5, 20)
            qty = item.get("qty", random.randint(1, 5))
            unit_price = item.get("unit", random.randint(10000, 100000))
            total = item.get("total", qty * unit_price)

            self._draw_text(desc, self.margin, self.y_cursor, body_font, text_color)
            self._advance_y(font=body_font)

            # Separate qty for standalone detection
            # Original: line = f"  {qty} x {u_str} = {t_str}"
            
            u_str = self._format_currency(unit_price)
            t_str = self._format_currency(total)
            
            s_qty = f"{qty}"
            s_rest = f"x {u_str} = {t_str}"
            
            # Calculate full width for alignment
            w_qty, _ = self._text_size(s_qty, footer_font)
            w_rest, _ = self._text_size(s_rest, footer_font)
            
            gap = 15 # increased spacing
            
            total_w = gap + w_qty + gap + w_rest
            start_x = self.width - self.margin - total_w
            
            # Draw
            self._draw_text(s_qty, start_x + gap, self.y_cursor, footer_font, text_color)
            self._draw_text(s_rest, start_x + gap + w_qty + gap, self.y_cursor, footer_font, text_color)
            self._advance_y(font=footer_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="dashed", color=text_color)
        self._advance_y(8)

        # === TOTALS ===
        subtotal = data.get("subtotal", sum(item.get("total", 0) for item in items))
        vat_rate = data.get("vat_rate", random.choice([0, 8, 10]))
        vat = data.get("vat", int(subtotal * vat_rate / 100))
        grand = data.get("grand_total", subtotal + vat)

        def draw_total_line(label: str, value: float):
            self._draw_text(label, self.margin, self.y_cursor, body_font, text_color)
            val_str = self._format_currency(value)
            tw, _ = self._text_size(val_str, body_font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, body_font, text_color)
            self._advance_y(font=body_font)

        sum_label = generate_random_text(3, 8)
        draw_total_line(f"{sum_label}:", subtotal)
        if vat_rate > 0:
            vat_label = generate_random_text(3, 6)
            draw_total_line(f"{vat_label} ({vat_rate}%):", vat)

        self._draw_line(self.y_cursor, style="solid", color=text_color)
        self._advance_y(5)

        # Grand total
        total_label = generate_random_text(5, 12)
        self._draw_text(f"{total_label}:", self.margin, self.y_cursor, header_font, text_color)
        grand_str = self._format_currency(grand)
        tw, _ = self._text_size(grand_str, header_font)
        self._draw_text(grand_str, self.width - self.margin - tw, self.y_cursor, header_font, text_color)
        self._advance_y(font=header_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="dashed", color=text_color)
        self._advance_y(10)

        # Payment method
        payment_label = generate_random_text(5, 10)
        payment_method = generate_random_text(4, 10)
        self._draw_text(f"{payment_label}: {payment_method}", self.margin, self.y_cursor, footer_font, text_color)
        self._advance_y(font=footer_font)

        # Footer message
        self._advance_y(15)
        msg = generate_random_text(10, 25)
        tw, _ = self._text_size(msg, body_font)
        self._draw_text(msg, (self.width - tw) // 2, self.y_cursor, body_font, text_color)

        # Barcode placeholder
        if self.config.features.get("barcode") and random.random() < 0.6:
            self._advance_y(25)
            barcode = generate_random_number_string(13)
            self._draw_barcode(barcode, text_color)

        return self._crop_to_content()

    def _draw_barcode(self, code: str, color: Tuple[int, int, int]):
        """Draw a simple barcode representation."""
        x = self.margin + 10
        y = self.y_cursor
        bar_height = 40

        for i, char in enumerate(code):
            digit = int(char)
            bar_width = 2 + (digit % 3)
            if i % 2 == 0:
                self.draw.rectangle((x, y, x + bar_width, y + bar_height), fill=color)
            x += bar_width + 1

        self.y_cursor += bar_height + 5

        font = FontManager.get_font("mono", 10)
        tw, _ = self._text_size(code, font)
        self._draw_text(code, (self.width - tw) // 2, self.y_cursor, font, color)
        self._advance_y(15)