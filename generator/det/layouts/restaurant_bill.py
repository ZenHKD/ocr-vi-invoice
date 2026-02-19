"""
Restaurant bill layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class RestaurantBillLayout(BaseLayout):
    """Restaurant dine-in bill layout with table number and service charge."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.RESTAURANT_BILL,
            width_range=(350, 450),
            height_range=(500, 900),
            margin=20,
            line_spacing=1.8,
            font_sizes={
                "header": (18, 24),
                "body": (12, 16),
                "footer": (10, 12),
            },
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render restaurant bill."""
        bg = random.choice([
            (255, 255, 255),
            (255, 252, 245),
            (248, 248, 245),
        ])
        self._init_canvas(bg)

        header_font = FontManager.get_font("serif", random.randint(*self.config.font_sizes["header"]), bold=True)
        body_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["body"]))
        small_font = FontManager.get_font("sans", random.randint(*self.config.font_sizes["footer"]))

        black = (0, 0, 0)
        gray = (100, 100, 100)
        red = (180, 50, 50)

        # === HEADER ===
        store_name = generate_random_text(8, 20).upper()
        tw, _ = self._text_size(store_name, header_font)
        self._draw_text(store_name, (self.width - tw) // 2, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)

        # Address
        address = generate_random_text(15, 40)
        tw, _ = self._text_size(address, small_font)
        self._draw_text(address, (self.width - tw) // 2, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="double", color=black)
        self._advance_y(15)

        # Table number and waiter
        table_label = generate_random_text(3, 6)
        table_num = random.randint(1, 30)
        waiter_label = generate_random_text(2, 4)
        waiter = generate_random_text(3, 8)

        self._draw_text(f"{table_label}: {table_num}", self.margin, self.y_cursor, body_font, red)
        tw, _ = self._text_size(f"{waiter_label}: {waiter}", body_font)
        self._draw_text(f"{waiter_label}: {waiter}", self.width - self.margin - tw, self.y_cursor, body_font, gray)
        self._advance_y(font=body_font)

        # Date
        date_label = generate_random_text(3, 6)
        date = data.get("date", generate_random_number_string(8))
        self._draw_text(f"{date_label}: {date}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(10)

        # === ITEMS ===
        items = data.get("items", [])

        # Randomly choose between bordered table and plain list
        if random.random() < 0.45:
            # === BORDERED TABLE ===
            table_headers = [
                generate_random_text(3, 6),
                generate_random_text(6, 12),
                generate_random_text(5, 10),
            ]
            col_widths = [
                50,
                self.width - 2 * self.margin - 50 - 100,
                100,
            ]
            rows = []
            for item in items:
                desc = generate_random_text(5, 25)
                qty = item.get("qty", random.randint(1, 5))
                total = item.get("total", random.randint(20000, 200000))
                rows.append([f"{qty}", desc, self._format_currency(total)])
            
            border_style = random.choice(["full", "no_vertical", "no_horizontal"])
            self._draw_table_with_borders(
                headers=table_headers, rows=rows,
                col_widths=col_widths, font=body_font, header_font=body_font,
                color=black, border_color=random.choice([black, gray]),
                draw_vertical_lines=border_style == "full",
                draw_horizontal_lines=border_style in ("full", "no_vertical"),
            )
        else:
            # === PLAIN LIST (original) ===
            for item in items:
                desc = generate_random_text(5, 25)
                qty = item.get("qty", random.randint(1, 5))
                total = item.get("total", random.randint(20000, 200000))

                s_qty = f"{qty}"
                s_rest = f"x {desc}"
                w_qty, _ = self._text_size(s_qty, body_font)
                self._draw_text(s_qty, self.margin, self.y_cursor, body_font, black)
                self._draw_text(s_rest, self.margin + w_qty + 12, self.y_cursor, body_font, black)
                price_str = self._format_currency(total)
                tw, _ = self._text_size(price_str, body_font)
                self._draw_text(price_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
                self._advance_y(font=body_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)


        # === TOTALS ===
        subtotal = data.get("subtotal", sum(item.get("total", 0) for item in items))
        vat = data.get("vat", int(subtotal * 0.1))
        grand = data.get("grand_total", subtotal + vat)

        # Service charge
        service_rate = random.choice([0, 5, 10])
        service_charge = int(subtotal * service_rate / 100)

        def draw_total_line(label, value, font=body_font, color=black):
            self._draw_text(label, self.margin, self.y_cursor, font, color)
            val_str = self._format_currency(value)
            tw, _ = self._text_size(val_str, font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, font, color)
            self._advance_y(font=font)

        subtotal_label = generate_random_text(5, 10)
        draw_total_line(f"{subtotal_label}:", subtotal, small_font, gray)
        
        if service_rate > 0:
            service_label = generate_random_text(6, 12)
            draw_total_line(f"{service_label} ({service_rate}%):", service_charge, small_font, gray)
        if vat > 0:
            vat_label = generate_random_text(3, 6)
            draw_total_line(f"{vat_label}:", vat, small_font, gray)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="double", color=black)
        self._advance_y(10)

        # Grand total
        total_with_service = grand + service_charge
        total_label = generate_random_text(6, 12).upper()
        self._draw_text(f"{total_label}:", self.margin, self.y_cursor, header_font, black)
        grand_str = self._format_currency(total_with_service)
        tw, _ = self._text_size(grand_str, header_font)
        self._draw_text(grand_str, self.width - self.margin - tw, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)

        # Footer
        self._advance_y(20)
        footer_msg = generate_random_text(12, 25)
        tw, _ = self._text_size(footer_msg, small_font)
        self._draw_text(footer_msg, (self.width - tw) // 2, self.y_cursor, small_font, gray)

        return self.img