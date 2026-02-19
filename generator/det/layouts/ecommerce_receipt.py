"""
E-commerce receipt layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class EcommerceReceiptLayout(BaseLayout):
    """Online shopping/e-commerce receipt layout."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.ECOMMERCE_RECEIPT,
            width_range=(500, 650),
            height_range=(650, 1000),
            margin=35,
            line_spacing=1.8,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render e-commerce receipt."""
        self._init_canvas((255, 255, 255))

        logo_font = FontManager.get_font("sans", 24, bold=True)
        header_font = FontManager.get_font("sans", 18, bold=True)
        body_font = FontManager.get_font("sans", 14)
        small_font = FontManager.get_font("sans", 12)

        brand_color = random.choice([(255, 153, 0), (0, 112, 255), (76, 175, 80)])
        black = (40, 40, 40)
        gray = (140, 140, 140)

        # Store logo/name
        store_name = generate_random_text(6, 15)
        tw, _ = self._text_size(store_name, logo_font)
        self._draw_text(store_name, (self.width - tw) // 2, self.y_cursor, logo_font, brand_color)
        self._advance_y(font=logo_font)

        # Tagline
        tagline = generate_random_text(15, 35)
        tw, _ = self._text_size(tagline, small_font)
        self._draw_text(tagline, (self.width - tw) // 2, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._advance_y(20)

        # Order info
        order_label = generate_random_text(6, 12)
        order_num = f"{generate_random_text(2, 4).upper()}-{generate_random_number_string(10)}"
        date_label = generate_random_text(5, 10)
        date = data.get("date", generate_random_number_string(12))

        self._draw_text(f"{order_label}: {order_num}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._draw_text(f"{date_label}: {date}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        # Customer
        customer_label = generate_random_text(6, 12)
        customer = generate_random_text(8, 20)
        self._advance_y(10)
        self._draw_text(f"{customer_label}: {customer}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(15)
        self._draw_line(self.y_cursor, style="solid", color=brand_color)
        self._advance_y(15)

        # Products
        products_label = generate_random_text(8, 15).upper()
        self._draw_text(products_label, self.margin, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)
        self._advance_y(10)

        items = data.get("items", [])
        subtotal = 0

        # Randomly choose between bordered table and plain list
        if random.random() < 0.45:
            # === BORDERED TABLE ===
            table_headers = [
                generate_random_text(6, 12),
                generate_random_text(3, 6),
                generate_random_text(5, 10),
            ]
            col_widths = [
                self.width - 2 * self.margin - 60 - 120,
                60,
                120,
            ]
            rows = []
            for item in items:
                name = generate_random_text(10, 35)
                qty = item.get("qty", random.randint(1, 5))
                total = item.get("total", random.randint(50000, 2000000))
                subtotal += total
                rows.append([name, f"{qty}x", self._format_currency(total)])
            
            border_style = random.choice(["full", "no_vertical", "outer_only"])
            self._draw_table_with_borders(
                headers=table_headers, rows=rows,
                col_widths=col_widths, font=body_font, header_font=header_font,
                color=black, border_color=random.choice([(40, 40, 40), (140, 140, 140)]),
                draw_vertical_lines=border_style == "full",
                draw_horizontal_lines=border_style in ("full", "outer_only"),
            )
        else:
            # === PLAIN LIST (original) ===
            for item in items:
                name = generate_random_text(10, 35)
                sku = f"{generate_random_text(2, 3).upper()}-{generate_random_number_string(6)}"
                qty = item.get("qty", random.randint(1, 5))
                total = item.get("total", random.randint(50000, 2000000))
                subtotal += total

                self._draw_text(name, self.margin, self.y_cursor, body_font, black)
                self._advance_y(font=body_font)
                s_qty_x = f"{qty}x"
                self._draw_text(s_qty_x, self.margin, self.y_cursor, small_font, gray)
                val_str = self._format_currency(total)
                tw, _ = self._text_size(val_str, body_font)
                self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
                self._advance_y(font=body_font)
                self._advance_y(5)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)


        # Summary
        shipping_label = generate_random_text(8, 15)
        shipping = random.choice([0, 20000, 30000, 50000])
        discount_label = generate_random_text(8, 15)
        discount = random.choice([0, -30000, -50000, -100000])
        total = subtotal + shipping + discount

        subtotal_label = generate_random_text(7, 14)
        self._draw_text(f"{subtotal_label}:", self.margin, self.y_cursor, small_font, gray)
        tw, _ = self._text_size(self._format_currency(subtotal), body_font)
        self._draw_text(self._format_currency(subtotal), self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        if shipping > 0:
            self._draw_text(f"{shipping_label}:", self. margin, self.y_cursor, small_font, gray)
            tw, _ = self._text_size(self._format_currency(shipping), body_font)
            self._draw_text(self._format_currency(shipping), self.width - self.margin - tw, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

        if discount < 0:
            self._draw_text(f"{discount_label}:", self.margin, self.y_cursor, small_font, gray)
            tw, _ = self._text_size(self._format_currency(discount), body_font)
            self._draw_text(self._format_currency(discount), self.width - self.margin - tw, self.y_cursor, body_font, brand_color)
            self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="double", color=black)
        self._advance_y(15)

        # Total
        total_label = generate_random_text(6, 12).upper()
        self._draw_text(total_label, self.margin, self.y_cursor, header_font, black)
        total_str = self._format_currency(total)
        tw, _ = self._text_size(total_str, header_font)
        self._draw_text(total_str, self.width - self.margin - tw, self.y_cursor, header_font, brand_color)
        self._advance_y(font=header_font)

        # Payment method
        self._advance_y(20)
        payment_label = generate_random_text(8, 15)
        payment = generate_random_text(6, 15)
        self._draw_text(f"{payment_label}: {payment}", self.margin, self.y_cursor, small_font, gray)

        return self.img
