"""
Food delivery receipt layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class DeliveryReceiptLayout(BaseLayout):
    """Food delivery app receipt layout."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.DELIVERY_RECEIPT,
            width_range=(360, 450),
            height_range=(500, 850),
            margin=20,
            line_spacing=1.5,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render delivery receipt."""
        self._init_canvas((255, 255, 255))

        brand_font = FontManager.get_font("sans", 20, bold=True)
        header_font = FontManager.get_font("sans", 14, bold=True)
        body_font = FontManager.get_font("sans", 11)
        small_font = FontManager.get_font("sans", 9)

        brand_color = random.choice([(230, 0, 0), (0, 150, 136), (255, 87, 34)])
        black = (30, 30, 30)
        gray = (130, 130, 130)

        # App name
        app_name = generate_random_text(5, 12)
        tw, _ = self._text_size(app_name, brand_font)
        self._draw_text(app_name, (self.width - tw) // 2, self.y_cursor, brand_font, brand_color)
        self._advance_y(font=brand_font)
        self._advance_y(15)

        # Order ID
        order_label = generate_random_text(5, 10)
        order_id = f"#{generate_random_text(2, 4).upper()}{generate_random_number_string(6)}"
        date = data.get("date", generate_random_number_string(12))

        self._draw_text(f"{order_label} {order_id}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._draw_text(date, self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="solid", color=brand_color)
        self._advance_y(15)

        # Restaurant/Store
        store_label = generate_random_text(6, 12)
        store = generate_random_text(10, 25)
        self._draw_text(f"{store_label}:", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(store, self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        # Address
        address = generate_random_text(20, 45)
        addr_label = generate_random_text(6, 12)
        self._advance_y(5)
        self._draw_text(f"{addr_label}:", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(address, self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Items
        items_label = generate_random_text(6, 12)
        self._draw_text(items_label, self.margin, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)
        self._advance_y(5)

        items = data.get("items", [])
        total_items = 0

        for item in items:
            desc = generate_random_text(8, 25)
            qty = item.get("qty", random.randint(1, 5))
            total = item.get("total", random.randint(25000, 200000))
            total_items += total

            # Separate qty for standalone detection
            s_qty = f"{qty}"
            s_rest = f"x {desc}"
            
            w_qty, _ = self._text_size(s_qty, body_font)
            self._draw_text(s_qty, self.margin, self.y_cursor, body_font, black)
            # Increase gap
            self._draw_text(s_rest, self.margin + w_qty + 12, self.y_cursor, body_font, black)
            val_str = self._format_currency(total)
            tw, _ = self._text_size(val_str, body_font)
            self._draw_text(val_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Fees
        subtotal = data.get("subtotal", total_items)
        delivery_fee = random.choice([10000, 15000, 20000, 0])
        discount = random.choice([0, -10000, -15000, -20000])
        total = subtotal + delivery_fee + discount

        subtotal_label = generate_random_text(6, 12)
        delivery_label = generate_random_text(8, 18)
        discount_label = generate_random_text(8, 15)

        self._draw_text(f"{subtotal_label}:", self.margin, self.y_cursor, small_font, gray)
        tw, _ = self._text_size(self._format_currency(subtotal), body_font)
        self._draw_text(self._format_currency(subtotal), self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        if delivery_fee > 0:
            self._draw_text(f"{delivery_label}:", self.margin, self.y_cursor, small_font, gray)
            tw, _ = self._text_size(self._format_currency(delivery_fee), body_font)
            self._draw_text(self._format_currency(delivery_fee), self.width - self.margin - tw, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

        if discount < 0:
            self._draw_text(f"{discount_label}:", self.margin, self.y_cursor, small_font, gray)
            tw, _ = self._text_size(self._format_currency(discount), body_font)
            self._draw_text(self._format_currency(discount), self.width - self.margin - tw, self.y_cursor, body_font, brand_color)
            self._advance_y(font=body_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)

        # Total
        total_label = generate_random_text(5, 10).upper()
        self._draw_text(total_label, self.margin, self.y_cursor, header_font, black)
        total_str = self._format_currency(total)
        tw, _ = self._text_size(total_str, header_font)
        self._draw_text(total_str, self.width - self.margin - tw, self.y_cursor, header_font, brand_color)

        return self.img
