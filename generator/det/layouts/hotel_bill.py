"""
Hotel bill layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class HotelBillLayout(BaseLayout):
    """Hotel/hospitality bill layout."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.HOTEL_BILL,
            width_range=(550, 700),
            height_range=(700, 1100),
            margin=40,
            line_spacing=1.7,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render hotel bill."""
        self._init_canvas((255, 255, 255))

        title_font = FontManager.get_font("serif", 24, bold=True)
        header_font = FontManager.get_font("serif", 16, bold=True)
        body_font = FontManager.get_font("sans", 12)
        small_font = FontManager.get_font("sans", 10)

        black = (0, 0, 0)
        gray = (100, 100, 100)
        gold = (184, 134, 11)

        # Hotel name
        hotel_name = generate_random_text(8, 20)
        tw, _ = self._text_size(hotel_name, title_font)
        self._draw_text(hotel_name, (self.width - tw) // 2, self.y_cursor, title_font, gold)
        self._advance_y(font=title_font)

        # Address
        address = generate_random_text(20, 50)
        tw, _ = self._text_size(address, small_font)
        self._draw_text(address, (self.width - tw) // 2, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._advance_y(20)

        # Guest info
        guest_label = generate_random_text(5, 10)
        guest = generate_random_text(8, 20)
        room_label = generate_random_text(4, 8)
        room = f"{random.randint(100, 999)}"

        self._draw_text(f"{guest_label}: {guest}", self.margin, self.y_cursor, body_font, black)
        tw, _ = self._text_size(f"{room_label}: {room}", body_font)
        self._draw_text(f"{room_label}: {room}", self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        # Dates
        checkin_label = generate_random_text(5, 10)
        checkout_label = generate_random_text(5, 10)
        checkin = generate_random_number_string(8)
        checkout = generate_random_number_string(8)

        self._draw_text(f"{checkin_label}: {checkin}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(f"{checkout_label}: {checkout}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(15)
        self._draw_line(self.y_cursor, style="double", color=gold)
        self._advance_y(20)

        # Charges
        charges_header = generate_random_text(10, 20)
        self._draw_text(charges_header, self.margin, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)
        self._advance_y(10)

        items = data.get("items", [])
        total_amount = 0

        # Randomly choose between bordered table and plain list
        if random.random() < 0.45:
            # === BORDERED TABLE ===
            table_headers = [
                generate_random_text(6, 12),
                generate_random_text(3, 6),
                generate_random_text(6, 10),
            ]
            col_widths = [
                self.width - 2 * self.margin - 80 - 120,
                80,
                120,
            ]
            rows = []
            for item in items:
                desc = generate_random_text(8, 25)
                nights = item.get("qty", random.randint(1, 7))
                amount = item.get("total", random.randint(500000, 3000000))
                total_amount += amount
                rows.append([desc, f"({nights})", self._format_currency(amount)])
            
            border_style = random.choice(["full", "no_vertical", "outer_only"])
            self._draw_table_with_borders(
                headers=table_headers, rows=rows,
                col_widths=col_widths, font=body_font, header_font=header_font,
                color=black, border_color=random.choice([black, gray]),
                draw_vertical_lines=border_style == "full",
                draw_horizontal_lines=border_style in ("full", "outer_only"),
            )
        else:
            # === PLAIN LIST (original) ===
            for item in items:
                desc = generate_random_text(8, 25)
                nights = item.get("qty", random.randint(1, 7))
                amount = item.get("total", random.randint(500000, 3000000))
                total_amount += amount

                amt_str = self._format_currency(amount)
                self._draw_text(desc, self.margin, self.y_cursor, body_font, black)
                w_desc, _ = self._text_size(desc, body_font)
                gap = 10 
                curr_x = self.margin + w_desc + gap
                s_nights_block = f"({nights})"
                self._draw_text(s_nights_block, curr_x, self.y_cursor, body_font, black)
                tw, _ = self._text_size(amt_str, body_font)
                self._draw_text(amt_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
                self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)


        # Subtotal, service, VAT
        subtotal = data.get("subtotal", total_amount)
        service_charge = int(subtotal * 0.05)
        vat = int((subtotal + service_charge) * 0.1)
        grand_total = subtotal + service_charge + vat

        subtotal_label = generate_random_text(7, 14)
        service_label = generate_random_text(10, 20)
        vat_label = generate_random_text(6, 12)

        self._draw_text(f"{subtotal_label}:", self.margin, self.y_cursor, body_font, gray)
        s_str = self._format_currency(subtotal)
        tw, _ = self._text_size(s_str, body_font)
        self._draw_text(s_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._draw_text(f"{service_label} (5%):", self.margin, self.y_cursor, body_font, gray)
        v_str = self._format_currency(service_charge)
        tw, _ = self._text_size(v_str, body_font)
        self._draw_text(v_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._draw_text(f"{vat_label} (10%):", self.margin, self.y_cursor, body_font, gray)
        va_str = self._format_currency(vat)
        tw, _ = self._text_size(va_str, body_font)
        self._draw_text(va_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="double", color=black)
        self._advance_y(15)

        # Grand total
        total_label = generate_random_text(8, 16).upper()
        self._draw_text(f"{total_label}:", self.margin, self.y_cursor, header_font, black)
        grand_str = self._format_currency(grand_total)
        tw, _ = self._text_size(grand_str, header_font)
        self._draw_text(grand_str, self.width - self.margin - tw, self.y_cursor, header_font, gold)
        self._advance_y(font=header_font)

        # Payment
        self._advance_y(20)
        payment_label = generate_random_text(8, 15)
        payment = generate_random_text(5, 12)
        self._draw_text(f"{payment_label}: {payment}", self.margin, self.y_cursor, small_font, gray)

        return self.img
