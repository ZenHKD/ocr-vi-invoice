"""
Utility bill layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class UtilityBillLayout(BaseLayout):
    """Utility bill layout (electricity, water, internet)."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.UTILITY_BILL,
            width_range=(600, 750),
            height_range=(700, 1000),
            margin=30,
            line_spacing=1.7,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render utility bill."""
        self._init_canvas((255, 255, 255))

        title_font = FontManager.get_font("sans", 22, bold=True)
        header_font = FontManager.get_font("sans", 14, bold=True)
        body_font = FontManager.get_font("sans", 12)
        small_font = FontManager.get_font("sans", 10)

        black = (0, 0, 0)
        gray = (100, 100, 100)
        blue = (0, 80, 160)

        # Utility type
        company = generate_random_text(10, 25)
        short_name = generate_random_text(3, 6).upper()
        bill_title = generate_random_text(12, 25)

        # Header
        tw, _ = self._text_size(company, title_font)
        self._draw_text(company, (self.width - tw) // 2, self.y_cursor, title_font, blue)
        self._advance_y(font=title_font)

        tw, _ = self._text_size(bill_title, header_font)
        self._draw_text(bill_title, (self.width - tw) // 2, self.y_cursor, header_font, black)
        self._advance_y(font=header_font)
        self._advance_y(15)

        self._draw_line(self.y_cursor, style="solid", color=blue)
        self._advance_y(15)

        # Customer info
        customer = generate_random_text(10, 20)
        customer_id = f"{short_name}{generate_random_number_string(8)}"

        cust_label = generate_random_text(7, 15)
        id_label = generate_random_text(4, 8)

        self._draw_text(f"{cust_label}: {customer}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._draw_text(f"{id_label}: {customer_id}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        address = generate_random_text(20, 50)
        addr_label = generate_random_text(5, 10)
        self._draw_text(f"{addr_label}: {address}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._advance_y(15)

        # Billing period
        period_label = generate_random_text(8, 18)
        month = random.choice([f"{i:02d}" for i in range(1, 13)])
        year = "2026"
        period_text = generate_random_text(4, 8)
        self._draw_text(f"{period_label}: {period_text} {month}/{year}", self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)
        self._advance_y(10)

        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Usage details
        usage_header = generate_random_text(10, 20)
        self._draw_text(usage_header, self.margin, self.y_cursor, header_font, blue)
        self._advance_y(font=header_font)
        self._advance_y(10)

        # Generate realistic usage
        old_reading = random.randint(10000, 50000)
        new_reading = old_reading + random.randint(100, 500)
        usage = new_reading - old_reading
        unit = generate_random_text(2, 5)
        rate = random.choice([2000, 2500, 3000])

        old_label = generate_random_text(5, 12)
        new_label = generate_random_text(5, 12)
        usage_label = generate_random_text(5, 10)

        # Randomly choose bordered table for usage details
        if random.random() < 0.4:
            # Bordered table version
            meter_headers = [generate_random_text(6, 12), generate_random_text(5, 10)]
            col_w = [(self.width - 2 * self.margin) // 2] * 2
            rows = [
                [old_label, f"{old_reading:,}"],
                [new_label, f"{new_reading:,}"],
                [usage_label, f"{usage:,} {unit}"],
            ]
            border_style = random.choice(["full", "no_vertical"])
            self._draw_table_with_borders(
                headers=meter_headers, rows=rows,
                col_widths=col_w, font=body_font, header_font=header_font,
                color=black, border_color=gray,
                draw_vertical_lines=border_style == "full",
                draw_horizontal_lines=True,
            )
        else:
            # Plain version (original)
            self._draw_text(f"{old_label}: {old_reading:,}", self.margin, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)
            self._draw_text(f"{new_label}: {new_reading:,}", self.margin, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)
            self._draw_text(f"{usage_label}:", self.margin, self.y_cursor, body_font, black)
            w_p1, _ = self._text_size(f"{usage_label}:", body_font)
            gap = 10
            s_usage = f"{usage:,}"
            self._draw_text(s_usage, self.margin + w_p1 + gap, self.y_cursor, body_font, black)
            w_u, _ = self._text_size(s_usage, body_font)
            s_unit = f"{unit}"
            self._draw_text(s_unit, self.margin + w_p1 + gap + w_u + gap, self.y_cursor, body_font, black)
            self._advance_y(font=body_font)

        amount = usage * rate
        vat = int(amount * 0.1)
        total = amount + vat

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)


        # Totals
        usage_fee_label = generate_random_text(8, 18)
        vat_label = generate_random_text(6, 12)
        
        self._draw_text(f"{usage_fee_label}:", self.margin, self.y_cursor, body_font, black)
        amt_str = self._format_currency(amount)
        tw, _ = self._text_size(amt_str, body_font)
        self._draw_text(amt_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._draw_text(f"{vat_label} (10%):", self.margin, self.y_cursor, body_font, gray)
        vat_str = self._format_currency(vat)
        tw, _ = self._text_size(vat_str, body_font)
        self._draw_text(vat_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)

        total_label = generate_random_text(6, 12).upper()
        self._draw_text(f"{total_label}:", self.margin, self.y_cursor, header_font, black)
        tot_str = self._format_currency(total)
        tw, _ = self._text_size(tot_str, header_font)
        self._draw_text(tot_str, self.width - self.margin - tw, self.y_cursor, header_font, blue)
        self._advance_y(font=header_font)

        # Payment deadline
        self._advance_y(20)
        deadline_label = generate_random_text(8, 18)
        deadline = f"{random.randint(1, 28)}/{month}/{year}"
        self._draw_text(f"{deadline_label}: {deadline}", self.margin, self.y_cursor, body_font, (180, 0, 0))

        return self.img