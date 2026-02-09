"""
Taxi/ride-hailing receipt layout with fully randomized text.
"""

import random
from typing import Dict
from PIL import Image

from .base_layout import (
    BaseLayout, LayoutConfig, LayoutType, FontManager,
    generate_random_text, generate_random_number_string
)


class TaxiReceiptLayout(BaseLayout):
    """Taxi/ride-hailing receipt layout."""

    def __init__(self):
        config = LayoutConfig(
            layout_type=LayoutType.TAXI_RECEIPT,
            width_range=(300, 380),
            height_range=(450, 650),
            margin=20,
            line_spacing=1.5,
        )
        super().__init__(config)

    def render(self, data: Dict) -> Image.Image:
        """Render taxi receipt."""
        self._init_canvas((255, 255, 255))

        title_font = FontManager.get_font("sans", 18, bold=True)
        header_font = FontManager.get_font("sans", 13, bold=True)
        body_font = FontManager.get_font("sans", 11)
        small_font = FontManager.get_font("sans", 9)

        black = (30, 30, 30)
        gray = (120, 120, 120)
        brand_color = random.choice([
            (0, 171, 102),
            (255, 201, 60),
            (0, 170, 90),
        ])

        # Header
        app_name = generate_random_text(4, 12)
        tw, _ = self._text_size(app_name, title_font)
        self._draw_text(app_name, (self.width - tw) // 2, self.y_cursor, title_font, brand_color)
        self._advance_y(font=title_font)

        trip_type = generate_random_text(5, 12)
        tw, _ = self._text_size(trip_type, body_font)
        self._draw_text(trip_type, (self.width - tw) // 2, self.y_cursor, body_font, gray)
        self._advance_y(font=body_font)
        self._advance_y(15)

        # Trip ID
        trip_label = generate_random_text(5, 10)
        trip_id = f"{generate_random_text(3, 6).upper()}{generate_random_number_string(6)}"
        date = data.get("date", generate_random_number_string(12))
        
        self._draw_text(f"{trip_label}: {trip_id}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        
        time_label = generate_random_text(5, 10)
        self._draw_text(f"{time_label}: {date}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="solid", color=brand_color)
        self._advance_y(15)

        # Route
        pickup_label = generate_random_text(6, 12)
        dropoff_label = generate_random_text(6, 12)
        
        pickup_point = generate_random_text(15, 35)
        dropoff_point = generate_random_text(15, 35)

        self._draw_text(f"{pickup_label}:", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(pickup_point, self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(5)
        self._draw_text(f"{dropoff_label}:", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)
        self._draw_text(dropoff_point, self.margin, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Trip details
        distance = round(random.uniform(2, 25), 1)
        duration = random.randint(10, 60)

        distance_label = generate_random_text(6, 15)
        duration_unit = generate_random_text(3, 6)

        self._draw_text(f"{distance_label}: {distance} km", self.margin, self.y_cursor, body_font, black)
        # Separate duration components
        # Original: self._draw_text(f"{duration} {duration_unit}", self.width - self.margin - tw, self.y_cursor, body_font, black)
        
        s_duration = f"{duration}"
        s_unit = f"{duration_unit}"
        
        w_dur, _ = self._text_size(s_duration, body_font)
        w_dunit, _ = self._text_size(s_unit, body_font)
        
        gap = 10
        full_w = w_dur + gap + w_dunit
        start_x = self.width - self.margin - full_w
        
        self._draw_text(s_duration, start_x, self.y_cursor, body_font, black)
        self._draw_text(s_unit, start_x + w_dur + gap, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        self._advance_y(10)
        self._draw_line(self.y_cursor, style="dashed", color=gray)
        self._advance_y(15)

        # Fare breakdown
        base_fare = random.choice([10000, 12000, 15000])
        per_km = random.choice([8000, 10000, 12000, 15000])
        distance_fare = int(distance * per_km)
        promo = random.choice([0, 0, -10000, -15000, -20000])
        total = base_fare + distance_fare + promo

        base_label = generate_random_text(6, 12)
        fare_label = generate_random_text(5, 10)
        promo_label = generate_random_text(6, 12)

        self._draw_text(f"{base_label}:", self.margin, self.y_cursor, small_font, gray)
        bf_str = self._format_currency(base_fare)
        tw, _ = self._text_size(bf_str, body_font)
        self._draw_text(bf_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        # Separate distance in label
        # Original: self._draw_text(f"{fare_label} ({distance} km):", self.margin, self.y_cursor, small_font, gray)
        
        self._draw_text(f"{fare_label} (", self.margin, self.y_cursor, small_font, gray)
        w_p1, _ = self._text_size(f"{fare_label} (", small_font)
        
        s_dist = f"{distance}"
        self._draw_text(s_dist, self.margin + w_p1, self.y_cursor, small_font, gray)
        w_dist, _ = self._text_size(s_dist, small_font)
        
        # Add gap before ' km):' ?? Actually usually it's "2.5 km". So a space is expected.
        # But user wants separation. So I will add a small gap.
        gap = 5
        self._draw_text(" km):", self.margin + w_p1 + w_dist + gap, self.y_cursor, small_font, gray)
        df_str = self._format_currency(distance_fare)
        tw, _ = self._text_size(df_str, body_font)
        self._draw_text(df_str, self.width - self.margin - tw, self.y_cursor, body_font, black)
        self._advance_y(font=body_font)

        if promo < 0:
            self._draw_text(f"{promo_label}:", self.margin, self.y_cursor, small_font, gray)
            p_str = self._format_currency(promo)
            tw, _ = self._text_size(p_str, body_font)
            self._draw_text(p_str, self.width - self.margin - tw, self.y_cursor, body_font, brand_color)
            self._advance_y(font=body_font)

        self._advance_y(5)
        self._draw_line(self.y_cursor, style="solid", color=black)
        self._advance_y(10)

        # Total
        total_label = generate_random_text(6, 12)
        self._draw_text(total_label, self.margin, self.y_cursor, header_font, black)
        tot_str = self._format_currency(total)
        tw, _ = self._text_size(tot_str, header_font)
        self._draw_text(tot_str, self.width - self.margin - tw, self.y_cursor, header_font, brand_color)
        self._advance_y(font=header_font)

        # Payment
        self._advance_y(15)
        payment_label = generate_random_text(8, 15)
        payment = generate_random_text(5, 12)
        self._draw_text(f"{payment_label}: {payment}", self.margin, self.y_cursor, small_font, gray)
        self._advance_y(font=small_font)

        # Driver info
        self._advance_y(10)
        driver_label = generate_random_text(5, 10)
        driver = generate_random_text(4, 10)
        plate = f"{generate_random_number_string(2)}{random.choice(['A', 'B', 'C'])}-{generate_random_number_string(3)}.{generate_random_number_string(2)}"
        self._draw_text(f"{driver_label}: {driver} • {plate}", self.margin, self.y_cursor, small_font, gray)

        return self.img