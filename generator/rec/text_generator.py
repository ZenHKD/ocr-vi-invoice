"""
Text Generator for generating random text images.

Provide by trdg library, that contain these text:
    - Random text (include number, Vietnamese, English, special character)
    - Date
    - Amount (include currency symbol, or pure number)
    - English text
    - Vietnamese text (Vietnamese with and without diacritics)
    - Number (Phone Number, Account Number, code, ...)
    - Code (Invoice Code, Product Code, ...) (include number and character)
"""

import os
import random
import string
from typing import List, Tuple, Generator
from PIL import Image, ImageFont
from faker import Faker
import unicodedata
import time
from datetime import datetime, timedelta

# Monkey patch for Pillow 10+ which removed getsize
if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def getsize(self, text, direction=None, features=None):
        bbox = self.getbbox(text, direction=direction, features=features)
        return bbox[2], bbox[3]
    ImageFont.FreeTypeFont.getsize = getsize

from trdg.generators import GeneratorFromStrings


class TextGenerator:
    """
    A wrapper around trdg.generators.GeneratorFromStrings to simplify text image generation.
    """
    def __init__(
        self,
        language: str = "en",
        count: int = -1,  # -1 for infinite
        fonts: List[str] = None,  # List of font paths, None for default
        blur: int = 0,
        random_blur: bool = True,
        background_type: int = 0,  # 0: Gaussian Noise, 1: Plain White, 2: Quasicrystal, 3: Image
        distorsion_type: int = 0,  # 0: None, 1: Sine wave, 2: Cosine wave, 3: Random (light noise)
        distorsion_orientation: int = 0,
        is_handwritten: bool = False,
        width: int = -1,
        alignment: int = 1,
        text_color: str = "#282828",
        orientation: int = 0,
        space_width: float = 1.0,
        character_spacing: int = 0,
        margins: Tuple[int, int, int, int] = (10, 10, 10, 10),
        fit: bool = True,
        output_mask: bool = False,
        word_split: bool = False,
        image_dir: str = os.path.join("..", "images"),
    ):
        self.language = language
        self.count = count
        
        # Set default fonts that support Vietnamese from local fonts directory
        if fonts is None:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fonts_dir = os.path.join(script_dir, 'fonts')
            
            self.fonts = [
                os.path.join(fonts_dir, "DejaVuSans.ttf"),
                os.path.join(fonts_dir, "LiberationSans-Regular.ttf"),
                os.path.join(fonts_dir, "FreeSans.ttf"),
                os.path.join(fonts_dir, "RobotoMono-Regular.ttf"),
                os.path.join(fonts_dir, "Inconsolata-Regular.ttf"),
                os.path.join(fonts_dir, "VT323-Regular.ttf"),
            ]
        else:
            self.fonts = fonts
            
        self.blur = blur
        self.random_blur = random_blur
        self.background_type = background_type
        self.distorsion_type = distorsion_type
        self.distorsion_orientation = distorsion_orientation
        self.is_handwritten = is_handwritten
        self.width = width
        self.alignment = alignment
        self.text_color = text_color
        self.orientation = orientation
        self.space_width = space_width
        self.character_spacing = character_spacing
        self.margins = margins
        self.fit = fit
        self.output_mask = output_mask
        self.word_split = word_split
        self.image_dir = image_dir

    def generate(self, strings: List[str]) -> Generator[Tuple[Image.Image, str], None, None]:
        """
        Generates images from a list of strings.
        """
        generator = GeneratorFromStrings(
            strings=strings,
            count=len(strings) if self.count == -1 else self.count,
            fonts=self.fonts,
            blur=self.blur,
            random_blur=self.random_blur,
            background_type=self.background_type,
            distorsion_type=self.distorsion_type,
            distorsion_orientation=self.distorsion_orientation,
            is_handwritten=self.is_handwritten,
            width=self.width,
            alignment=self.alignment,
            text_color=self.text_color,
            orientation=self.orientation,
            space_width=self.space_width,
            character_spacing=self.character_spacing,
            margins=self.margins,
            fit=self.fit,
            output_mask=self.output_mask,
            word_split=self.word_split,
            image_dir=self.image_dir,
            language=self.language
        )
        
        return generator

    def generate_random_strings(self, min_length: int = 1, max_length: int = 20, count: int = 100) -> List[str]:
        """
        Generates a list of random alphanumeric strings with special characters including Vietnamese.
        Each string has a random length between min_length and max_length.
        """
        # Ensure minimum length is at least 2 to avoid rendering issues
        min_length = max(2, min_length)
        
        # Vietnamese vowels with tones
        vowels = "aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ"
        # Consonants
        consonants = "bcdđghklmnpqrstvxfjwz"
        # Digits and symbols
        digits = "0123456789"
        currency = "$€£¥₫"  # USD, EUR, GBP, JPY, VND
        symbols = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        
        # Combine all characters (lowercase and uppercase)
        chars = vowels + vowels.upper() + consonants + consonants.upper() + digits + currency + symbols
        alphanumeric = vowels + vowels.upper() + consonants + consonants.upper() + digits
        
        results = []
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loop
        
        while len(results) < count and attempts < max_attempts:
            attempts += 1
            length = random.randint(min_length, max_length)
            text = ''.join(random.choice(chars) for _ in range(length))
            
            # Validate: must have at least 30% alphanumeric characters to avoid whitespace/symbol-only strings
            alnum_count = sum(1 for c in text if c in alphanumeric)
            if alnum_count >= max(1, int(length * 0.3)) and text.strip():  # Not just whitespace
                results.append(text)
        
        # If we couldn't generate enough valid strings, fill with simple alphanumeric
        while len(results) < count:
            length = random.randint(min_length, max_length)
            results.append(''.join(random.choices(alphanumeric, k=length)))
        
        return results

    def generate_date_strings(self, count: int = 100) -> List[str]:
        """
        Generates a list of random date strings in various formats.
        """

        start_date = datetime(2000, 1, 1)
        end_date = datetime(2025, 12, 31)
        delta_days = (end_date - start_date).days
        
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d",
            "%d.%m.%Y", "%B %d, %Y", "%d %b %Y"
        ]
        
        dates = []
        for _ in range(count):
            random_days = random.randrange(delta_days)
            date = start_date + timedelta(days=random_days)
            fmt = random.choice(formats)
            dates.append(date.strftime(fmt))
            
        return dates

    def generate_amount_strings(self, count: int = 100) -> List[str]:
        """
        Generates a list of random amount strings with various currency symbols and formats.
        """
        currencies = ["$", "€", "£", "¥", "VNĐ", "USD", "EUR", "đ", "₫"]
        amounts = []
        
        for _ in range(count):
            amount = random.uniform(10.0, 100000.0)
            currency = random.choice(currencies)
            
            # Format with 0 or 2 decimal places
            if random.random() > 0.5:
                amt_str = f"{amount:,.2f}"
            else:
                amt_str = f"{int(amount):,}"
                
            # Randomly place currency symbol
            if random.random() > 0.5:
                final_str = f"{currency} {amt_str}"
            else:
                final_str = f"{amt_str} {currency}"
            
            amounts.append(final_str)
            
        return amounts

    def generate_english_strings(self, count: int = 100) -> List[str]:
        """
        Generates English text strings.
        """
        fake = Faker('en_US')
        return [fake.sentence() for _ in range(count)]
        

    def generate_vietnamese_strings(self, count: int = 100) -> List[str]:
        """
        Generates Vietnamese text strings (with and without diacritics).
        """
        
        fake = Faker('vi_VN')
        
        def remove_diacritics(text):
            """Remove Vietnamese diacritics from text."""
            # Normalize to NFD (decomposed form)
            nfd = unicodedata.normalize('NFD', text)
            # Filter out combining characters (diacritics)
            without_diacritics = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
            # Handle special Vietnamese characters
            replacements = {
                'đ': 'd', 'Đ': 'D',
            }
            for viet_char, replacement in replacements.items():
                without_diacritics = without_diacritics.replace(viet_char, replacement)
            return without_diacritics
        
        results = []
        for i in range(count):
            sentence = fake.sentence()
            # 50% with diacritics, 50% without
            if i % 2 == 0:
                results.append(sentence)  # With diacritics
            else:
                results.append(remove_diacritics(sentence))  # Without diacritics
        
        return results

    def generate_number_strings(self, min_length: int = 1, max_length: int = 15, count: int = 100) -> List[str]:
        """
        Generates number strings with random lengths (Phone, Account, Code).
        Each number has a random length between min_length and max_length.
        """
        numbers = []
        for _ in range(count):
            # Random length for this number
            length = random.randint(min_length, max_length)
            # Generate number string with the random length
            number = ''.join(random.choices(string.digits, k=length))
            numbers.append(number)
        return numbers

    def generate_code_strings(self, min_length: int = 1, max_length: int = 15, count: int = 100) -> List[str]:
        """
        Generates alphanumeric codes with random lengths (Invoice Code, Product Code).
        Each code has a random length between min_length and max_length.
        """
        codes = []
        chars = string.ascii_uppercase + string.digits
        for _ in range(count):
            # Random length for this code
            length = random.randint(min_length, max_length)
            # 50% chance to have prefix-suffix format, 50% random string
            if random.random() > 0.5 and length >= 6:
                # Format: ABC-12345 (prefix 3 chars, suffix rest)
                prefix_len = min(3, length // 3)
                suffix_len = length - prefix_len - 1  # -1 for the dash
                prefix = "".join(random.choices(string.ascii_uppercase, k=prefix_len))
                suffix = "".join(random.choices(string.digits, k=suffix_len))
                codes.append(f"{prefix}-{suffix}")
            else:
                # Random alphanumeric string
                codes.append("".join(random.choices(chars, k=length)))
        return codes
