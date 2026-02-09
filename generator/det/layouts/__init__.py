"""
Layout modules for invoice generation with randomized text.

All layouts use vocabulary from model.rec.vocab for text generation.
"""

from .base_layout import (
    BaseLayout,
    LayoutConfig,
    LayoutType,
    FontManager,
    generate_random_text,
    generate_random_number_string,
    generate_random_phone,
    generate_random_label
)

from .supermarket_thermal import ThermalReceiptLayout
from .formal_vat import FormalVATLayout
from .handwritten import HandwrittenLayout
from .cafe_minimal import CafeMinimalLayout
from .restaurant_bill import RestaurantBillLayout
from .modern_pos import ModernPOSLayout
from .traditional_market import TraditionalMarketLayout
from .delivery_receipt import DeliveryReceiptLayout
from .hotel_bill import HotelBillLayout
from .utility_bill import UtilityBillLayout
from .ecommerce_receipt import EcommerceReceiptLayout
from .taxi_receipt import TaxiReceiptLayout

import random
from typing import Dict


class LayoutFactory:
    """Factory for creating layout instances."""

    LAYOUTS = {
        LayoutType.SUPERMARKET_THERMAL: ThermalReceiptLayout,
        LayoutType.FORMAL_VAT: FormalVATLayout,
        LayoutType.HANDWRITTEN: HandwrittenLayout,
        LayoutType.CAFE_MINIMAL: CafeMinimalLayout,
        LayoutType.RESTAURANT_BILL: RestaurantBillLayout,
        LayoutType.MODERN_POS: ModernPOSLayout,
        LayoutType.TRADITIONAL_MARKET: TraditionalMarketLayout,
        LayoutType.DELIVERY_RECEIPT: DeliveryReceiptLayout,
        LayoutType.HOTEL_BILL: HotelBillLayout,
        LayoutType.UTILITY_BILL: UtilityBillLayout,
        LayoutType.ECOMMERCE_RECEIPT: EcommerceReceiptLayout,
        LayoutType.TAXI_RECEIPT: TaxiReceiptLayout,
    }

    @classmethod
    def create(cls, layout_type: LayoutType = None) -> BaseLayout:
        """Create a layout instance."""
        if layout_type is None:
            layout_type = random.choice(list(cls.LAYOUTS.keys()))

        layout_class = cls.LAYOUTS.get(layout_type)
        if layout_class:
            return layout_class()

        # Default to thermal
        return ThermalReceiptLayout()

    @classmethod
    def create_random(cls, weights: Dict[LayoutType, float] = None) -> BaseLayout:
        """Create a random layout with optional weights."""
        if weights is None:
            weights = {
                LayoutType.SUPERMARKET_THERMAL: 0.12,
                LayoutType.FORMAL_VAT: 0.08,
                LayoutType.HANDWRITTEN: 0.08,
                LayoutType.CAFE_MINIMAL: 0.10,
                LayoutType.RESTAURANT_BILL: 0.10,
                LayoutType.MODERN_POS: 0.09,
                LayoutType.TRADITIONAL_MARKET: 0.08,
                LayoutType.DELIVERY_RECEIPT: 0.09,
                LayoutType.HOTEL_BILL: 0.08,
                LayoutType.UTILITY_BILL: 0.08,
                LayoutType.ECOMMERCE_RECEIPT: 0.06,
                LayoutType.TAXI_RECEIPT: 0.04,
            }

        types = list(weights.keys())
        probs = [weights[t] for t in types]
        total = sum(probs)
        probs = [p / total for p in probs]

        chosen = random.choices(types, weights=probs, k=1)[0]
        return cls.create(chosen)


__all__ = [
    # Base classes and utilities
    'BaseLayout',
    'LayoutConfig',
    'LayoutType',
    'FontManager',
    'generate_random_text',
    'generate_random_number_string',
    'generate_random_phone',
    'generate_random_label',
    # Layout classes
    'ThermalReceiptLayout',
    'FormalVATLayout',
    'HandwrittenLayout',
    'CafeMinimalLayout',
    'RestaurantBillLayout',
    'ModernPOSLayout',
    'TraditionalMarketLayout',
    'DeliveryReceiptLayout',
    'HotelBillLayout',
    'UtilityBillLayout',
    'EcommerceReceiptLayout',
    'TaxiReceiptLayout',
    # Factory
    'LayoutFactory',
]
