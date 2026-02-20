"""
Main Synthetic Invoice Generator - Orchestrates all components.

Features:
    - Configurable generation scenarios
    - Realistic data distribution
    - Multiple output formats
    - Batch generation with progress tracking
    - Quality assurance checks
"""

import os
import random
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict
from enum import Enum
from pathlib import Path
from PIL import Image

from .layouts import LayoutFactory, LayoutType
from .defects import apply_defects_light, apply_defects_medium, apply_defects_heavy, DefectSimulator
from .edge_cases import apply_random_edge_case, EdgeCaseGenerator


class GenerationScenario(Enum):
    """Predefined generation scenarios."""
    TRAINING_BALANCED = "training_balanced"      # Balanced mix for training
    TRAINING_HARD = "training_hard"              # More difficult cases
    VALIDATION = "validation"                     # Clean for validation
    EDGE_CASES_FOCUS = "edge_cases_focus"        # Heavy edge cases
    RETAIL_FOCUS = "retail_focus"                # Focus on retail receipts
    RESTAURANT_FOCUS = "restaurant_focus"        # Focus on restaurant bills
    FORMAL_INVOICES = "formal_invoices"          # VAT invoices
    PURE_RANDOM_FOCUS = "pure_random_focus"      # 100% Type 1
    PSEUDO_FOCUS = "pseudo_focus"                # 100% Type 2


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    output_dir: str = "data/raw"
    num_samples: int = 100
    scenario: GenerationScenario = GenerationScenario.TRAINING_BALANCED

    # Distribution controls
    realistic_ratio: float = 0.80       # Normal invoices
    edge_case_ratio: float = 0.10       # Edge cases
    blank_ratio: float = 0.05           # Blank pages
    unreadable_ratio: float = 0.05      # Unreadable

    # Quality settings
    min_jpeg_quality: int = 40
    max_jpeg_quality: int = 95
    defect_level: str = "mixed"

    # Layout preferences (weights) — leave empty to use LayoutFactory defaults
    layout_weights: Dict[LayoutType, float] = field(default_factory=dict)


def get_scenario_config(scenario: GenerationScenario) -> GenerationConfig:
    """Get configuration for a specific scenario."""
    configs = {
        GenerationScenario.TRAINING_BALANCED: GenerationConfig(
            realistic_ratio=0.75,
            edge_case_ratio=0.15,
            blank_ratio=0.05,
            unreadable_ratio=0.05,
            defect_level="mixed"
        ),

        GenerationScenario.TRAINING_HARD: GenerationConfig(
            realistic_ratio=0.50,
            edge_case_ratio=0.35,
            blank_ratio=0.08,
            unreadable_ratio=0.07,
            defect_level="heavy"
        ),

        GenerationScenario.VALIDATION: GenerationConfig(
            realistic_ratio=0.95,
            edge_case_ratio=0.05,
            blank_ratio=0.00,
            unreadable_ratio=0.00,
            defect_level="light"
        ),

        GenerationScenario.EDGE_CASES_FOCUS: GenerationConfig(
            realistic_ratio=0.30,
            edge_case_ratio=0.60,
            blank_ratio=0.05,
            unreadable_ratio=0.05,
            defect_level="heavy"
        ),
        
        # Add basic logical defaults for others to avoid key errors
        GenerationScenario.RETAIL_FOCUS: GenerationConfig(
            realistic_ratio=0.90,
            edge_case_ratio=0.10,
            layout_weights={LayoutType.SUPERMARKET_THERMAL: 0.8, LayoutType.MODERN_POS: 0.2}
        ),
        GenerationScenario.RESTAURANT_FOCUS: GenerationConfig(
             realistic_ratio=0.90,
             edge_case_ratio=0.10,
             layout_weights={LayoutType.RESTAURANT_BILL: 0.6, LayoutType.CAFE_MINIMAL: 0.4}
        ),
        GenerationScenario.FORMAL_INVOICES: GenerationConfig(
             realistic_ratio=0.95,
             edge_case_ratio=0.05,
             layout_weights={LayoutType.FORMAL_VAT: 1.0}
        ),
        GenerationScenario.PURE_RANDOM_FOCUS: GenerationConfig(
            realistic_ratio=0.90,
            edge_case_ratio=0.10,
            defect_level="light"
        ),
        GenerationScenario.PSEUDO_FOCUS: GenerationConfig(
            realistic_ratio=0.90,
            edge_case_ratio=0.10,
            defect_level="light"
        ),
    }

    return configs.get(scenario, GenerationConfig())


class SyntheticInvoiceGenerator:
    """
    Main generator for synthetic invoice detection dataset.
    Combines layouts, defects, and edge cases to create realistic training data.
    """

    def __init__(self, config: GenerationConfig = None):
        """
        Initialize generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config or GenerationConfig()
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_scenario(cls, scenario: GenerationScenario,
                      output_dir: str = "data/raw",
                      num_samples: int = 100) -> "SyntheticInvoiceGenerator":
        """Create generator from a predefined scenario."""
        config = get_scenario_config(scenario)
        config.output_dir = output_dir
        config.num_samples = num_samples
        return cls(config)

    def generate_sample_data(self) -> Dict:
        """Generate random invoice data."""
        num_items = random.randint(3, 15)
        items = []
        
        for _ in range(num_items):
            items.append({
                'qty': random.randint(1, 5),
                'total': random.randint(10000, 500000)
            })
        
        grand_total = sum(item['total'] for item in items)
        
        return {
            'items': items,
            'grand_total': grand_total,
            'tax': int(grand_total * 0.1) if random.random() < 0.5 else 0,
            'discount': random.randint(0, int(grand_total * 0.2)) if random.random() < 0.3 else 0
        }

    def _save_sample(self, sample_id: int, img: Image.Image,
                     data: Dict, **extra_metadata) -> Dict:
        """Save image and metadata."""
        # Generate unique ID
        unique_id = f"{sample_id:06d}_{uuid.uuid4().hex[:8]}"
        
        img_filename = f"invoice_{unique_id}.jpg"
        img_path = self.output_dir / img_filename
        
        # Save image with random quality
        quality = random.randint(self.config.min_jpeg_quality, self.config.max_jpeg_quality)
        img.convert("RGB").save(img_path, "JPEG", quality=quality)
        
        # Prepare annotation data
        ocr_annotations = extra_metadata.get("ocr_annotations", [])
        
        metadata = {
            "id": unique_id,
            "image_path": str(img_path),
            "width": img.width,
            "height": img.height,
            "sample_type": extra_metadata.get("sample_type", "unknown"),
            "layout_type": extra_metadata.get("layout_type", "unknown"),
            "defects": extra_metadata.get("defects", []),
            "edge_case": extra_metadata.get("edge_case", None),
            "annotations": ocr_annotations,
            # Legacy fields
            "ocr_text": [a["text"] for a in ocr_annotations] if ocr_annotations else [],
        }
        
        json_filename = f"invoice_{unique_id}.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        return metadata

    def _generate_realistic(self, sample_id: int) -> Dict:
        """Generate a realistic invoice."""
        # Select layout using weights
        layout = LayoutFactory.create_random(self.config.layout_weights)
        data = self.generate_sample_data()
        img = layout.render(data)
        annotations = layout.get_ocr_annotations()
        
        # Apply defects based on config level
        applied_defects = []
        if self.config.defect_level == "light":
            img = apply_defects_light(img)
            applied_defects = ["light"]
        elif self.config.defect_level == "medium":
            img = apply_defects_medium(img)
            applied_defects = ["medium"]
        elif self.config.defect_level == "heavy":
            img = apply_defects_heavy(img)
            applied_defects = ["heavy"]
        elif self.config.defect_level == "mixed":
            level = random.choice(['none', 'light', 'medium', 'heavy'])
            if level == 'light':
                img = apply_defects_light(img)
            elif level == 'medium':
                img = apply_defects_medium(img)
            elif level == 'heavy':
                img = apply_defects_heavy(img)
            applied_defects = [level]
            
        return self._save_sample(
            sample_id, img, data,
            sample_type="realistic",
            layout_type=layout.config.layout_type.value,
            defects=applied_defects,
            ocr_annotations=annotations
        )

    def _generate_with_edge_case(self, sample_id: int) -> Dict:
        """Generate an invoice with edge case applied."""
        # Base invoice
        layout = LayoutFactory.create_random(self.config.layout_weights)
        data = self.generate_sample_data()
        img = layout.render(data)
        annotations = layout.get_ocr_annotations()
        
        # Apply edge case
        img, annotations = apply_random_edge_case(img, annotations)
        
        # Maybe add defects too
        applied_defects = []
        if random.random() < 0.5:
            img = apply_defects_light(img)
            applied_defects = ["light_mixed"]
            
        return self._save_sample(
            sample_id, img, data,
            sample_type="edge_case",
            layout_type=layout.config.layout_type.value,
            defects=applied_defects,
            edge_case="random",
            ocr_annotations=annotations
        )

    def _generate_blank(self, sample_id: int) -> Dict:
        """Generate a blank page."""
        # 600x800 is arbitrary default size
        img = EdgeCaseGenerator.create_blank_with_artifacts((600, 800))
        
        return self._save_sample(
            sample_id, img, {},
            sample_type="blank",
            ocr_annotations=[]
        )

    def _generate_unreadable(self, sample_id: int) -> Dict:
        """Generate an unreadable/corrupted image."""
        # Reuse blank or heavy defects for now since we don't have dedicated unreadable generator
        layout = LayoutFactory.create_random()
        data = self.generate_sample_data()
        img = layout.render(data)
        annotations = layout.get_ocr_annotations()
        
        # Apply heavy blur and noise to make it unreadable
        img = DefectSimulator.apply_blur(img, radius=3.0)
        img = DefectSimulator.apply_noise(img, intensity=0.1)
        img = DefectSimulator.apply_brightness_variation(img, factor=0.3) # Very dark
        
        return self._save_sample(
            sample_id, img, data,
            sample_type="unreadable",
            ocr_annotations=annotations  # Keep annotations even for unreadable images
        )