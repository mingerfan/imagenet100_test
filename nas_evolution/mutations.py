"""Mutation Operators for Network Architectures

Implements mutation operators that modify NetworkConfig objects from the
network_gen search space.
"""

import random
import copy
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MutationOperator:
    """Mutations for NetworkConfig

    Leverages the existing network_gen infrastructure to mutate architecture configurations.
    Supports mutations of:
    - Block types (from 22 block options)
    - Stem configuration (4 options)
    - Stride patterns (1344 stride encodings)
    - CT (ciphertext) policies
    - Downsample methods (6 options)
    """

    def __init__(self, mutation_probs: Dict[str, float] = None):
        """Initialize mutation operator

        Args:
            mutation_probs: Dict mapping mutation types to probabilities
        """
        self.probs = mutation_probs or {
            'block': 0.5,        # Mutate block type (most important)
            'stem': 0.15,        # Mutate stem config
            'stride': 0.15,      # Mutate stride pattern
            'ct_policy': 0.1,    # Mutate CT policy
            'downsample': 0.1,   # Mutate downsample method
        }

        # Normalize probabilities
        total = sum(self.probs.values())
        self.probs = {k: v/total for k, v in self.probs.items()}

        # Cache for stride codes grouped by block count
        # Format: {num_blocks: [stride_code1, stride_code2, ...]}
        self._stride_cache = None

    def mutate(self, parent_config):
        """Apply random mutation to network config

        Args:
            parent_config: NetworkConfig object to mutate

        Returns:
            Mutated NetworkConfig object (deep copy)
        """
        config = copy.deepcopy(parent_config)

        # Randomly select mutation type based on probabilities
        mutation_type = random.choices(
            list(self.probs.keys()),
            weights=list(self.probs.values())
        )[0]

        if mutation_type == 'block':
            self._mutate_block(config)
        elif mutation_type == 'stem':
            self._mutate_stem(config)
        elif mutation_type == 'stride':
            self._mutate_stride(config)
        elif mutation_type == 'ct_policy':
            self._mutate_ct_policy(config)
        elif mutation_type == 'downsample':
            self._mutate_downsample(config)

        return config

    def _mutate_block(self, config):
        """Mutate a block type in the hierarchical selection

        The network_gen uses hierarchical block selection:
        - Front 4 blocks: Independent choices (indices 0-3)
        - Remaining blocks: Paired choices (every 2 blocks share a choice)

        We mutate one choice to a different block type (0-21).
        """
        # Get the block_choices attribute (hierarchical representation)
        if not hasattr(config, 'block_choices'):
            # Fallback: reconstruct from blocks if needed
            print("Warning: config missing block_choices, cannot mutate block")
            return

        block_choices = config.block_choices

        # Randomly select which choice to mutate
        choice_idx = random.randint(0, len(block_choices) - 1)

        # Mutate to a different block type (0-21)
        current_block = block_choices[choice_idx]
        new_block = random.randint(0, 21)

        # Ensure it's different from current
        while new_block == current_block:
            new_block = random.randint(0, 21)

        block_choices[choice_idx] = new_block
        config.block_choices = block_choices

        # Regenerate blocks from hierarchical choices
        self._expand_block_choices(config)

    def _expand_block_choices(self, config):
        """Expand hierarchical block choices to full block list

        Matches the logic in HierarchicalBlockSelector from network_gen.
        """
        expanded = []

        # Front 4 blocks: direct mapping
        for i in range(min(4, len(config.block_choices))):
            expanded.append(config.block_choices[i])

        # Remaining blocks: pairs
        # Each choice after the first 4 applies to 2 consecutive blocks
        remaining_choices = config.block_choices[4:]
        for choice in remaining_choices:
            expanded.append(choice)
            expanded.append(choice)

        # Update the blocks attribute with new block types
        # Keep existing channels and strides
        if hasattr(config, 'blocks'):
            for i, block_config in enumerate(config.blocks):
                if i < len(expanded):
                    block_config.block_id = expanded[i]  # Use attribute access

    def _mutate_stem(self, config):
        """Mutate stem configuration (0-3)"""
        current_stem = config.stem_code
        new_stem = random.randint(0, 3)

        # Ensure different
        while new_stem == current_stem:
            new_stem = random.randint(0, 3)

        config.stem_code = new_stem

    def _build_stride_cache(self):
        """Build cache of stride codes grouped by block count

        This is done once on first stride mutation to speed up subsequent mutations.
        """
        if self._stride_cache is not None:
            return

        try:
            from network_gen.search_space import StrideEncoder
            encoder = StrideEncoder()

            self._stride_cache = {}
            for stride_code in range(1344):  # 0-1343
                num_blocks, _ = encoder.decode(stride_code)
                if num_blocks not in self._stride_cache:
                    self._stride_cache[num_blocks] = []
                self._stride_cache[num_blocks].append(stride_code)

        except ImportError:
            print("Warning: Could not import StrideEncoder")
            self._stride_cache = {}

    def _mutate_stride(self, config):
        """Mutate stride encoding (0-1343)

        Only selects stride codes that maintain the same number of blocks
        to ensure the mutation is valid.

        IMPORTANT: After changing strides, recalculates channels to maintain
        FHE constraints (channels depend on feature map sizes).
        """
        # Build cache on first use
        self._build_stride_cache()

        current_stride = config.stride_code
        current_num_blocks = len(config.blocks)

        # Get valid stride codes for current block count
        valid_stride_codes = self._stride_cache.get(current_num_blocks, [])

        # Remove current stride from options
        valid_stride_codes = [s for s in valid_stride_codes if s != current_stride]

        # Check if we have valid alternatives
        if not valid_stride_codes:
            # No alternatives with same block count, skip mutation
            return

        # Randomly select a new stride code
        new_stride = random.choice(valid_stride_codes)
        config.stride_code = new_stride

        # Update block strides based on new stride pattern
        try:
            from network_gen.search_space import StrideEncoder
            encoder = StrideEncoder()
            num_blocks, stride_positions = encoder.decode(new_stride)
            strides = encoder.get_strides_list(num_blocks, stride_positions)

            for i, block in enumerate(config.blocks):
                block.stride = strides[i]

            # ✅ FIX: Recalculate channels after stride change
            # This ensures channel counts match the new feature map sizes
            self._recalculate_channels(config)

        except ImportError:
            print("Warning: Could not import StrideEncoder, stride not updated")

    def _mutate_ct_policy(self, config):
        """Mutate CT policy at random downsample step

        IMPORTANT: After changing CT policy, recalculates channels to maintain
        FHE constraints (channels depend on CT count which is affected by policy).
        """
        if len(config.ct_policies) == 0:
            return

        idx = random.randint(0, len(config.ct_policies) - 1)
        current_policy = config.ct_policies[idx]

        # Toggle between 'keep' and 'half'
        new_policy = 'half' if current_policy == 'keep' else 'keep'
        config.ct_policies[idx] = new_policy

        # ✅ FIX: Recalculate channels after CT policy change
        # CT policy affects CT count progression, which determines channel counts
        self._recalculate_channels(config)

    def _mutate_downsample(self, config):
        """Mutate second downsample method (0-5)"""
        current_ds = config.second_ds_code
        new_ds = random.randint(0, 5)

        # Ensure different
        while new_ds == current_ds:
            new_ds = random.randint(0, 5)

        config.second_ds_code = new_ds

    def _recalculate_channels(self, config):
        """Recalculate all block channels after stride or CT policy mutation

        This is critical for maintaining FHE constraints where channel counts
        are derived from CT count and feature map sizes.

        Args:
            config: NetworkConfig to update
        """
        try:
            from network_gen.search_space import ChannelCalculator

            # Initialize calculator with same parameters as generator
            calculator = ChannelCalculator(
                ct_slots=32768,  # Standard FHE parameter
                input_size=224,  # ImageNet input size
                stem_downsample=4  # Stem does 2x downsampling twice
            )

            # Get current strides from blocks
            strides = [block.stride for block in config.blocks]

            # Recalculate channel sequence based on current strides and CT policies
            channels, feature_sizes, ct_counts = calculator.compute_channels_sequence(
                strides=strides,
                ct_policies=config.ct_policies,
                initial_ct_count=config.initial_ct_count
            )

            # Get stem output channels
            stem_out_channels = calculator.get_initial_channels(config.initial_ct_count)

            # Update each block's in_channels and out_channels
            for i, block in enumerate(config.blocks):
                if i == 0:
                    block.in_channels = stem_out_channels
                else:
                    block.in_channels = config.blocks[i-1].out_channels
                block.out_channels = channels[i]

            # Update config's stem_out_channels
            config.stem_out_channels = stem_out_channels

        except ImportError as e:
            print(f"Warning: Could not import ChannelCalculator: {e}")
        except Exception as e:
            print(f"Warning: Error recalculating channels: {e}")


def test_mutation():
    """Test mutation operators"""
    from network_gen import RandomNetworkGenerator
    from network_gen.generator_config import GeneratorConfig

    # Load config
    config = GeneratorConfig.from_yaml('network_gen/configs/imagenet_224.yaml')
    generator = RandomNetworkGenerator(config)

    # Generate random network
    network_config = generator.generate_random()

    # Create mutator
    mutator = MutationOperator()

    # Test mutations
    print(f"Original config:")
    print(f"  Stem: {network_config.stem_code}")
    print(f"  Stride: {network_config.stride_code}")
    print(f"  Block choices: {network_config.block_choices[:5]}...")

    # Apply 10 mutations
    current = network_config
    for i in range(10):
        mutated = mutator.mutate(current)
        print(f"\nMutation {i+1}:")
        print(f"  Stem: {mutated.stem_code}")
        print(f"  Stride: {mutated.stride_code}")
        print(f"  Block choices: {mutated.block_choices[:5]}...")
        current = mutated


if __name__ == '__main__':
    test_mutation()
