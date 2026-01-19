"""Mutation Operators for Network Architectures

Implements mutation operators that modify NetworkConfig objects from the
network_gen search space.
"""

import random
import copy
from typing import Dict, Any, Optional, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_gen.search_space import UNIFIED_BLOCKS
from models.gate_net_cmp.block_def import StablePoly4


class MutationOperator:
    """Mutations for NetworkConfig

    Leverages the existing network_gen infrastructure to mutate architecture configurations.
    Supports mutations of:
    - Block types (from 22 block options)
    - Stem configuration (4 options)
    - Stride patterns (1344 stride encodings)
    - CT (ciphertext) policies
    - Downsample methods (6 options)
    - Initial CT count
    """

    def __init__(
        self,
        mutation_probs: Dict[str, float] = None,
        ct_slots: int = 32768,
        input_size: int = 224,
        stem_downsample: int = 4,
        initial_min_channels: int = 16,
        initial_max_channels: Optional[int] = 64,
    ):
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
            'initial_ct': 0.05,  # Mutate initial CT count
        }

        # Normalize probabilities
        total = sum(self.probs.values())
        self.probs = {k: v/total for k, v in self.probs.items()}

        # Cache for stride codes grouped by block count
        # Format: {num_blocks: [stride_code1, stride_code2, ...]}
        self._stride_cache = None
        self.ct_slots = ct_slots
        self.input_size = input_size
        self.stem_downsample = stem_downsample
        self.initial_min_channels = initial_min_channels
        self.initial_max_channels = initial_max_channels
        self._channel_calculator = None

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
        elif mutation_type == 'initial_ct':
            self._mutate_initial_ct_count(config)

        self._enforce_tail_no_poly4(config)
        self._sync_blocks_from_choices(config)
        return config

    def _mutate_block(self, config):
        """Mutate a block type in per-block block_choices."""
        block_choices = self._ensure_block_choices(config)
        if not block_choices:
            return

        num_blocks = len(block_choices)
        choice_idx = random.randint(0, num_blocks - 1)
        group_start, group_end = self._group_bounds(choice_idx, num_blocks)
        tail_start = self._tail_start(num_blocks)
        group_in_tail = group_end >= tail_start

        candidate_ids = self._all_block_ids()
        if group_in_tail:
            candidate_ids = self._non_poly4_block_ids(candidate_ids)
            if not candidate_ids:
                raise ValueError("No non-poly4 block IDs available for tail mutation")

        current_ids = {block_choices[i] for i in range(group_start, group_end + 1)}
        candidate_ids = [block_id for block_id in candidate_ids if block_id not in current_ids]
        if not candidate_ids:
            return

        new_block = random.choice(candidate_ids)
        for i in range(group_start, group_end + 1):
            block_choices[i] = new_block

        config.block_choices = block_choices
        self._enforce_tail_no_poly4(config)
        self._sync_blocks_from_choices(config)

    def _ensure_block_choices(self, config) -> List[int]:
        if hasattr(config, 'blocks') and getattr(config, 'blocks') is not None:
            block_ids = [block.block_id for block in config.blocks]
        else:
            block_ids = []

        if not hasattr(config, 'block_choices') or config.block_choices is None:
            config.block_choices = block_ids
            return config.block_choices

        if block_ids and len(config.block_choices) != len(block_ids):
            config.block_choices = block_ids
        return config.block_choices

    def _sync_blocks_from_choices(self, config) -> None:
        if not hasattr(config, 'blocks') or config.blocks is None:
            return
        for i, block_config in enumerate(config.blocks):
            if i < len(config.block_choices):
                block_config.block_id = config.block_choices[i]

    def _all_block_ids(self) -> List[int]:
        return list(range(len(UNIFIED_BLOCKS)))

    def _is_poly4_block_id(self, block_id: int) -> bool:
        return UNIFIED_BLOCKS[block_id].activation_class is StablePoly4

    def _non_poly4_block_ids(self, block_ids: List[int]) -> List[int]:
        return [block_id for block_id in block_ids if not self._is_poly4_block_id(block_id)]

    def _tail_start(self, num_blocks: int) -> int:
        return num_blocks // 2

    def _group_bounds(self, index: int, num_blocks: int) -> Tuple[int, int]:
        if index < 4 or num_blocks <= 4:
            return index, index
        group_start = 4 + ((index - 4) // 2) * 2
        group_end = min(group_start + 1, num_blocks - 1)
        return group_start, group_end

    def _enforce_tail_no_poly4(self, config) -> None:
        block_choices = self._ensure_block_choices(config)
        if not block_choices:
            return

        num_blocks = len(block_choices)
        tail_start = self._tail_start(num_blocks)
        non_poly4_ids = self._non_poly4_block_ids(self._all_block_ids())
        if not non_poly4_ids:
            raise ValueError("No non-poly4 block IDs available for tail constraint")

        idx = 0
        while idx < num_blocks:
            group_start, group_end = self._group_bounds(idx, num_blocks)
            group_in_tail = group_end >= tail_start
            if group_in_tail and any(self._is_poly4_block_id(block_choices[i]) for i in range(group_start, group_end + 1)):
                new_block = random.choice(non_poly4_ids)
                for i in range(group_start, group_end + 1):
                    block_choices[i] = new_block
            idx = group_end + 1

        config.block_choices = block_choices

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
            calculator = self._get_channel_calculator()
            if calculator is None:
                return

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

    def _get_channel_calculator(self):
        if self._channel_calculator is not None:
            return self._channel_calculator
        try:
            from network_gen.search_space import ChannelCalculator
        except ImportError:
            return None
        self._channel_calculator = ChannelCalculator(
            ct_slots=self.ct_slots,
            input_size=self.input_size,
            stem_downsample=self.stem_downsample,
        )
        return self._channel_calculator

    def _mutate_initial_ct_count(self, config):
        """Mutate initial CT count and recompute channels"""
        calculator = self._get_channel_calculator()
        if calculator is None:
            return

        feature_size = calculator.feature_size_after_stem
        min_ct = calculator.compute_ct_from_channels(
            self.initial_min_channels,
            feature_size,
        )
        if self.initial_max_channels is not None:
            max_ct = calculator.compute_ct_from_channels(
                self.initial_max_channels,
                feature_size,
            )
            max_ct = max(max_ct, min_ct)
        else:
            max_ct = min_ct

        current = config.initial_ct_count
        if min_ct == max_ct:
            new_ct = min_ct
        else:
            new_ct = random.randint(min_ct, max_ct)
            if new_ct == current:
                if current < max_ct:
                    new_ct = current + 1
                elif current > min_ct:
                    new_ct = current - 1

        if new_ct == current:
            return

        config.initial_ct_count = new_ct
        self._recalculate_channels(config)


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
