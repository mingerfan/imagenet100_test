#!/usr/bin/env python3
"""Test the modified fitness function with latency constraints

Verifies:
1. FLOPs ranking (higher is better)
2. Latency hard constraint (> baseline filtered)
3. Latency soft reward (multipliers applied correctly)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from nas_evolution.fitness_function import AZNASFitnessFunction


def test_flops_ranking():
    """Test that higher FLOPs gets higher ranking"""
    print("\n" + "="*70)
    print("Test 1: FLOPs Ranking (Higher FLOPs = Better)")
    print("="*70)

    fitness_fn = AZNASFitnessFunction(latency_baseline=22334905.50)

    # Two architectures with same proxy scores but different FLOPs
    scores = [
        {
            'expressivity': 5.0,
            'progressivity': 0.8,
            'trainability': 1.0,
            'flops': 1e9,  # Lower FLOPs
            'fhe_latency': 20000000  # Under baseline
        },
        {
            'expressivity': 5.0,
            'progressivity': 0.8,
            'trainability': 1.0,
            'flops': 2e9,  # Higher FLOPs
            'fhe_latency': 20000000  # Under baseline
        }
    ]

    fitness = fitness_fn.compute_fitness(scores)

    print(f"Architecture 1 (FLOPs=1e9): fitness={fitness[0]:.4f}")
    print(f"Architecture 2 (FLOPs=2e9): fitness={fitness[1]:.4f}")

    if fitness[1] > fitness[0]:
        print("✓ PASS: Higher FLOPs gets higher fitness")
    else:
        print("✗ FAIL: Higher FLOPs should get higher fitness")


def test_latency_constraint():
    """Test latency hard constraint filters out violators"""
    print("\n" + "="*70)
    print("Test 2: Latency Hard Constraint (> Baseline Filtered)")
    print("="*70)

    baseline = 22334905.50
    fitness_fn = AZNASFitnessFunction(latency_baseline=baseline)

    scores = [
        {
            'expressivity': 6.0,  # Better proxy scores
            'progressivity': 0.95,
            'trainability': 1.5,
            'flops': 3e9,
            'fhe_latency': 25000000  # OVER baseline
        },
        {
            'expressivity': 5.0,  # Worse proxy scores
            'progressivity': 0.7,
            'trainability': 0.8,
            'flops': 2e9,
            'fhe_latency': 20000000  # Under baseline
        }
    ]

    fitness = fitness_fn.compute_fitness(scores)

    print(f"Architecture 1 (latency={scores[0]['fhe_latency']:.0f}, "
          f"{scores[0]['fhe_latency']/baseline:.1%} of baseline): fitness={fitness[0]:.4f}")
    print(f"Architecture 2 (latency={scores[1]['fhe_latency']:.0f}, "
          f"{scores[1]['fhe_latency']/baseline:.1%} of baseline): fitness={fitness[1]:.4f}")

    if fitness[0] < -1e9:  # Should be filtered
        print("✓ PASS: Architecture exceeding baseline is filtered (fitness << 0)")
    else:
        print("✗ FAIL: Architecture exceeding baseline should be filtered")

    if fitness[1] > -10:  # Should be normal
        print("✓ PASS: Architecture under baseline has normal fitness")
    else:
        print("✗ FAIL: Architecture under baseline should have normal fitness")


def test_latency_reward_tiers():
    """Test latency soft rewards are applied correctly"""
    print("\n" + "="*70)
    print("Test 3: Latency Soft Rewards (Multiplier Tiers)")
    print("="*70)

    baseline = 22334905.50
    fitness_fn = AZNASFitnessFunction(latency_baseline=baseline)

    # Same proxy scores and FLOPs, different latencies
    base_arch = {
        'expressivity': 5.5,
        'progressivity': 0.85,
        'trainability': 1.2,
        'flops': 2e9,
    }

    test_cases = [
        (baseline * 0.35, 1.30, "40% tier"),   # Should get 1.30x
        (baseline * 0.45, 1.25, "50% tier"),   # Should get 1.25x
        (baseline * 0.55, 1.20, "60% tier"),   # Should get 1.20x
        (baseline * 0.75, 1.15, "70% tier"),   # Should get 1.15x
        (baseline * 0.85, 1.10, "80% tier"),   # Should get 1.10x
        (baseline * 0.95, 1.05, "90% tier"),   # Should get 1.05x
        (baseline * 0.99, 1.00, "100% tier"),  # Should get 1.00x
    ]

    scores_list = []
    for latency, expected_mult, tier_name in test_cases:
        arch = base_arch.copy()
        arch['fhe_latency'] = latency
        scores_list.append(arch)

    fitness = fitness_fn.compute_fitness(scores_list)

    # Since base scores are the same for all, fitness differences come from multipliers
    # We can verify the ratio
    print("\nLatency tier verification:")
    for i, (latency, expected_mult, tier_name) in enumerate(test_cases):
        print(f"  {tier_name}: latency={latency:.0f} ({latency/baseline:.1%} of baseline)")
        print(f"    Expected multiplier: {expected_mult:.2f}x")
        print(f"    Fitness: {fitness[i]:.4f}")

    # Compare adjacent tiers - lower latency should have better (less negative) fitness
    all_correct = True
    for i in range(len(fitness) - 1):
        if fitness[i] <= fitness[i+1]:  # Lower latency should be better
            print(f"\n✗ FAIL: {test_cases[i][2]} should have better fitness than {test_cases[i+1][2]}")
            all_correct = False

    if all_correct:
        print("\n✓ PASS: All latency tiers correctly ordered")


def test_combined_scenario():
    """Test realistic population with diverse architectures"""
    print("\n" + "="*70)
    print("Test 4: Combined Scenario (Realistic Population)")
    print("="*70)

    baseline = 22334905.50
    fitness_fn = AZNASFitnessFunction(latency_baseline=baseline)

    scores = [
        {
            'name': 'Shallow-Fast',
            'expressivity': 4.5,
            'progressivity': 0.6,
            'trainability': -0.3,
            'flops': 5e8,  # Low FLOPs
            'fhe_latency': 8000000  # Very low latency (40% tier)
        },
        {
            'name': 'ResNet18-Like',
            'expressivity': 5.8,
            'progressivity': 0.85,
            'trainability': 1.2,
            'flops': 1.8e9,  # Medium FLOPs
            'fhe_latency': 22000000  # At baseline
        },
        {
            'name': 'Deep-Slow',
            'expressivity': 6.2,
            'progressivity': 0.95,
            'trainability': 1.5,
            'flops': 4e9,  # High FLOPs
            'fhe_latency': 25000000  # Over baseline (FILTERED)
        },
        {
            'name': 'Efficient-Deep',
            'expressivity': 6.0,
            'progressivity': 0.90,
            'trainability': 1.3,
            'flops': 3e9,  # High FLOPs
            'fhe_latency': 18000000  # Low latency (80% tier)
        }
    ]

    fitness = fitness_fn.compute_fitness(scores)

    print("\nFitness ranking:")
    ranking = sorted(enumerate(fitness), key=lambda x: x[1], reverse=True)
    for rank, (idx, score) in enumerate(ranking, 1):
        arch = scores[idx]
        status = "FILTERED" if score < -1e9 else "VALID"
        print(f"{rank}. {arch['name']}: {score:.4f} ({status})")
        print(f"   Latency: {arch['fhe_latency']:.0f} ({arch['fhe_latency']/baseline:.1%} of baseline)")

    # Verify Deep-Slow is filtered
    deep_slow_idx = 2
    if fitness[deep_slow_idx] < -1e9:
        print("\n✓ PASS: Deep-Slow (latency > baseline) is filtered")
    else:
        print("\n✗ FAIL: Deep-Slow should be filtered")

    # Verify high-FLOPs architectures with valid latency rank well
    valid_scores = [(i, f) for i, f in enumerate(fitness) if f > -1e9]
    if len(valid_scores) >= 2:
        # Efficient-Deep should rank well due to high FLOPs + latency bonus
        efficient_deep_idx = 3
        if fitness[efficient_deep_idx] > fitness[0]:  # Better than Shallow-Fast
            print("✓ PASS: Efficient-Deep ranks better than Shallow-Fast")
        else:
            print("✗ FAIL: Efficient-Deep should rank better (higher FLOPs + latency bonus)")


def test_statistics():
    """Test detailed statistics computation"""
    print("\n" + "="*70)
    print("Test 5: Statistics Computation")
    print("="*70)

    baseline = 22334905.50
    fitness_fn = AZNASFitnessFunction(latency_baseline=baseline)

    # Population with 2 violations, 2 bonus-eligible
    scores = [
        {'expressivity': 5.0, 'progressivity': 0.8, 'trainability': 1.0,
         'flops': 1e9, 'fhe_latency': 25000000},  # Violation
        {'expressivity': 5.0, 'progressivity': 0.8, 'trainability': 1.0,
         'flops': 1e9, 'fhe_latency': 23000000},  # Violation
        {'expressivity': 5.0, 'progressivity': 0.8, 'trainability': 1.0,
         'flops': 1e9, 'fhe_latency': 15000000},  # Bonus (67%)
        {'expressivity': 5.0, 'progressivity': 0.8, 'trainability': 1.0,
         'flops': 1e9, 'fhe_latency': 18000000},  # Bonus (81%)
        {'expressivity': 5.0, 'progressivity': 0.8, 'trainability': 1.0,
         'flops': 1e9, 'fhe_latency': 21000000},  # No bonus (94%)
    ]

    fitness = fitness_fn.compute_fitness(scores)
    stats = fitness_fn.compute_detailed_stats(scores, fitness)

    print(f"Total architectures: {len(scores)}")
    print(f"Valid architectures: {stats['valid_count']}")
    print(f"Filtered (latency > baseline): {stats['invalid_count']}")
    print(f"Latency violations: {stats['latency_violation_count']}")
    print(f"Latency bonus eligible (≤90%): {stats['latency_bonus_count']}")

    if stats['latency_violation_count'] == 2:
        print("✓ PASS: Correctly counted 2 latency violations")
    else:
        print(f"✗ FAIL: Expected 2 violations, got {stats['latency_violation_count']}")

    if stats['latency_bonus_count'] == 2:  # 15M and 18M are ≤ 90%
        print("✓ PASS: Correctly counted bonus-eligible architectures")
    else:
        print(f"✗ FAIL: Expected 2 bonus-eligible, got {stats['latency_bonus_count']}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MODIFIED FITNESS FUNCTION TEST SUITE")
    print("="*70)
    print(f"ResNet-18 Latency Baseline: 22,334,905.50")

    test_flops_ranking()
    test_latency_constraint()
    test_latency_reward_tiers()
    test_combined_scenario()
    test_statistics()

    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
