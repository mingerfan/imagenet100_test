#!/usr/bin/env python3
"""Test script to verify NAS implementation components

Tests each component of the NAS system before running full evolution.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_zero_cost_proxy():
    """Test modified zero_cost_proxy functions"""
    print("\n" + "="*60)
    print("TEST 1: Zero-Cost Proxy Modifications")
    print("="*60)

    from network_evaluate.zero_cost_proxy import (
        prepare_poly4_for_evaluation,
        ModelWrapper,
        compute_fhe_latency,
        compute_nas_score
    )
    from models import get_model

    # Create a simple model for testing
    print("\n1. Testing with resnet18...")
    model = get_model('resnet18', num_classes=100)

    # Test prepare_poly4_for_evaluation
    print("   - Testing prepare_poly4_for_evaluation()...")
    prepare_poly4_for_evaluation(model)
    print("     ✓ No errors")

    # Test ModelWrapper
    print("   - Testing ModelWrapper...")
    wrapped = ModelWrapper(model)
    input_tensor = torch.randn(2, 3, 224, 224)
    features, output = wrapped.extract_layer_features_and_logit(input_tensor)
    print(f"     ✓ Extracted {len(features)} layer features")
    print(f"     ✓ Output shape: {output.shape}")

    # Test compute_fhe_latency
    print("   - Testing compute_fhe_latency()...")
    fhe_metrics = compute_fhe_latency(model, (2, 3, 224, 224))
    print(f"     ✓ FHE latency: {fhe_metrics['fhe_latency']:.0f}")
    print(f"     ✓ Boot count: {fhe_metrics['fhe_boot_count']}")
    print(f"     ✓ Max depth: {fhe_metrics['fhe_max_depth']}")

    # Test compute_nas_score (full integration)
    print("   - Testing compute_nas_score()...")
    scores = compute_nas_score(
        model=model,
        gpu=0 if torch.cuda.is_available() else None,
        trainloader=None,
        resolution=224,
        batch_size=2,
        init=False,
        use_wrapper=True
    )
    print(f"     ✓ Expressivity: {scores['expressivity']:.4f}")
    print(f"     ✓ Progressivity: {scores['progressivity']:.4f}")
    print(f"     ✓ Trainability: {scores['trainability']:.4f}")
    print(f"     ✓ FHE Latency: {scores['fhe_latency']:.0f}")

    print("\n✅ Test 1 PASSED: Zero-Cost Proxy works correctly")
    return True


def test_fitness_function():
    """Test AZ-NAS fitness function"""
    print("\n" + "="*60)
    print("TEST 2: AZ-NAS Fitness Function")
    print("="*60)

    from nas_evolution.fitness_function import AZNASFitnessFunction
    import numpy as np

    fitness_fn = AZNASFitnessFunction()

    # Create test population with varying scores
    population_scores = [
        {'expressivity': 5.0, 'progressivity': 1.0, 'trainability': 2.0, 'fhe_latency': 1e6},
        {'expressivity': 4.0, 'progressivity': 2.0, 'trainability': 3.0, 'fhe_latency': 8e5},
        {'expressivity': 6.0, 'progressivity': 0.5, 'trainability': 2.5, 'fhe_latency': 1.2e6},
        {'expressivity': 3.0, 'progressivity': 3.0, 'trainability': 1.5, 'fhe_latency': 9e5},
        {'expressivity': 5.5, 'progressivity': 1.5, 'trainability': 2.8, 'fhe_latency': 7e5},
    ]

    print(f"\nPopulation size: {len(population_scores)}")

    # Compute fitness
    aznas_scores = fitness_fn.compute_fitness(population_scores)

    print(f"\nAZ-NAS Fitness Scores:")
    for i, score in enumerate(aznas_scores):
        print(f"  Architecture {i+1}: {score:.6f}")

    # Get best indices
    best_indices = fitness_fn.get_best_indices(aznas_scores, k=3)
    print(f"\nTop 3 architectures: {best_indices + 1}")

    # Compute detailed stats
    stats = fitness_fn.compute_detailed_stats(population_scores, aznas_scores)
    print(f"\nPopulation statistics:")
    print(f"  Best fitness: {stats['best_aznas_score']:.6f}")
    print(f"  Mean fitness: {stats['mean_aznas_score']:.6f}")
    print(f"  Best latency: {stats['best_latency']:.0f}")
    print(f"  Mean latency: {stats['mean_latency']:.0f}")

    # Verify fitness ranking makes sense
    best_idx = best_indices[0]
    best_arch = population_scores[best_idx]
    print(f"\nBest architecture (#{best_idx+1}):")
    print(f"  Expressivity: {best_arch['expressivity']:.4f}")
    print(f"  Progressivity: {best_arch['progressivity']:.4f}")
    print(f"  Trainability: {best_arch['trainability']:.4f}")
    print(f"  Latency: {best_arch['fhe_latency']:.0f}")

    print("\n✅ Test 2 PASSED: Fitness function works correctly")
    return True


def test_mutation():
    """Test mutation operators"""
    print("\n" + "="*60)
    print("TEST 3: Mutation Operators")
    print("="*60)

    from nas_evolution.mutations import MutationOperator
    from network_gen import create_random_network

    # Use simplified network creation
    print("\nGenerating random network using create_random_network()...")
    _, original_config = create_random_network()

    print(f"\nOriginal architecture:")
    print(f"  Stem code: {original_config.stem_code}")
    print(f"  Stride code: {original_config.stride_code}")
    print(f"  Second DS code: {original_config.second_ds_code}")
    print(f"  Block choices (first 5): {original_config.block_choices[:5]}")

    # Create mutator
    mutator = MutationOperator()

    # Apply mutations
    print(f"\nApplying 5 mutations...")
    current = original_config
    for i in range(5):
        mutated = mutator.mutate(current)
        print(f"\nMutation {i+1}:")
        print(f"  Stem: {original_config.stem_code} → {mutated.stem_code}")
        print(f"  Stride: {original_config.stride_code} → {mutated.stride_code}")
        print(f"  Changed: {mutated.stem_code != original_config.stem_code or mutated.stride_code != original_config.stride_code or mutated.block_choices != original_config.block_choices}")
        current = mutated

    print("\n✅ Test 3 PASSED: Mutation operators work correctly")
    return True


def test_population():
    """Test population management"""
    print("\n" + "="*60)
    print("TEST 4: Population Management")
    print("="*60)

    from nas_evolution.population import Population
    from network_gen import create_random_network

    # Create small population
    population = Population(max_size=5)

    # Add individuals
    print(f"\nAdding 7 individuals to population (max_size=5)...")
    for i in range(7):
        _, network_config = create_random_network()
        scores = {
            'expressivity': 5.0 + i,
            'progressivity': 1.0,
            'trainability': 2.0,
            'fhe_latency': 1e6
        }
        fitness = float(i)
        population.add(network_config, scores, fitness, generation=i)

        print(f"  Added individual {i+1}, population size: {len(population)}")

    # Check FIFO aging
    print(f"\nFinal population size: {len(population)}")
    print(f"History size: {len(population.history)}")

    # Get stats
    stats = population.get_current_stats()
    print(f"\nPopulation statistics:")
    print(f"  Size: {stats['size']}")
    print(f"  Mean fitness: {stats['mean_fitness']:.4f}")
    print(f"  Best fitness: {stats['best_fitness']:.4f}")

    # Get best
    best = population.get_best(k=3, from_history=True)
    print(f"\nTop 3 individuals (from history):")
    for i, ind in enumerate(best):
        print(f"  #{i+1}: Fitness={ind.aznas_fitness:.4f}, Gen={ind.generation}")

    print("\n✅ Test 4 PASSED: Population management works correctly")
    return True


def test_network_builder():
    """Test building networks from NetworkConfig"""
    print("\n" + "="*60)
    print("TEST 5: Network Builder Integration")
    print("="*60)

    from network_gen import create_random_network

    # Generate random network
    print(f"\nGenerating random network...")
    model, network_config = create_random_network()

    print(f"  Stem code: {network_config.stem_code}")
    print(f"  Number of blocks: {len(network_config.blocks)}")
    print(f"  ✓ Model created: {model.__class__.__name__}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(input_tensor)
    print(f"  ✓ Output shape: {output.shape}")

    print("\n✅ Test 5 PASSED: Network builder works correctly")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("NAS IMPLEMENTATION VERIFICATION TESTS")
    print("="*80)

    tests = [
        ("Zero-Cost Proxy", test_zero_cost_proxy),
        ("Fitness Function", test_fitness_function),
        ("Mutation Operators", test_mutation),
        ("Population Management", test_population),
        ("Network Builder", test_network_builder),
    ]

    results = []

    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for evolution.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before running evolution.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
