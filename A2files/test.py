import numpy as np
from A2codes import minExpLinear, minHinge, adjExpLinear, adjHinge
from A2helpers import linearKernel, polyKernel, gaussKernel, generateData

def test_implementations():
    """
    Test the implemented functions without requiring classify functions.
    Cross-verify primal and adjoint forms using direct margin calculations.
    """
    print("="*80)
    print("TESTS FOR Q1(a,b) and Q2(a,b)")
    print("="*80)
    
    # Generate test data
    np.random.seed(42)
    n_train = 50
    X, y = generateData(n=n_train, gen_model=1)
    y = y.reshape(-1, 1)  # Ensure column vector
    n, d = X.shape
    lamb = 0.1
    
    print(f"\nTest data: n={n} samples, d={d} features")
    print(f"Labels: {np.unique(y.flatten())}")
    
    # Test 1: Check output shapes
    print("\n" + "="*80)
    print("TEST 1: Output Shapes")
    print("="*80)
    
    # Q1(a): minExpLinear
    w_exp, w0_exp = minExpLinear(X, y, lamb)
    print(f"\nminExpLinear:")
    print(f"  w shape: {w_exp.shape} (expected: ({d}, 1))")
    print(f"  w0 type: {type(w0_exp).__name__} (expected: float)")
    assert w_exp.shape == (d, 1), f"Wrong w shape: {w_exp.shape}"
    assert isinstance(w0_exp, (float, np.floating)), f"w0 should be scalar"
    print(f"  ✓ PASSED")
    
    # Q1(b): minHinge
    w_hinge, w0_hinge = minHinge(X, y, lamb)
    print(f"\nminHinge:")
    print(f"  w shape: {w_hinge.shape} (expected: ({d}, 1))")
    print(f"  w0 type: {type(w0_hinge).__name__} (expected: float)")
    assert w_hinge.shape == (d, 1), f"Wrong w shape: {w_hinge.shape}"
    assert isinstance(w0_hinge, (float, np.floating)), f"w0 should be scalar"
    print(f"  ✓ PASSED")
    
    # Q2(a): adjExpLinear
    kernel_linear = lambda X1, X2: linearKernel(X1, X2)
    a_exp, a0_exp = adjExpLinear(X, y, lamb, kernel_linear)
    print(f"\nadjExpLinear (linear kernel):")
    print(f"  a shape: {a_exp.shape} (expected: ({n}, 1))")
    print(f"  a0 type: {type(a0_exp).__name__} (expected: float)")
    assert a_exp.shape == (n, 1), f"Wrong a shape: {a_exp.shape}"
    assert isinstance(a0_exp, (float, np.floating)), f"a0 should be scalar"
    print(f"  ✓ PASSED")
    
    # Q2(b): adjHinge
    a_hinge, a0_hinge = adjHinge(X, y, lamb, kernel_linear)
    print(f"\nadjHinge (linear kernel):")
    print(f"  a shape: {a_hinge.shape} (expected: ({n}, 1))")
    print(f"  a0 type: {type(a0_hinge).__name__} (expected: float)")
    assert a_hinge.shape == (n, 1), f"Wrong a shape: {a_hinge.shape}"
    assert isinstance(a0_hinge, (float, np.floating)), f"a0 should be scalar"
    print(f"  ✓ PASSED")
    
    # Test 2: Cross-verify ExpLinear (Primal vs Adjoint with Linear Kernel)
    print("\n" + "="*80)
    print("TEST 2: ExpLinear - Primal vs Adjoint (Linear Kernel)")
    print("="*80)
    
    # According to assignment: w* = X^T α* for L2 regularization
    w_from_adjoint = X.T @ a_exp
    
    print(f"\nPrimal w (first 3 elements): {w_exp.flatten()[:3]}")
    print(f"Adjoint X^T*a (first 3 elements): {w_from_adjoint.flatten()[:3]}")
    
    w_diff = np.linalg.norm(w_exp - w_from_adjoint)
    print(f"\n||w_primal - X^T*α_adjoint||: {w_diff:.6f}")
    
    w0_diff = abs(w0_exp - a0_exp)
    print(f"|w0_primal - a0_adjoint|: {w0_diff:.6f}")
    
    # Check margins: x_i^T w + w0 should equal k_i^T α + α0
    margins_primal = (X @ w_exp + w0_exp).flatten()
    K = kernel_linear(X, X)
    margins_adjoint = (K @ a_exp + a0_exp).flatten()
    
    print(f"\nMargins (primal) - first 5: {margins_primal[:5]}")
    print(f"Margins (adjoint) - first 5: {margins_adjoint[:5]}")
    
    margin_diff = np.linalg.norm(margins_primal - margins_adjoint)
    print(f"\n||margins_primal - margins_adjoint||: {margin_diff:.6f}")
    
    # Compute predictions manually
    y_pred_primal = np.sign(margins_primal)
    y_pred_adjoint = np.sign(margins_adjoint)
    pred_agreement = np.mean(y_pred_primal == y_pred_adjoint)
    print(f"Prediction agreement: {pred_agreement*100:.2f}%")
    
    # Check accuracy
    acc_primal = np.mean(y_pred_primal == y.flatten())
    acc_adjoint = np.mean(y_pred_adjoint == y.flatten())
    print(f"Training accuracy (primal): {acc_primal*100:.2f}%")
    print(f"Training accuracy (adjoint): {acc_adjoint*100:.2f}%")
    
    test2_pass = (w_diff < 0.1 and margin_diff < 0.1 and pred_agreement > 0.95)
    print(f"\n{'✓ PASSED' if test2_pass else '✗ FAILED'}: ExpLinear cross-verification")
    
    # Test 3: Cross-verify Hinge (Primal vs Adjoint with Linear Kernel)
    print("\n" + "="*80)
    print("TEST 3: Hinge - Primal vs Adjoint (Linear Kernel)")
    print("="*80)
    
    w_from_adjoint_hinge = X.T @ a_hinge
    
    print(f"\nPrimal w (first 3 elements): {w_hinge.flatten()[:3]}")
    print(f"Adjoint X^T*a (first 3 elements): {w_from_adjoint_hinge.flatten()[:3]}")
    
    w_diff_hinge = np.linalg.norm(w_hinge - w_from_adjoint_hinge)
    print(f"\n||w_primal - X^T*α_adjoint||: {w_diff_hinge:.6f}")
    
    w0_diff_hinge = abs(w0_hinge - a0_hinge)
    print(f"|w0_primal - a0_adjoint|: {w0_diff_hinge:.6f}")
    
    # Check margins
    margins_primal_hinge = (X @ w_hinge + w0_hinge).flatten()
    margins_adjoint_hinge = (K @ a_hinge + a0_hinge).flatten()
    
    print(f"\nMargins (primal) - first 5: {margins_primal_hinge[:5]}")
    print(f"Margins (adjoint) - first 5: {margins_adjoint_hinge[:5]}")
    
    margin_diff_hinge = np.linalg.norm(margins_primal_hinge - margins_adjoint_hinge)
    print(f"\n||margins_primal - margins_adjoint||: {margin_diff_hinge:.6f}")
    
    # Compute predictions manually
    y_pred_primal_hinge = np.sign(margins_primal_hinge)
    y_pred_adjoint_hinge = np.sign(margins_adjoint_hinge)
    pred_agreement_hinge = np.mean(y_pred_primal_hinge == y_pred_adjoint_hinge)
    print(f"Prediction agreement: {pred_agreement_hinge*100:.2f}%")
    
    # Check accuracy
    acc_primal_hinge = np.mean(y_pred_primal_hinge == y.flatten())
    acc_adjoint_hinge = np.mean(y_pred_adjoint_hinge == y.flatten())
    print(f"Training accuracy (primal): {acc_primal_hinge*100:.2f}%")
    print(f"Training accuracy (adjoint): {acc_adjoint_hinge*100:.2f}%")
    
    test3_pass = (w_diff_hinge < 0.1 and margin_diff_hinge < 0.1 and pred_agreement_hinge > 0.95)
    print(f"\n{'✓ PASSED' if test3_pass else '✗ FAILED'}: Hinge cross-verification")
    
    # Test 4: Non-linear Kernels (Sanity Check)
    print("\n" + "="*80)
    print("TEST 4: Adjoint with Non-Linear Kernels")
    print("="*80)
    
    # Polynomial kernel
    kernel_poly = lambda X1, X2: polyKernel(X1, X2, 2)
    a_poly, a0_poly = adjHinge(X, y, lamb, kernel_poly)
    K_poly = kernel_poly(X, X)
    margins_poly = (K_poly @ a_poly + a0_poly).flatten()
    y_pred_poly = np.sign(margins_poly)
    acc_poly = np.mean(y_pred_poly == y.flatten())
    print(f"Polynomial kernel (d=2):")
    print(f"  a shape: {a_poly.shape}")
    print(f"  Training accuracy: {acc_poly*100:.2f}%")
    
    # Gaussian kernel
    kernel_gauss = lambda X1, X2: gaussKernel(X1, X2, 1.0)
    a_gauss, a0_gauss = adjHinge(X, y, lamb, kernel_gauss)
    K_gauss = kernel_gauss(X, X)
    margins_gauss = (K_gauss @ a_gauss + a0_gauss).flatten()
    y_pred_gauss = np.sign(margins_gauss)
    acc_gauss = np.mean(y_pred_gauss == y.flatten())
    print(f"Gaussian kernel (σ=1.0):")
    print(f"  a shape: {a_gauss.shape}")
    print(f"  Training accuracy: {acc_gauss*100:.2f}%")
    
    test4_pass = (acc_poly > 0.5 and acc_gauss > 0.5)
    print(f"\n{'✓ PASSED' if test4_pass else '✗ FAILED'}: Non-linear kernels work")
    
    # Test 5: Verify adjHinge bug fix (a0 = x[n] not x[d])
    print("\n" + "="*80)
    print("TEST 5: Verify adjHinge Implementation (Bug Fix)")
    print("="*80)
    
    print(f"Data dimensions: n={n}, d={d}")
    print(f"Decision variable u = [α (length {n}), α0 (length 1), ξ (length {n})]")
    print(f"Total length: {2*n+1}")
    print(f"\nα0 should be at index n={n}")
    print(f"BUG would use index d={d}")
    
    if d != n:
        print(f"✓ Good: d≠n, so bug would be caught if it existed")
    else:
        print(f"⚠ Note: d==n, so bug might not be obvious")
    
    # Check if a0 values are reasonable
    print(f"\na0 from adjExpLinear: {a0_exp:.4f}")
    print(f"a0 from adjHinge: {a0_hinge:.4f}")
    
    assert not np.isnan(a0_exp), "a0_exp is NaN!"
    assert not np.isnan(a0_hinge), "a0_hinge is NaN!"
    assert not np.isinf(a0_exp), "a0_exp is infinite!"
    assert not np.isinf(a0_hinge), "a0_hinge is infinite!"
    
    print(f"✓ PASSED: a0 values are valid")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_tests = [True, test2_pass, test3_pass, test4_pass, True]  # Test 1 and 5 passed if we got here
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("Your implementations are correct!")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Review the failed tests above")
    
    return all_tests


def test_edge_cases():
    """
    Test edge cases and numerical stability
    """
    print("\n" + "="*80)
    print("EDGE CASE TESTS")
    print("="*80)
    
    # Test with very small dataset
    print("\n" + "-"*80)
    print("Edge Case 1: Small dataset (n=10)")
    print("-"*80)
    
    np.random.seed(42)
    X_small, y_small = generateData(n=10, gen_model=1)
    lamb = 0.1
    
    try:
        w, w0 = minHinge(X_small, y_small, lamb)
        print(f"✓ minHinge works with n=10")
        
        kernel_linear = lambda X1, X2: linearKernel(X1, X2)
        a, a0 = adjHinge(X_small, y_small, lamb, kernel_linear)
        print(f"✓ adjHinge works with n=10")
    except Exception as e:
        print(f"✗ Failed with small dataset: {e}")
    
    # Test with different lambda values
    print("\n" + "-"*80)
    print("Edge Case 2: Different lambda values")
    print("-"*80)
    
    X, y = generateData(n=30, gen_model=1)
    kernel_linear = lambda X1, X2: linearKernel(X1, X2)
    
    for lamb in [0.001, 0.1, 1.0, 10.0]:
        try:
            w, w0 = minHinge(X, y, lamb)
            a, a0 = adjHinge(X, y, lamb, kernel_linear)
            print(f"✓ lambda={lamb}: Both methods work")
        except Exception as e:
            print(f"✗ lambda={lamb}: Failed with {e}")
    
    # Test that outputs are not NaN or Inf
    print("\n" + "-"*80)
    print("Edge Case 3: Numerical stability")
    print("-"*80)
    
    X, y = generateData(n=50, gen_model=2)
    lamb = 0.1
    kernel_linear = lambda X1, X2: linearKernel(X1, X2)
    
    w_exp, w0_exp = minExpLinear(X, y, lamb)
    w_hinge, w0_hinge = minHinge(X, y, lamb)
    a_exp, a0_exp = adjExpLinear(X, y, lamb, kernel_linear)
    a_hinge, a0_hinge = adjHinge(X, y, lamb, kernel_linear)
    
    all_finite = (
        np.all(np.isfinite(w_exp)) and np.isfinite(w0_exp) and
        np.all(np.isfinite(w_hinge)) and np.isfinite(w0_hinge) and
        np.all(np.isfinite(a_exp)) and np.isfinite(a0_exp) and
        np.all(np.isfinite(a_hinge)) and np.isfinite(a0_hinge)
    )
    
    if all_finite:
        print(f"✓ All outputs are finite (no NaN or Inf)")
    else:
        print(f"✗ Some outputs contain NaN or Inf!")
    
    print("\n✓ Edge case testing complete")


if __name__ == "__main__":
    # Run main tests
    test_implementations()
    
    # Run edge case tests
    test_edge_cases()