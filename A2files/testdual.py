import numpy as np
from A2codes import minHinge, adjHinge, dualHinge
from A2helpers import linearKernel

def test_dualHinge_vs_primal():
    """
    Test dualHinge by comparing with minHinge and adjHinge using linear kernel.
    
    Debug hint from assignment:
    When using linear kernel, w* = (1/λ)X^T Δ(y)α*
    
    The w* from dualHinge should match w* from minHinge and adjHinge.
    """
    print("="*70)
    print("Testing dualHinge against Q1(b) minHinge and Q2(b) adjHinge")
    print("="*70)
    
    # Create simple test data
    np.random.seed(42)
    n = 20  # Small dataset for easy debugging
    d = 2   # 2D features
    
    X = np.random.randn(n, d)
    y = np.random.choice([-1, 1], size=n).reshape(-1, 1)
    lamb = 0.1
    
    print(f"\nTest data: n={n}, d={d}, λ={lamb}")
    print(f"y distribution: {np.sum(y==1)} positive, {np.sum(y==-1)} negative")
    
    # ===== Q1(b): Primal form =====
    print("\n" + "-"*70)
    print("Q1(b) - minHinge (Primal form)")
    print("-"*70)
    w_primal, w0_primal = minHinge(X, y, lamb)
    print(f"w_primal shape: {w_primal.shape}")
    print(f"w_primal: {w_primal.flatten()}")
    print(f"w0_primal: {w0_primal}")
    
    # ===== Q2(b): Adjoint form =====
    print("\n" + "-"*70)
    print("Q2(b) - adjHinge (Adjoint form with linear kernel)")
    print("-"*70)
    a_adj, a0_adj = adjHinge(X, y, lamb, linearKernel)
    
    # Recover w from adjoint: w* = (1/λ)X^T Δ(y)α*
    Delta_y = np.diag(y.flatten())
    w_adj = (1.0 / lamb) * X.T @ Delta_y @ a_adj
    
    print(f"a_adj shape: {a_adj.shape}")
    print(f"w_adj (recovered): {w_adj.flatten()}")
    print(f"a0_adj: {a0_adj}")
    
    # ===== Q3(a): Dual form =====
    print("\n" + "-"*70)
    print("Q3(a) - dualHinge (Dual form with linear kernel)")
    print("-"*70)
    a_dual, b_dual = dualHinge(X, y, lamb, linearKernel)
    
    # Recover w from dual: w* = (1/λ)X^T Δ(y)α*
    w_dual = (1.0 / lamb) * X.T @ Delta_y @ a_dual
    
    print(f"a_dual shape: {a_dual.shape}")
    print(f"w_dual (recovered): {w_dual.flatten()}")
    print(f"b_dual: {b_dual}")
    
    # ===== Compare w vectors =====
    print("\n" + "="*70)
    print("COMPARISON: w vectors should be similar")
    print("="*70)
    
    print(f"\nw_primal:  {w_primal.flatten()}")
    print(f"w_adj:     {w_adj.flatten()}")
    print(f"w_dual:    {w_dual.flatten()}")
    
    # Compute differences
    diff_primal_adj = np.linalg.norm(w_primal - w_adj)
    diff_primal_dual = np.linalg.norm(w_primal - w_dual)
    diff_adj_dual = np.linalg.norm(w_adj - w_dual)
    
    print(f"\n||w_primal - w_adj||:  {diff_primal_adj:.6f}")
    print(f"||w_primal - w_dual||: {diff_primal_dual:.6f}")
    print(f"||w_adj - w_dual||:    {diff_adj_dual:.6f}")
    
    # ===== Compare intercepts =====
    print("\n" + "="*70)
    print("COMPARISON: Intercepts (may differ slightly)")
    print("="*70)
    
    print(f"\nw0_primal: {w0_primal:.6f}")
    print(f"a0_adj:    {a0_adj:.6f}")
    print(f"b_dual:    {b_dual:.6f}")
    
    diff_b_primal_adj = abs(w0_primal - a0_adj)
    diff_b_primal_dual = abs(w0_primal - b_dual)
    diff_b_adj_dual = abs(a0_adj - b_dual)
    
    print(f"\n|w0_primal - a0_adj|:  {diff_b_primal_adj:.6f}")
    print(f"|w0_primal - b_dual|:  {diff_b_primal_dual:.6f}")
    print(f"|a0_adj - b_dual|:     {diff_b_adj_dual:.6f}")
    
    # ===== Test predictions =====
    print("\n" + "="*70)
    print("COMPARISON: Predictions should be identical")
    print("="*70)
    
    # Make predictions
    pred_primal = np.sign(X @ w_primal + w0_primal)
    pred_adj = np.sign(X @ w_adj + a0_adj)
    pred_dual = np.sign(X @ w_dual + b_dual)
    
    # Count agreements
    agree_primal_adj = np.sum(pred_primal == pred_adj)
    agree_primal_dual = np.sum(pred_primal == pred_dual)
    agree_adj_dual = np.sum(pred_adj == pred_dual)
    
    print(f"\nPrediction agreement (out of {n}):")
    print(f"  primal vs adj:  {agree_primal_adj}/{n} ({100*agree_primal_adj/n:.1f}%)")
    print(f"  primal vs dual: {agree_primal_dual}/{n} ({100*agree_primal_dual/n:.1f}%)")
    print(f"  adj vs dual:    {agree_adj_dual}/{n} ({100*agree_adj_dual/n:.1f}%)")
    
    # ===== Final verdict =====
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    tolerance_w = 1e-3
    tolerance_b = 0.5  # Intercepts can differ more
    
    w_match = (diff_primal_dual < tolerance_w) and (diff_adj_dual < tolerance_w)
    predictions_match = (agree_primal_dual == n) and (agree_adj_dual == n)
    
    if w_match and predictions_match:
        print("\n✅ SUCCESS! dualHinge produces correct results!")
        print(f"   - w vectors match within tolerance ({tolerance_w})")
        print(f"   - All predictions agree")
        return True
    else:
        print("\n❌ FAILURE! Issues detected:")
        if not w_match:
            print(f"   - w vectors differ by more than {tolerance_w}")
        if not predictions_match:
            print(f"   - Predictions don't all agree")
        return False

# Run the test
test_dualHinge_vs_primal()