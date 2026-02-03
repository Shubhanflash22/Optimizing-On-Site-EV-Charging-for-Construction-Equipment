"""
Quick Validation Script - Analyze Training Results
===================================================

This script analyzes your training history to check for:
1. Overfitting (train/val gap)
2. Per-class performance
3. Training curve quality
4. Potential issues

Run this immediately after training to validate your 98.14% accuracy!
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================
# Configuration
# ============================
HISTORY_PATH = Path(r"C:\Users\shubh\Desktop\DELETE AFTER USE\training_history.json")
OUTPUT_DIR = Path(r"C:\Users\shubh\Desktop\DELETE AFTER USE\validation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

ACTIVITY_NAMES = ['digging', 'idling', 'loading', 'swinging', 'travelling']

# ============================
# Load Training History
# ============================
print("="*60)
print("TRAINING RESULTS VALIDATION")
print("="*60)

with open(HISTORY_PATH, 'r') as f:
    history = json.load(f)

num_epochs = len(history['train_acc'])
print(f"\nLoaded training history: {num_epochs} epochs")

# ============================
# 1. CHECK FOR OVERFITTING
# ============================
print("\n" + "="*60)
print("1. OVERFITTING CHECK")
print("="*60)

final_train_acc = history['train_acc'][-1]
final_val_acc = history['val_acc'][-1]
gap = final_train_acc - final_val_acc

print(f"\nFinal Epoch Performance:")
print(f"  Training Accuracy:   {final_train_acc:.2f}%")
print(f"  Validation Accuracy: {final_val_acc:.2f}%")
print(f"  Gap:                 {gap:.2f}%")

if gap < 2:
    print(f"\n✅ EXCELLENT: Gap < 2% - Perfect generalization!")
elif gap < 5:
    print(f"\n✅ GOOD: Gap < 5% - Healthy generalization")
elif gap < 10:
    print(f"\n⚠️  ACCEPTABLE: Gap < 10% - Slight overfitting")
else:
    print(f"\n🚨 WARNING: Gap > 10% - Significant overfitting detected!")
    print("   Consider: More augmentation, dropout, or more training data")

# Check maximum gap throughout training
max_gap = max(history['train_acc'][i] - history['val_acc'][i] 
              for i in range(num_epochs))
print(f"\nMaximum gap during training: {max_gap:.2f}%")

# ============================
# 2. PER-CLASS PERFORMANCE
# ============================
print("\n" + "="*60)
print("2. PER-CLASS PERFORMANCE")
print("="*60)

final_precision = np.array(history['val_precision'][-1])
final_recall = np.array(history['val_recall'][-1])

# Calculate F1 scores
f1_scores = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-10)

print(f"\n{'Activity':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 55)

issues = []
for i, activity in enumerate(ACTIVITY_NAMES):
    precision = final_precision[i] * 100
    recall = final_recall[i] * 100
    f1 = f1_scores[i] * 100
    
    # Flag potential issues
    status = "✅"
    if precision < 90 or recall < 90:
        status = "⚠️ "
        issues.append(f"{activity}: P={precision:.1f}% R={recall:.1f}%")
    if precision < 80 or recall < 80:
        status = "🚨"
    
    print(f"{status} {activity:<12} {precision:>10.1f}%  {recall:>10.1f}%  {f1:>10.1f}%")

# Average metrics
avg_precision = np.mean(final_precision) * 100
avg_recall = np.mean(final_recall) * 100
avg_f1 = np.mean(f1_scores) * 100

print("-" * 55)
print(f"   {'Average':<12} {avg_precision:>10.1f}%  {avg_recall:>10.1f}%  {avg_f1:>10.1f}%")

if not issues:
    print("\n✅ EXCELLENT: All classes performing above 90%!")
else:
    print(f"\n⚠️  ATTENTION: Some classes need review:")
    for issue in issues:
        print(f"   - {issue}")

# ============================
# 3. PLOT TRAINING CURVES
# ============================
print("\n" + "="*60)
print("3. GENERATING TRAINING CURVES")
print("="*60)

fig = plt.figure(figsize=(18, 10))

epochs = range(1, num_epochs + 1)

# 3.1 Accuracy Plot
ax1 = plt.subplot(2, 3, 1)
ax1.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training')
ax1.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('Accuracy Over Time', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add annotations
best_val_acc = max(history['val_acc'])
best_epoch = history['val_acc'].index(best_val_acc) + 1
ax1.plot(best_epoch, best_val_acc, 'r*', markersize=15)
ax1.annotate(f'Best: {best_val_acc:.2f}%', 
            xy=(best_epoch, best_val_acc),
            xytext=(best_epoch, best_val_acc - 5),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 3.2 Loss Plot
ax2 = plt.subplot(2, 3, 2)
ax2.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training')
ax2.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Loss Over Time', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3.3 Overfitting Gap Plot
ax3 = plt.subplot(2, 3, 3)
gaps = [history['train_acc'][i] - history['val_acc'][i] for i in range(num_epochs)]
ax3.plot(epochs, gaps, 'purple', linewidth=2)
ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% threshold')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Train - Val Gap (%)', fontsize=11)
ax3.set_title('Generalization Gap', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.fill_between(epochs, 0, gaps, alpha=0.3, color='purple')

# 3.4 Per-Class Precision Over Time
ax4 = plt.subplot(2, 3, 4)
for i, activity in enumerate(ACTIVITY_NAMES):
    precisions = [epoch[i] * 100 for epoch in history['val_precision']]
    ax4.plot(epochs, precisions, linewidth=2, label=activity, marker='o', markersize=3)
ax4.set_xlabel('Epoch', fontsize=11)
ax4.set_ylabel('Precision (%)', fontsize=11)
ax4.set_title('Per-Class Precision', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 3.5 Per-Class Recall Over Time
ax5 = plt.subplot(2, 3, 5)
for i, activity in enumerate(ACTIVITY_NAMES):
    recalls = [epoch[i] * 100 for epoch in history['val_recall']]
    ax5.plot(epochs, recalls, linewidth=2, label=activity, marker='o', markersize=3)
ax5.set_xlabel('Epoch', fontsize=11)
ax5.set_ylabel('Recall (%)', fontsize=11)
ax5.set_title('Per-Class Recall', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 3.6 Final Confusion-Style Bar Chart
ax6 = plt.subplot(2, 3, 6)
x = np.arange(len(ACTIVITY_NAMES))
width = 0.35
ax6.bar(x - width/2, final_precision * 100, width, label='Precision', alpha=0.8)
ax6.bar(x + width/2, final_recall * 100, width, label='Recall', alpha=0.8)
ax6.set_xlabel('Activity', fontsize=11)
ax6.set_ylabel('Performance (%)', fontsize=11)
ax6.set_title('Final Per-Class Metrics', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(ACTIVITY_NAMES, rotation=45, ha='right', fontsize=9)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_ylim([75, 105])  # Focus on high performance range

plt.suptitle('Training Analysis Dashboard - 98.14% Validation Accuracy', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
plot_path = OUTPUT_DIR / "training_analysis_dashboard.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Dashboard saved to: {plot_path}")

# ============================
# 4. CONVERGENCE ANALYSIS
# ============================
print("\n" + "="*60)
print("4. CONVERGENCE ANALYSIS")
print("="*60)

# Check if validation loss is still decreasing
recent_val_losses = history['val_loss'][-5:]  # Last 5 epochs
is_decreasing = all(recent_val_losses[i] >= recent_val_losses[i+1] 
                   for i in range(len(recent_val_losses)-1))

if is_decreasing:
    print("\n📉 Validation loss still decreasing")
    print("   Consider: Training for more epochs might improve performance")
else:
    print("\n📊 Validation loss has plateaued")
    print("   ✅ Model has likely converged")

# Check stability in last 3 epochs
last_3_val_acc = history['val_acc'][-3:]
std_val_acc = np.std(last_3_val_acc)

if std_val_acc < 0.5:
    print(f"\n✅ Validation accuracy is stable (std: {std_val_acc:.3f}%)")
else:
    print(f"\n⚠️  Validation accuracy is fluctuating (std: {std_val_acc:.3f}%)")

# ============================
# 5. LEARNING RATE ANALYSIS
# ============================
print("\n" + "="*60)
print("5. TRAINING DYNAMICS")
print("="*60)

# Check improvement rate
early_val_acc = np.mean(history['val_acc'][:3])  # First 3 epochs
late_val_acc = np.mean(history['val_acc'][-3:])  # Last 3 epochs
total_improvement = late_val_acc - early_val_acc

print(f"\nValidation accuracy improvement:")
print(f"  Early epochs (1-3):  {early_val_acc:.2f}%")
print(f"  Late epochs (-3):    {late_val_acc:.2f}%")
print(f"  Total improvement:   {total_improvement:.2f}%")

if total_improvement > 10:
    print("  ✅ Excellent learning progress")
elif total_improvement > 5:
    print("  ✅ Good learning progress")
else:
    print("  ⚠️  Limited improvement - might have started near optimal")

# ============================
# 6. GENERATE SUMMARY REPORT
# ============================
print("\n" + "="*60)
print("6. SUMMARY REPORT")
print("="*60)

summary = {
    "final_validation_accuracy": final_val_acc,
    "final_training_accuracy": final_train_acc,
    "overfitting_gap": gap,
    "per_class_precision": {
        ACTIVITY_NAMES[i]: float(final_precision[i]) 
        for i in range(len(ACTIVITY_NAMES))
    },
    "per_class_recall": {
        ACTIVITY_NAMES[i]: float(final_recall[i]) 
        for i in range(len(ACTIVITY_NAMES))
    },
    "average_precision": float(avg_precision / 100),
    "average_recall": float(avg_recall / 100),
    "num_epochs": num_epochs,
    "best_epoch": best_epoch,
    "best_validation_accuracy": best_val_acc,
    "final_validation_loss": history['val_loss'][-1],
    "convergence_status": "Converged" if not is_decreasing else "Could improve further"
}

summary_path = OUTPUT_DIR / "validation_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Summary saved to: {summary_path}")

# ============================
# 7. RECOMMENDATIONS
# ============================
print("\n" + "="*60)
print("7. RECOMMENDATIONS")
print("="*60)

recommendations = []

# Based on overfitting
if gap > 10:
    recommendations.append("🔴 HIGH: Address overfitting - add more augmentation or training data")
elif gap > 5:
    recommendations.append("🟡 MEDIUM: Monitor for overfitting in future training")
else:
    recommendations.append("✅ Generalization is excellent")

# Based on convergence
if is_decreasing:
    recommendations.append("🟡 MEDIUM: Consider training for 5-10 more epochs")
else:
    recommendations.append("✅ Model has converged")

# Based on per-class performance
if issues:
    recommendations.append(f"🟡 MEDIUM: Improve performance for: {', '.join([i.split(':')[0] for i in issues])}")
else:
    recommendations.append("✅ All classes performing excellently")

# Next steps
recommendations.append("🟢 LOW: Test on completely new videos to validate generalization")
recommendations.append("🟢 LOW: Apply majority voting and measure improvement")
recommendations.append("🟢 LOW: Calculate productivity on real work videos")

print("\nAction Items:")
for rec in recommendations:
    print(f"  {rec}")

# ============================
# 8. FINAL VERDICT
# ============================
print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)

if final_val_acc >= 95 and gap < 5 and not issues:
    verdict = "🌟 OUTSTANDING"
    message = "Your model is performing exceptionally well with excellent generalization!"
elif final_val_acc >= 90 and gap < 8:
    verdict = "✅ EXCELLENT"
    message = "Your model is performing very well and ready for deployment testing."
elif final_val_acc >= 85:
    verdict = "✅ GOOD"
    message = "Your model is performing well, minor improvements could help."
else:
    verdict = "⚠️  NEEDS IMPROVEMENT"
    message = "Consider retraining with adjustments."

print(f"\n{verdict}")
print(f"{message}")

print(f"\n📊 Final Metrics:")
print(f"   Validation Accuracy: {final_val_acc:.2f}%")
print(f"   Train/Val Gap:       {gap:.2f}%")
print(f"   Average Precision:   {avg_precision:.2f}%")
print(f"   Average Recall:      {avg_recall:.2f}%")

print("\n" + "="*60)
print("✅ VALIDATION COMPLETE")
print("="*60)
print(f"\nResults saved to: {OUTPUT_DIR}")
print("Review the dashboard visualization for detailed analysis.")
print("\nNext: Test on new videos with resnet3d_inference_corrected.py")
