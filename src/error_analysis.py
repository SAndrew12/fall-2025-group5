# error_analysis.py
# Comprehensive error analysis for BERT classifier

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_errors(model, X_test, y_test, y_pred=None, y_proba=None):
    """
    Comprehensive error analysis

    Args:
        model: Trained BERT model
        X_test: Test texts
        y_test: True labels
        y_pred: Predictions (will compute if None)
        y_proba: Probabilities (will compute if None)

    Returns:
        DataFrame with detailed error analysis
    """
    print("\n" + "=" * 80)
    print("DETAILED ERROR ANALYSIS")
    print("=" * 80)

    # Get predictions if not provided
    if y_pred is None:
        y_pred = model.predict(X_test)
    if y_proba is None:
        y_proba = model.predict_proba(X_test)

    # Convert to numpy arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Identify error types
    fn_mask = (y_test == 1) & (y_pred == 0)  # False Negatives
    fp_mask = (y_test == 0) & (y_pred == 1)  # False Positives
    tp_mask = (y_test == 1) & (y_pred == 1)  # True Positives
    tn_mask = (y_test == 0) & (y_pred == 0)  # True Negatives

    # Get texts and probabilities for each category
    fn_texts = X_test[fn_mask]
    fn_probs = y_proba[fn_mask, 1]

    fp_texts = X_test[fp_mask]
    fp_probs = y_proba[fp_mask, 1]

    tp_texts = X_test[tp_mask]
    tp_probs = y_proba[tp_mask, 1]

    tn_texts = X_test[tn_mask]
    tn_probs = y_proba[tn_mask, 1]

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print(f"\n📊 ERROR BREAKDOWN:")
    print(f"   False Negatives (FN): {fn_mask.sum()} ({fn_mask.sum() / len(y_test) * 100:.1f}%)")
    print(f"   False Positives (FP): {fp_mask.sum()} ({fp_mask.sum() / len(y_test) * 100:.1f}%)")
    print(f"   True Positives (TP): {tp_mask.sum()} ({tp_mask.sum() / len(y_test) * 100:.1f}%)")
    print(f"   True Negatives (TN): {tn_mask.sum()} ({tn_mask.sum() / len(y_test) * 100:.1f}%)")

    # ========================================================================
    # CONFIDENCE ANALYSIS
    # ========================================================================
    print(f"\n🎯 CONFIDENCE DISTRIBUTIONS:")

    print(f"\n   False Negatives (Missed Minority Class):")
    if len(fn_probs) > 0:
        print(f"      Mean confidence: {fn_probs.mean():.3f}")
        print(f"      Median: {np.median(fn_probs):.3f}")
        print(f"      Q1-Q3: {np.percentile(fn_probs, 25):.3f} - {np.percentile(fn_probs, 75):.3f}")
        print(f"      Min-Max: {fn_probs.min():.3f} - {fn_probs.max():.3f}")

        # Confidence buckets
        low_conf = (fn_probs < 0.3).sum()
        med_conf = ((fn_probs >= 0.3) & (fn_probs < 0.45)).sum()
        high_conf = (fn_probs >= 0.45).sum()
        print(f"      Confidence buckets:")
        print(f"         <0.30: {low_conf} ({low_conf / len(fn_probs) * 100:.1f}%)")
        print(f"         0.30-0.45: {med_conf} ({med_conf / len(fn_probs) * 100:.1f}%)")
        print(f"         ≥0.45: {high_conf} ({high_conf / len(fn_probs) * 100:.1f}%)")
    else:
        print("      No false negatives!")

    print(f"\n   False Positives (Incorrect Minority Predictions):")
    if len(fp_probs) > 0:
        print(f"      Mean confidence: {fp_probs.mean():.3f}")
        print(f"      Median: {np.median(fp_probs):.3f}")
        print(f"      Q1-Q3: {np.percentile(fp_probs, 25):.3f} - {np.percentile(fp_probs, 75):.3f}")
        print(f"      Min-Max: {fp_probs.min():.3f} - {fp_probs.max():.3f}")

        # Confidence buckets
        low_conf = (fp_probs < 0.55).sum()
        med_conf = ((fp_probs >= 0.55) & (fp_probs < 0.7)).sum()
        high_conf = (fp_probs >= 0.7).sum()
        print(f"      Confidence buckets:")
        print(f"         <0.55: {low_conf} ({low_conf / len(fp_probs) * 100:.1f}%)")
        print(f"         0.55-0.70: {med_conf} ({med_conf / len(fp_probs) * 100:.1f}%)")
        print(f"         ≥0.70: {high_conf} ({high_conf / len(fp_probs) * 100:.1f}%)")
    else:
        print("      No false positives!")

    print(f"\n   True Positives (Correct Minority Predictions):")
    if len(tp_probs) > 0:
        print(f"      Mean confidence: {tp_probs.mean():.3f}")
        print(f"      Median: {np.median(tp_probs):.3f}")

    # ========================================================================
    # SAMPLE ERRORS
    # ========================================================================
    print(f"\n📝 SAMPLE FALSE NEGATIVES (Lowest Confidence):")
    if len(fn_texts) > 0:
        sorted_fn = sorted(zip(fn_texts.tolist(), fn_probs), key=lambda x: x[1])
        for i, (text, prob) in enumerate(sorted_fn[:3], 1):
            print(f"\n   [{i}] Confidence: {prob:.3f} (threshold: {model.prediction_threshold:.2f})")
            print(f"       Text: {text[:300]}...")

    print(f"\n📝 SAMPLE FALSE POSITIVES (Highest Confidence):")
    if len(fp_texts) > 0:
        sorted_fp = sorted(zip(fp_texts.tolist(), fp_probs), key=lambda x: x[1], reverse=True)
        for i, (text, prob) in enumerate(sorted_fp[:3], 1):
            print(f"\n   [{i}] Confidence: {prob:.3f} (threshold: {model.prediction_threshold:.2f})")
            print(f"       Text: {text[:300]}...")

    # ========================================================================
    # TEXT LENGTH ANALYSIS
    # ========================================================================
    print(f"\n📏 TEXT LENGTH ANALYSIS:")

    fn_lengths = [len(str(t).split()) for t in fn_texts] if len(fn_texts) > 0 else []
    fp_lengths = [len(str(t).split()) for t in fp_texts] if len(fp_texts) > 0 else []
    tp_lengths = [len(str(t).split()) for t in tp_texts] if len(tp_texts) > 0 else []

    if len(fn_lengths) > 0:
        print(f"   FN avg length: {np.mean(fn_lengths):.1f} words")
    if len(fp_lengths) > 0:
        print(f"   FP avg length: {np.mean(fp_lengths):.1f} words")
    if len(tp_lengths) > 0:
        print(f"   TP avg length: {np.mean(tp_lengths):.1f} words")

    # ========================================================================
    # WORD FREQUENCY ANALYSIS
    # ========================================================================
    print(f"\n🔤 WORD FREQUENCY PATTERNS:")

    def get_word_freq(texts, top_n=15):
        """Get word frequency from texts"""
        if len(texts) == 0:
            return Counter()

        all_words = []
        for text in texts:
            words = str(text).lower().split()
            # Filter out very short words and common words
            words = [w for w in words if len(w) > 3 and w not in
                     ['that', 'this', 'with', 'from', 'have', 'were', 'been', 'their']]
            all_words.extend(words)

        return Counter(all_words)

    fn_counter = get_word_freq(fn_texts)
    fp_counter = get_word_freq(fp_texts)
    tp_counter = get_word_freq(tp_texts)

    if fn_counter:
        print(f"\n   Most common words in FALSE NEGATIVES:")
        for word, count in fn_counter.most_common(15):
            print(f"      - {word}: {count}")

    if fp_counter:
        print(f"\n   Most common words in FALSE POSITIVES:")
        for word, count in fp_counter.most_common(15):
            print(f"      - {word}: {count}")

    # ========================================================================
    # DISTINCTIVE WORDS
    # ========================================================================
    print(f"\n🎯 DISTINCTIVE WORDS (FN vs TP):")
    if fn_counter and tp_counter:
        # Normalize by total word count
        fn_total = sum(fn_counter.values())
        tp_total = sum(tp_counter.values())

        distinctive_fn = []
        for word, count in fn_counter.most_common(50):
            fn_freq = count / fn_total
            tp_freq = tp_counter.get(word, 0) / tp_total if tp_total > 0 else 0

            if fn_freq > tp_freq * 2:  # At least 2x more common in FN
                distinctive_fn.append((word, fn_freq, tp_freq))

        if distinctive_fn:
            print(f"\n   Words more common in MISSED samples (FN):")
            for word, fn_freq, tp_freq in distinctive_fn[:10]:
                print(f"      - {word}: {fn_freq:.3f} (FN) vs {tp_freq:.3f} (TP)")

    print(f"\n🎯 DISTINCTIVE WORDS (FP vs TN):")
    if fp_counter and tn_mask.sum() > 0:
        tn_counter = get_word_freq(tn_texts)

        # Normalize
        fp_total = sum(fp_counter.values())
        tn_total = sum(tn_counter.values())

        distinctive_fp = []
        for word, count in fp_counter.most_common(50):
            fp_freq = count / fp_total
            tn_freq = tn_counter.get(word, 0) / tn_total if tn_total > 0 else 0

            if fp_freq > tn_freq * 2:
                distinctive_fp.append((word, fp_freq, tn_freq))

        if distinctive_fp:
            print(f"\n   Words that trigger FALSE POSITIVES:")
            for word, fp_freq, tn_freq in distinctive_fp[:10]:
                print(f"      - {word}: {fp_freq:.3f} (FP) vs {tn_freq:.3f} (TN)")

    # ========================================================================
    # CREATE ERROR DATAFRAME
    # ========================================================================
    error_records = []

    # Add FN
    for text, prob in zip(fn_texts, fn_probs):
        error_records.append({
            'text': text,
            'true_label': 1,
            'pred_label': 0,
            'confidence': prob,
            'error_type': 'FN',
            'text_length': len(str(text).split())
        })

    # Add FP
    for text, prob in zip(fp_texts, fp_probs):
        error_records.append({
            'text': text,
            'true_label': 0,
            'pred_label': 1,
            'confidence': prob,
            'error_type': 'FP',
            'text_length': len(str(text).split())
        })

    error_df = pd.DataFrame(error_records)

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print(f"\n📊 Creating error visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Confidence distributions
    if len(fn_probs) > 0 and len(fp_probs) > 0:
        axes[0, 0].hist(fn_probs, bins=20, alpha=0.7, label='FN', color='red')
        axes[0, 0].hist(fp_probs, bins=20, alpha=0.7, label='FP', color='orange')
        axes[0, 0].axvline(model.prediction_threshold, color='black',
                           linestyle='--', label=f'Threshold: {model.prediction_threshold:.2f}')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Error Confidence Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # All predictions
    if len(fn_probs) > 0 and len(fp_probs) > 0 and len(tp_probs) > 0 and len(tn_probs) > 0:
        all_confs = [fn_probs, fp_probs, tp_probs, tn_probs]
        labels = ['FN', 'FP', 'TP', 'TN']
        colors = ['red', 'orange', 'green', 'blue']

        bp = axes[0, 1].boxplot(all_confs, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0, 1].axhline(model.prediction_threshold, color='black',
                           linestyle='--', label=f'Threshold')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].set_title('Confidence by Prediction Type')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Text length comparison
    if len(fn_lengths) > 0 and len(fp_lengths) > 0:
        axes[1, 0].hist(fn_lengths, bins=20, alpha=0.7, label='FN', color='red')
        axes[1, 0].hist(fp_lengths, bins=20, alpha=0.7, label='FP', color='orange')
        axes[1, 0].set_xlabel('Text Length (words)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Error Text Length Distributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Error counts
    error_counts = {
        'False\nNegatives': fn_mask.sum(),
        'False\nPositives': fp_mask.sum(),
        'True\nPositives': tp_mask.sum(),
        'True\nNegatives': tn_mask.sum()
    }

    colors_map = {'False\nNegatives': 'red', 'False\nPositives': 'orange',
                  'True\nPositives': 'green', 'True\nNegatives': 'blue'}

    bars = axes[1, 1].bar(error_counts.keys(), error_counts.values(),
                          color=[colors_map[k] for k in error_counts.keys()],
                          alpha=0.7)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Breakdown')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('error_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("   ✓ Error visualization saved to 'error_analysis_plots.png'")
    plt.close()

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 80)

    return error_df


def compare_models_errors(original_preds, improved_preds, y_test, X_test):
    """
    Compare errors between original and improved model

    Args:
        original_preds: Predictions from original model
        improved_preds: Predictions from improved model
        y_test: True labels
        X_test: Test texts
    """
    print("\n" + "=" * 80)
    print("COMPARING MODEL ERRORS")
    print("=" * 80)

    y_test = np.array(y_test)
    original_preds = np.array(original_preds)
    improved_preds = np.array(improved_preds)

    # Find cases where models differ
    differ_mask = original_preds != improved_preds

    # Cases where improved model fixed an error
    fixed_mask = differ_mask & (improved_preds == y_test) & (original_preds != y_test)

    # Cases where improved model introduced a new error
    broke_mask = differ_mask & (improved_preds != y_test) & (original_preds == y_test)

    # Cases where both are wrong but predicted differently
    both_wrong_mask = differ_mask & (improved_preds != y_test) & (original_preds != y_test)

    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"   Total test samples: {len(y_test)}")
    print(f"   Cases where models differ: {differ_mask.sum()} ({differ_mask.sum() / len(y_test) * 100:.1f}%)")
    print(f"\n   ✅ Fixed by improved model: {fixed_mask.sum()}")
    print(f"   ❌ Broken by improved model: {broke_mask.sum()}")
    print(f"   ⚠️  Both wrong (different predictions): {both_wrong_mask.sum()}")

    net_improvement = fixed_mask.sum() - broke_mask.sum()
    print(f"\n   Net improvement: {net_improvement:+d} samples")

    # Show samples
    if fixed_mask.sum() > 0:
        print(f"\n✅ SAMPLES FIXED BY IMPROVED MODEL:")
        fixed_texts = X_test[fixed_mask]
        fixed_true = y_test[fixed_mask]
        fixed_old = original_preds[fixed_mask]
        fixed_new = improved_preds[fixed_mask]

        for i, (text, true, old, new) in enumerate(zip(fixed_texts[:3], fixed_true, fixed_old, fixed_new), 1):
            print(f"\n   [{i}] True: {true}, Original: {old} → Improved: {new}")
            print(f"       {str(text)[:200]}...")

    if broke_mask.sum() > 0:
        print(f"\n❌ SAMPLES BROKEN BY IMPROVED MODEL:")
        broke_texts = X_test[broke_mask]
        broke_true = y_test[broke_mask]
        broke_old = original_preds[broke_mask]
        broke_new = improved_preds[broke_mask]

        for i, (text, true, old, new) in enumerate(zip(broke_texts[:3], broke_true, broke_old, broke_new), 1):
            print(f"\n   [{i}] True: {true}, Original: {old} → Improved: {new}")
            print(f"       {str(text)[:200]}...")

    print("\n" + "=" * 80)

    return {
        'fixed': fixed_mask.sum(),
        'broke': broke_mask.sum(),
        'net_improvement': net_improvement,
        'both_wrong': both_wrong_mask.sum()
    }