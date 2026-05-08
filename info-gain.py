import math
from collections import Counter

# 1. DEFINE THE PLAY TENNIS DATASET
#    Each record: [Outlook, Temperature, Humidity, Wind, PlayTennis
dataset = [
    ["Sunny",    "Hot",  "High",   "Weak",   "No"],
    ["Sunny",    "Hot",  "High",   "Strong", "No"],
    ["Overcast", "Hot",  "High",   "Weak",   "Yes"],
    ["Rain",     "Mild", "High",   "Weak",   "Yes"],
    ["Rain",     "Cool", "Normal", "Weak",   "Yes"],
    ["Rain",     "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny",    "Mild", "High",   "Weak",   "No"],
    ["Sunny",    "Cool", "Normal", "Weak",   "Yes"],
    ["Rain",     "Mild", "Normal", "Weak",   "Yes"],
    ["Sunny",    "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High",   "Strong", "Yes"],
    ["Overcast", "Hot",  "Normal", "Weak",   "Yes"],
    ["Rain",     "Mild", "High",   "Strong", "No"],
]

# Feature names and their column indices
feature_names = ["Outlook", "Temperature", "Humidity", "Wind"]
target_index = 4  # PlayTennis column

# 2. ENTROPY FUNCTION
#    Entropy(S) = -Σ p_i * log2(p_i)
def calculate_entropy(data):
    """
    Calculate the entropy of a dataset based on the target variable.

    Parameters:
        data (list): List of data records (each record is a list)

    Returns:
        float: Entropy value in bits
    """
    total = len(data)
    if total == 0:
        return 0

    # Count occurrences of each class (Yes/No)
    class_counts = Counter(row[target_index] for row in data)

    entropy = 0.0
    for count in class_counts.values():
        # Calculate probability of this class
        probability = count / total

        # Apply entropy formula: -p * log2(p)
        # We skip if probability is 0 (log2(0) is undefined)
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy

# 3. INFORMATION GAIN FUNCTION
#    InformationGain(S, A) = entropy(S) - Σ (|Sv|/|S|) * entropy(Sv)
#    Where:
#    - Sv is the subset where feature A has value v
#    - |Sv|/|S| is the proportion of examples with value 
def calculate_information_gain(data, feature_index):
    """
    Calculate the information gain of splitting on a given feature.

    Parameters:
        data (list): Full dataset
        feature_index (int): Column index of the feature to evaluate

    Returns:
        float: Information gain in bits
    """
    total = len(data)

    # Step 1: Calculate entropy of the entire dataset
    total_entropy = calculate_entropy(data)

    # Step 2: Get all unique values of this feature
    feature_values = set(row[feature_index] for row in data)

    # Step 3: Calculate weighted average entropy after splitting
    weighted_entropy = 0.0

    for value in sorted(feature_values):
        # Create subset where feature equals this value
        subset = [row for row in data if row[feature_index] == value]
        subset_size = len(subset)

        # Weight by proportion of examples in this subset
        weight = subset_size / total
        subset_entropy = calculate_entropy(subset)
        weighted_entropy += weight * subset_entropy

    # Step 4: Information Gain = Total Entropy - Weighted Entropy
    info_gain = total_entropy - weighted_entropy

    return info_gain

# Calculate and Display Result
if __name__ == "__main__":
    print("=" * 60)
    print("  PLAY TENNIS — Entropy & Information Gain Calculator")
    print("=" * 60)

    # Calculate and display the overall entropy
    overall_entropy = calculate_entropy(dataset)
    print(f"\n  Overall Entropy H(S) = {overall_entropy:.4f} bits")
    print(f"  (Yes: 9, No: 5, Total: 14)")

    print("\n" + "-" * 60)
    print("  Information Gain for Each Feature:")
    print("-" * 60)

    # Calculate information gain for each feature
    results = []
    for i, feature_name in enumerate(feature_names):
        ig = calculate_information_gain(dataset, i)
        results.append((feature_name, ig))

        # Show  breakdown
        print(f"\n  Feature: {feature_name}")

        # Show value distribution
        feature_values = set(row[i] for row in dataset)
        for value in sorted(feature_values):
            subset = [row for row in dataset if row[i] == value]
            yes_count = sum(1 for row in subset if row[target_index] == "Yes")
            no_count = sum(1 for row in subset if row[target_index] == "No")
            subset_entropy = calculate_entropy(subset)
            print(f"    {value:>10}: Yes={yes_count}, No={no_count}, "
                  f"Total={len(subset)}, Entropy={subset_entropy:.4f}")

        print(f"    → Information Gain = {ig:.4f} bits")

    # Summary table sorted by information gain
    print("\n" + "=" * 60)
    print("  SUMMARY — Features Ranked by Information Gain")
    print("=" * 60)

    results.sort(key=lambda x: x[1], reverse=True)  # Sort descending

    print(f"\n  {'Rank':<6} {'Feature':<15} {'Info Gain (bits)':<18}")
    print(f"  {'-'*6} {'-'*15} {'-'*18}")

    for rank, (name, ig) in enumerate(results, 1):
        marker = " ← Best split" if rank == 1 else ""
        print(f"  {rank:<6} {name:<15} {ig:<18.4f}{marker}")

    # Identify best feature
    best_feature = results[0][0]
    print(f"\n  ✅ Best feature to split on: {best_feature}")
    print(f"     (Highest information gain = {results[0][1]:.4f} bits)")
    print()