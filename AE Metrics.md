Here's a detailed explanation of each metric used in the accuracy measurement:
Detection Metrics (Object-Level)
Precision

What it measures: Of all the objects your model predicted, how many were actually correct?
Formula: True Positives / (True Positives + False Positives)
Golf Course Example: If your model predicts 10 bunkers but only 7 are actually bunkers, precision = 7/10 = 0.7
Good Score: Higher is better (0-1 scale). 0.8+ is good.

Recall (Sensitivity)

What it measures: Of all the actual objects in the image, how many did your model find?
Formula: True Positives / (True Positives + False Negatives)
Golf Course Example: If there are 8 actual bunkers in the image but your model only finds 6, recall = 6/8 = 0.75
Good Score: Higher is better (0-1 scale). 0.8+ is good.

F1-Score

What it measures: Harmonic mean of precision and recall - balances both metrics
Formula: 2 × (Precision × Recall) / (Precision + Recall)
Why it's useful: A model could have high precision but low recall (finds few objects but correctly), or high recall but low precision (finds all objects but many wrong). F1 balances both.
Good Score: 0.7+ is good, 0.8+ is excellent

Segmentation Metrics (Pixel-Level)
IoU (Intersection over Union)

What it measures: How well do the predicted masks overlap with the true masks?
Formula: Area of Overlap / Area of Union
Good Score: 0.5+ is acceptable, 0.7+ is good, 0.8+ is excellent

Mean IoU (mIoU)

What it measures: Average IoU across all classes
Why important: In golf course segmentation, this tells you how accurately your model outlines each feature (greens, fairways, bunkers, etc.)
Good Score: Same as IoU (0.5+ acceptable, 0.7+ good)

Pixel Accuracy

What it measures: Percentage of pixels classified correctly
Formula: Correct Pixels / Total Pixels
Limitation: Can be misleading if classes are imbalanced (e.g., if 90% of image is fairway, you could get 90% accuracy by just predicting everything as fairway)
Good Score: 0.8+ but interpret carefully

Confusion Matrix Components
True Positives (TP)

Meaning: Model correctly identified an object (e.g., correctly found a bunker)
Example: Model predicts bunker, and there actually is a bunker there

False Positives (FP)

Meaning: Model incorrectly identified an object that wasn't there
Example: Model predicts bunker, but it's actually rough grass

False Negatives (FN)

Meaning: Model missed an object that was actually there
Example: There's a water hazard in the image, but model didn't detect it

True Negatives (TN)

Meaning: Model correctly identified absence of an object
Note: Less commonly reported in object detection

Practical Golf Course Examples
High Precision, Low Recall Scenario:

Model finds 3 bunkers with 100% accuracy, but misses 5 other bunkers
Precision = 3/3 = 1.0 (perfect)
Recall = 3/8 = 0.375 (poor)
Problem: Conservative model that only detects obvious cases

Low Precision, High Recall Scenario:

Model detects 12 bunkers, but 5 are actually rough areas
Precision = 7/12 = 0.58 (poor)
Recall = 7/8 = 0.875 (good)
Problem: Aggressive model that over-detects

Good Balance:

Model detects 7 out of 8 bunkers correctly, with 1 false positive
Precision = 7/8 = 0.875
Recall = 7/8 = 0.875
F1-Score = 0.875
This is what you want!

Which Metrics Matter Most for Golf Course Mapping?

F1-Score: Best overall indicator of model performance
Mean IoU: Critical for segmentation quality - tells you how precisely boundaries are drawn
Per-class Recall: Important to ensure you don't miss critical features (especially water hazards for safety)
Per-class Precision: Important to avoid false alarms that could mislead navigation

Interpreting Your Results

F1 > 0.8: Excellent model
F1 0.6-0.8: Good model, may need some tuning
F1 < 0.6: Needs significant improvement
High IoU (>0.7): Boundaries are well-defined
Low IoU (<0.5): Boundaries are poorly defined, may need more training data or different model architecture

The key is to look at these metrics together, not in isolation, to get a complete picture of your model's performance.