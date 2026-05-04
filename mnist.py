import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 1. LOAD THE DATA
# The 'digits' dataset contains 1,797 8x8 pixel images of hand-written digits.
digits = datasets.load_digits()

# 2. PRE-PROCESS THE IMAGES
# Machine learning models usually can't "see" a 2D grid (8x8). 
# We must 'flatten' each image into a single row of 64 numbers (8*8=64).
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 3. SPLIT DATA INTO TRAIN AND TEST SETS
# We train the model on one portion (70%) and save the rest (30%) 
# to test if it can recognize digits it has never seen before.
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False
)

# 4. INITIALIZE THE MODEL (The "Brain")
# SVC stands for Support Vector Classifier. 
# 'gamma' is a tuning parameter that determines how much influence 
# a single training example has. Small gamma means 'far' influence.
clf = svm.SVC(gamma=0.001)

# 5. TRAINING PHASE (The Learning)
# This is where the algorithm looks at the training images (X_train) 
# and their known labels (y_train) to learn the patterns of each digit.
clf.fit(X_train, y_train)

# 6. PREDICTION PHASE
# We ask the trained model to guess the digits for our test images.
predicted = clf.predict(X_test)

# 7. EVALUATION
# We compare the 'predicted' labels against the actual 'y_test' labels.
# This produces a report showing precision (accuracy for each digit).
print(f"Classification report:\n{metrics.classification_report(y_test, predicted)}")

# OPTIONAL: Visualizing a result
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
plt.title(f"Prediction: {predicted[-1]}")
plt.show()
