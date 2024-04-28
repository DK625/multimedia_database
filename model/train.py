import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load extracted features and labels
with open('../storage/shape_features.pkl', 'rb') as f:
    shape_features = pickle.load(f)

with open('../storage/environmental_features.pkl', 'rb') as f:
    environmental_features = pickle.load(f)

with open('../storage/color_features.pkl', 'rb') as f:
    color_features = pickle.load(f)

with open('../storage/object_features.pkl', 'rb') as f:
    object_features = pickle.load(f)

# Combine features into a single array
features = np.concatenate((shape_features, environmental_features, color_features, object_features), axis=1)

with open('../storage/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Train SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict labels for test set
y_pred = svm_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save SVM model to file
with open('../storage/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_classifier, f)
