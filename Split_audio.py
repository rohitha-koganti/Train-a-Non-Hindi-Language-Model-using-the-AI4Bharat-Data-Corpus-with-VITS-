from sklearn.model_selection import train_test_split

X = 281474976883728_f1099_chunk_0.wav  # Assign your data to X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Assuming X is your feature matrix and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
