import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split

def train_oracle(X, y):
    # Load dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model definition
    model = Sequential([
        Conv1D(filters=32, kernel_size=4, activation='relu', input_shape=(8, 4)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear') # Output layer for the regression score
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

    # Train the model
    training_history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

    model.save('oracle_model.h5')
    return training_history
