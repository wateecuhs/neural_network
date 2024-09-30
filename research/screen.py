import pygame
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# Load and preprocess MNIST data
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Build and train the model
model = keras.models.Sequential([
	keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
	keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Dropout(0.25),
	keras.layers.Flatten(),
	keras.layers.Dense(256, activation='relu'),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adadelta',
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(x_train, y_train, epochs=5, validation_split=0.1, batch_size=128, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 28
PIXEL_SIZE = 20
WINDOW_SIZE = GRID_SIZE * PIXEL_SIZE

# Set up the display
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
pygame.display.set_caption("MNIST Digit Recognizer")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Create a 28x28 grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

def draw_grid():
	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			color = tuple(int(grid[y][x] * 255) for _ in range(3))
			pygame.draw.rect(screen, color,
							 (x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

def draw_ui():
	pygame.draw.rect(screen, GRAY, (0, WINDOW_SIZE, WINDOW_SIZE, 50))
	font = pygame.font.SysFont(None, 36)
	text = font.render(f"Prediction: {prediction}", True, BLACK)
	screen.blit(text, (10, WINDOW_SIZE + 10))

# Main game loop
clock = pygame.time.Clock()
running = True
drawing = False
prediction = None

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		elif event.type == pygame.MOUSEBUTTONDOWN:
			if event.button == 1:  # Left mouse button
				drawing = True
		elif event.type == pygame.MOUSEBUTTONUP:
			if event.button == 1:  # Left mouse button
				drawing = False
		elif event.type == pygame.MOUSEMOTION:
			if drawing:
				x, y = event.pos
				grid_x, grid_y = x // PIXEL_SIZE, y // PIXEL_SIZE
				if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
					for dy in range(0, 1):
						for dx in range(0, 1):
							ny, nx = grid_y + dy, grid_x + dx
							if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
								distance = np.sqrt(dx**2 + dy**2)
								grid[ny][nx] = min(grid[ny][nx] + max(0, 1 - distance/2), 1)
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_SPACE:
				# Make prediction
				image = grid.reshape(1, 28, 28)
				pred = model.predict(image)
				prediction = np.argmax(pred[0])
			elif event.key == pygame.K_c:
				# Clear the grid
				grid.fill(0)
				prediction = None

	screen.fill(BLACK)
	draw_grid()
	draw_ui()
	pygame.display.flip()
	clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()