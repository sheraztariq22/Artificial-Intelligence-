#Humna Amjad and Sheraz Tariq Group Assignment
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import re
import glob
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR') 

# Path where the CSV files are located
data_set = r'C:\Users\Hp EliteBook\Documents\Python_Assignments\Facial Landmark\archive'

# Function to load the data
def select_csv(data_set, emotions=('03', '04')):
    data = []
    labels = []
    max_shape = 0 

    # List all CSV files
    files = glob.glob(os.path.join(data_set, '*.csv'))

    # Regex pattern to match file names with desired modality and emotion codes
    pattern = re.compile(r'01-01-(' + '|'.join(emotions) + ')-')

    for file in files:
        match = pattern.search(os.path.basename(file))
        if match:
            emotion_code = match.group(1)
            df = pd.read_csv(file)

            # Skip files with no data or not matching expected columns
            if df.empty or not {'x_0', 'y_0'}.issubset(df.columns):
                print(f"Warning: No data found in file {file}.")
                continue

            # Flatten the dataframe and check if its shape matches the expected shape
            flat_data = df.values.flatten()
            if max_shape == 0:
                max_shape = flat_data.shape[0]  # Initialize max_shape with the first row shape
            elif flat_data.shape[0] != max_shape:
                print(f"Warning: Data in file {file} has inconsistent shape and will be skipped.")
                continue

            data.append(flat_data)
            labels.append(int(emotion_code == '03'))
    if not data:
        print("No valid data was loaded. Check the file names and contents.")
        return None, None

    return np.array(data), np.array(labels)

# Load the data
X, y = select_csv(data_set)
if X is not None and y is not None:
    y = to_categorical(y) 

    # Scale the input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network model for fitness evaluation
def create_model(input_dim):
    model = Sequential([
        Dense(120, input_dim=input_dim, activation='relu'),
        Dense(80, activation='relu'),
        Dense(80, activation='relu'),
        Dense(80, activation='relu'),
        Dense(80, activation='relu'),
        Dense(80, activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Fitness function to evaluate a single chromosome
def evaluate_chromosome(chromosome, X_train, y_train, X_val, y_val):
    mask = chromosome.astype(bool)
    X_train_selected = X_train[:, mask]
    X_val_selected = X_val[:, mask]
    
    # Create and train a neural network model
    model = create_model(X_train_selected.shape[1])
    model.fit(X_train_selected, y_train, epochs=10, verbose=0)  # Adjust epochs as needed
    
    # Evaluate the model on the validation set
    scores = model.evaluate(X_val_selected, y_val, verbose=0)
    return scores[1]  # Return the accuracy score

#initialize population
def initialize_population(pop_size, num_features):
    return np.random.randint(2, size=(pop_size, num_features))

#roulette wheel selection
def roulette_wheel_selection(population, fitness_scores):
    total_fitness = np.sum(fitness_scores)
    selection_probs = fitness_scores / total_fitness
    selected = np.random.choice(population.shape[0], size=population.shape[0], p=selection_probs)
    return population[selected]

#single point crossover
def single_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

#mutation
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def run_genetic_algorithm(X_train, y_train, X_val, y_val, pop_size, num_generations, mutation_rate):
    num_features = X_train.shape[1]
    population = initialize_population(pop_size, num_features)
    best_fitness = 0
    best_chromo = None

    print("Initial Generation:")
    for i in range(10): 
        fitness = evaluate_chromosome(population[i], X_train, y_train, X_val, y_val)
        print(f"Chromosome {i + 1:2d}: Accuracy = {fitness:.4f}")
        if fitness > best_fitness:
            best_fitness = fitness
            best_chromo = population[i]

    for generation in range(1, num_generations + 1):
        fitness_scores = np.array([evaluate_chromosome(chromo, X_train, y_train, X_val, y_val) for chromo in population])
        best_gen_idx = np.argmax(fitness_scores)
        if fitness_scores[best_gen_idx] > best_fitness:
            best_fitness = fitness_scores[best_gen_idx]
            best_chromo = population[best_gen_idx]

        selected = roulette_wheel_selection(population, fitness_scores)
        children = []
        
        for i in range(0, pop_size, 2):
            child1, child2 = single_point_crossover(selected[i], selected[i+1])
            children.append(mutate(child1, mutation_rate))
            children.append(mutate(child2, mutation_rate))
        
        population = np.array(children)
        
        if generation == 2 or generation == num_generations:
            print(f"\nGeneration {generation}:")
            for i in range(10):
                fitness = evaluate_chromosome(population[i], X_train, y_train, X_val, y_val)
                print(f"Chromosome {i + 1:2d}: Accuracy = {fitness:.4f}")

    print("\nBest Chromosome and Fitness:")
    print(f"Best Chromosome: {best_chromo}")
    print(f"Best Fitness: {best_fitness:.4f}")

    return best_chromo, best_fitness

# Settings for the genetic algorithm
pop_size = 100
num_generations = 2
mutation_rate = 0.01
best_chromo, best_fitness = run_genetic_algorithm(X_train, y_train, X_val, y_val, pop_size, num_generations, mutation_rate)

print(f"Best Chromosome: {best_chromo}")
print(f"Best Fitness: {best_fitness}")