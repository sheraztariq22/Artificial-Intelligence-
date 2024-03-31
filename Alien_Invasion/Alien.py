import heapq
import random

class City:
    def __init__(self, name, military_base=False, weapons=0, civilians=0, aliens=0):
        self.name = name
        self.military_base = military_base
        self.weapons = weapons
        self.civilians = civilians
        self.aliens = aliens
        self.neighbors = {}

    def add_neighbor(self, city, distance):
        self.neighbors[city] = distance

    def is_defeated(self):
        return self.aliens > self.weapons

    def __str__(self):
    return "City: {}, Population: {}, Defense Material: {}, Alien Population: {}".format(self.name, self.civilians, self.weapons, self.aliens)


def create_graph():
    cities = {}
    for i in range(10):
        city = City(f'City {i}')
        cities[city.name] = city
        city.military_base = random.choice([True, False])
        city.weapons = random.randint(0, 100)
        city.civilians = random.randint(0, 100)

    for city_name, city in cities.items():
        for neighbor_name, neighbor in cities.items():
            if city_name != neighbor_name and random.random() > 0.5:
                distance = random.randint(1, 10)
                city.add_neighbor(neighbor, distance)
                neighbor.add_neighbor(city, distance)

    spawn_cities = random.sample(list(cities.values()), k=3)
    for spawn_city in spawn_cities:
        assign_aliens(spawn_city, cities["City 9"])

    return cities

def assign_aliens(start, end):
    visited = set()
    queue = [(start, [start])]

    while queue:
        current, path = queue.pop(0)
        visited.add(current)

        if current == end:
            for city in path:
                city.aliens = random.randint(0, 100)
            return

        for neighbor in current.neighbors:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

def bfs_search(start, end):
    visited = set()
    queue = [(start, [start])]

    while queue:
        current, path = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current == end:
            print("Path found!")
            for city in path:
                print(city)
            return

        for neighbor in current.neighbors:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    print("No path found")

def save_city(city):
    if city.is_defeated():
        print(f"{city.name} has been defeated by aliens.")
    else:
        print(f"{city.name} has been saved from an alien attack.")

if __name__ == "__main__":
    cities = create_graph()
    for city_name, city in cities.items():
        print(city)
        print("Neighbors:", ", ".join([f"{neighbor.name} (distance: {distance})" for neighbor, distance in city.neighbors.items()]))
    start = cities["City 0"]
    end = cities["City 9"]
    bfs_search(start, end)
    save_city(cities["City 9"])
