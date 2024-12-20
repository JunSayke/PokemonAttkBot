from enum import Enum

import pandas as pd


# https://www.delugerpg.com/pages/type-chart/
# Define the column names
columns = ["Type 1", "Type 2", "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]

# Read the CSV file and skip the first row
df = pd.read_csv('pokemon_type_chart.csv', names=columns, skiprows=1)

# Define a dictionary to map fractional strings to their float values
fraction_to_float = {
    '½': 0.5,
    '¼': 0.25,
}

# Replace fractional strings with their float values
df.replace(fraction_to_float, inplace=True)

# Convert all values to numeric, errors='coerce' will convert non-numeric values to NaN
df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

# TODO: Predict the best move to use against a given Pokemon type without using machine learning

class PokemonType(Enum):
    NORMAL = "Normal"
    FIRE = "Fire"
    WATER = "Water"
    ELECTRIC = "Electric"
    GRASS = "Grass"
    ICE = "Ice"
    FIGHTING = "Fighting"
    POISON = "Poison"
    GROUND = "Ground"
    FLYING = "Flying"
    PSYCHIC = "Psychic"
    BUG = "Bug"
    ROCK = "Rock"
    GHOST = "Ghost"
    DRAGON = "Dragon"
    DARK = "Dark"
    STEEL = "Steel"
    FAIRY = "Fairy"

class PokemonClass(Enum):
    NORMAL = "Normal"
    SHINY = "Shiny"
    DARK = "Dark"
    GHOSTLY = "Ghostly"
    METALLIC = "Metallic"
    SHADOW = "Shadow"
    MIRAGE = "Mirage"
    CHROME = "Chrome"
    NEGATIVE = "Negative"
    RETRO = "Retro"

def get_effectiveness(primary_type: PokemonType, secondary_type: PokemonType | None, invert=False):
    if secondary_type:
        row = df[(df['Type 1'] == primary_type.value) & (df['Type 2'] == secondary_type.value)]
    else:
        row = df[(df['Type 1'] == primary_type.value) & (df['Type 2'].isnull())]

    if row.empty:
        return None

    effectiveness = row.iloc[0].copy()

    if invert:
        print("Inverting effectiveness")
        inversion_map = {0: 2, 2: 0.5, 0.5: 2, 0.25: 4}
        effectiveness = effectiveness.apply(lambda x: inversion_map.get(x, x))

    return effectiveness


def best_move(def_primary_type: PokemonType, def_secondary_type: PokemonType | None, attk_moves, attk_class: PokemonClass = PokemonClass.NORMAL, def_class: PokemonClass = PokemonClass.NORMAL):
    invert = False

    if (attk_class == PokemonClass.NEGATIVE and def_class != PokemonClass.NEGATIVE) or (attk_class != PokemonClass.NEGATIVE and def_class == PokemonClass.NEGATIVE):
        invert = True

    row = get_effectiveness(def_primary_type, def_secondary_type, invert)
    if row is None:
        return None, 0

    effectiveness = {move: row[move.value] for move in attk_moves if move.value in row.index}
    effective_move = max(effectiveness, key=effectiveness.get)

    return effective_move, effectiveness[effective_move]


def calculate_winning_chance(def_primary_type: PokemonType, def_secondary_type: PokemonType | None, attk_class: PokemonClass,
                             def_class: PokemonClass, attk_moves):
    # Get the effectiveness of each available move
    invert = False

    if (attk_class == PokemonClass.NEGATIVE and def_class != PokemonClass.NEGATIVE) or (attk_class != PokemonClass.NEGATIVE and def_class == PokemonClass.NEGATIVE):
        invert = True

    row = get_effectiveness(def_primary_type, def_secondary_type, invert)
    if row is None:
        return 0

    effectiveness = {move: row[move.value] for move in attk_moves if move.value in row.index}

    if not effectiveness:
        return 0

    best_move_effectiveness = max(effectiveness.values())

    # Base winning chance
    winning_chance = best_move_effectiveness

    # https://www.delugerpg.com/pages/pokemon-class
    # Modify winning chance based on class abilities
    if attk_class == PokemonClass.SHINY:
        winning_chance *= 1.25
    elif attk_class == PokemonClass.DARK:
        winning_chance *= 1.25
    elif attk_class == PokemonClass.CHROME:
        winning_chance *= 1.35
    elif attk_class == PokemonClass.GHOSTLY:
        # Assuming flinching reduces the opponent's effectiveness
        winning_chance *= 1.1
    elif attk_class == PokemonClass.METALLIC:
        # Assuming immunity to status ailments increases winning chance
        winning_chance *= 1.1
    elif attk_class == PokemonClass.SHADOW:
        if def_class in [PokemonClass.METALLIC, PokemonClass.GHOSTLY]:
            winning_chance *= 1.2
    elif attk_class == PokemonClass.MIRAGE:
        if def_class == PokemonClass.DARK:
            winning_chance *= 1.2
        elif def_class == PokemonClass.CHROME:
            winning_chance *= 1.35
        elif def_class == PokemonClass.SHINY:
            winning_chance *= 1.2
        elif def_class in [PokemonClass.GHOSTLY, PokemonClass.METALLIC]:
            winning_chance *= 1.1
        elif def_class == PokemonClass.SHADOW:
            winning_chance *= 1.1
        elif def_class == PokemonClass.NORMAL:
            winning_chance *= 0.9

    if def_class == PokemonClass.SHINY:
        winning_chance /= 1.25
    elif def_class == PokemonClass.CHROME:
        winning_chance /= 1.35

    return winning_chance


def calculate_normalized_winning_chances(ally_primary_type: PokemonType, ally_secondary_type: PokemonType | None,
                                         ally_class: PokemonClass,
                                         enemy_primary_type: PokemonType, enemy_secondary_type: PokemonType | None,
                                         enemy_class: PokemonClass,
                                         ally_moves, enemy_moves):
    # Calculate ally winning chance
    ally_winning_chance = calculate_winning_chance(enemy_primary_type, enemy_secondary_type, ally_class, enemy_class,
                                                   ally_moves)

    # Calculate enemy winning chance
    enemy_winning_chance = calculate_winning_chance(ally_primary_type, ally_secondary_type, enemy_class, ally_class,
                                                    enemy_moves)

    # Normalize the chances
    total_chance = ally_winning_chance + enemy_winning_chance
    if total_chance == 0:
        return 0, 0

    normalized_ally_chance = ally_winning_chance / total_chance
    normalized_enemy_chance = enemy_winning_chance / total_chance

    return normalized_ally_chance, normalized_enemy_chance

# Example usage
ally_type1 = PokemonType.FIRE  # Ally Pokemon Primary Type
ally_type2 = None  # Ally Pokemon Secondary Type
ally_moves = [PokemonType.FIRE, PokemonType.FIRE, PokemonType.FIRE, PokemonType.FIRE]
ally_class = PokemonClass.NORMAL  # Ally Pokemon class

enemy_type1 = PokemonType.WATER  # Enemy Pokemon Primary Type
enemy_type2 = None  # Enemy Pokemon Secondary Type
enemy_moves = [PokemonType.WATER, PokemonType.WATER, PokemonType.WATER, PokemonType.WATER]
enemy_class = PokemonClass.NORMAL  # Enemy Pokemon class

normalized_ally_chance, normalized_enemy_chance = calculate_normalized_winning_chances(
    ally_type1, ally_type2, ally_class, enemy_type1, enemy_type2, enemy_class, ally_moves, enemy_moves
)

print(f"Ally Winning Chance: {normalized_ally_chance}")
print(f"Enemy Winning Chance: {normalized_enemy_chance}")

ally_best_move, ally_move_effectiveness = best_move(enemy_type1, enemy_type2, ally_moves, ally_class, enemy_class)
print(f"Ally Best Move: {ally_best_move.value} with effectiveness {ally_move_effectiveness}")
enemy_best_move, enemy_move_effectiveness = best_move(ally_type1, ally_type2, enemy_moves, enemy_class, ally_class)
print(f"Enemy Best Move: {enemy_best_move.value} with effectiveness {enemy_move_effectiveness}")

# TODO: This time using machine learning
# One-hot encode the categorical columns
df_encoded = pd.get_dummies(df, columns=["Type 1", "Type 2"])
# print(df_encoded.head())


