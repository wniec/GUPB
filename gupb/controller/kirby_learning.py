import os.path
from collections import defaultdict
from itertools import chain, product
from queue import Queue
from typing import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from gupb import controller
from gupb.controller.neural_networks import ActorCriticNet, ActorLoss
from gupb.model import arenas, characters
from gupb.model.arenas import Arena

from gupb.model.characters import Facing, ChampionDescription
from gupb.model.coordinates import Coords
from gupb.model.effects import Mist
from gupb.model.tiles import TileDescription
from gupb.model.weapons import (
    Knife,
    Sword,
    Axe,
    Amulet,
    Scroll,
    WeaponDescription,
    Bow,
    Weapon,
)

POSSIBLE_ACTIONS = [
    characters.Action.TURN_LEFT,
    characters.Action.TURN_RIGHT,
    characters.Action.STEP_FORWARD,
    characters.Action.ATTACK,
    characters.Action.STEP_RIGHT,
    characters.Action.STEP_LEFT,
    characters.Action.STEP_BACKWARD,
    characters.Action.DO_NOTHING,
]
ROUNDS_NO = 3001
EPSILON = 0.0
LR_ARRAY: np.ndarray[float] = 5e-7 * (np.cumprod(
    np.full(shape=(ROUNDS_NO,), fill_value=0.99)
)+1e-4)
BOTS_NO = 5  # 12
MAP_PADDING = 2
POLICIES_NUM = 7
DIRECTIONS_NUM = 4

DISCOUNT_FACTOR_ARRAY = np.linspace(0.98, 0.98, ROUNDS_NO)
EPSILON_ARRAY = np.linspace(EPSILON, 0.00, ROUNDS_NO)
DISCOUNT_FACTOR = DISCOUNT_FACTOR_ARRAY[0]

weapons_dict = {
    Knife().description(): (0, 0, 0),
    Sword().description(): (0, 0, 1),
    WeaponDescription(name="bow_loaded"): (0, 1, 0),
    WeaponDescription(name="bow_unloaded"): (0, 1, 1),
    Axe().description(): (1, 0, 0),
    Amulet().description(): (1, 0, 1),
    Scroll().description(): (1, 1, 0),
}

weapons_hierarchy: dict[WeaponDescription, int] = {
    Knife().description(): -80,
    Sword().description(): 1,
    WeaponDescription(name="bow_loaded"): 3,
    WeaponDescription(name="bow_unloaded"): 3,
    Axe().description(): 3,
    Amulet().description(): 5,
    Scroll().description(): 6,
}

weapons_names_dict: dict[WeaponDescription, Weapon] = {
    Knife().description(): Knife(),
    Sword().description(): Sword(),
    WeaponDescription(name="bow_loaded"): Bow(),
    WeaponDescription(name="bow_unloaded"): Bow(),
    Axe().description(): Axe(),
    Amulet().description(): Amulet(),
    Scroll().description(): Scroll(),
}
directions_values_relative = {
    (Facing.UP, Facing.UP): (0, 0),
    (Facing.DOWN, Facing.UP): (1, 0),
    (Facing.LEFT, Facing.UP): (0, 1),
    (Facing.RIGHT, Facing.UP): (1, 1),
    (Facing.UP, Facing.DOWN): (1, 0),
    (Facing.DOWN, Facing.DOWN): (0, 0),
    (Facing.LEFT, Facing.DOWN): (1, 1),
    (Facing.RIGHT, Facing.DOWN): (0, 1),
    (Facing.UP, Facing.LEFT): (1, 1),
    (Facing.DOWN, Facing.LEFT): (0, 1),
    (Facing.LEFT, Facing.LEFT): (0, 0),
    (Facing.RIGHT, Facing.LEFT): (1, 0),
    (Facing.UP, Facing.RIGHT): (0, 1),
    (Facing.DOWN, Facing.RIGHT): (1, 1),
    (Facing.LEFT, Facing.RIGHT): (1, 0),
    (Facing.RIGHT, Facing.RIGHT): (0, 0),
}
directions_to_rotations = {
    Facing.UP: lambda x: x,
    Facing.DOWN: lambda x: torch.rot90(x, 2),
    Facing.LEFT: lambda x: torch.rot90(x, 3),
    Facing.RIGHT: lambda x: torch.rot90(x, 1),
}

dir_to_coords_change = {
    Facing.UP: lambda x, y: (-x, y),
    Facing.DOWN: lambda x, y: (x, -y),
    Facing.LEFT: lambda x, y: (-y, -x),
    Facing.RIGHT: lambda x, y: (y, x),
}
neighbourhood_coords_list = [
    (-2, 0),
    (-1, 0),
    (0, 0),
    (-1, -1),
    (-1, 1),
    (0, -2),
    (0, -1),
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, -1),
    (2, 0),
]


directions_to_indices = {Facing.UP: 0, Facing.LEFT: 1, Facing.DOWN: 2, Facing.RIGHT: 3}
indices_to_directions = {val: key for key, val in directions_to_indices.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"


def weapon_power(weapon_name):
    return getattr(weapons_names_dict[weapon_name].cut_effect(), "damage", 5)


def neighbourhood_4(position: tuple[int, int]):
    return [
        (position[0] + i, position[1] + j)
        for i, j in [(0, -1), (-1, 0), (0, 1), (1, 0)]
    ]  # UP, LEFT, DOWN, RIGHT


def distance_x_y(x: tuple[float, float]):
    return abs(x[0]) + abs(x[1])


class KirbyLearningController(controller.Controller):
    def __init__(self, first_name: str = "Kirby"):
        self.characters_no = None
        self.prev_attack_effects = None
        self.first_name: str = first_name
        self.map: torch.Tensor = torch.zeros((0,))
        self.transparent: torch.Tensor = torch.zeros((0,))
        self.terrain: dict = {}
        self.seen: torch.Tensor = torch.zeros((0,))
        self.menhir: tuple = (0, 0)
        self.prev_map = None
        self.prev_actions: list[int] = []
        self.mist: np.ndarray = np.zeros((0,))
        self.found_menhir: bool = False
        self.weapon = Knife().description()

        self.consumables: set = set()
        self.loot: dict[tuple[int, int], WeaponDescription] = {}
        self.effects: set = set()
        self.trees: list = []

        self.characters: dict[str, ChampionDescription] = {}
        self.positions_to_characters: dict = {}
        self.characters_to_positions: dict = {}

        self.model_A = ActorCriticNet(action_size=POLICIES_NUM).to(device)
        self.model_B = ActorCriticNet(action_size=POLICIES_NUM).to(device)
        self.model_B.load_state_dict(self.model_A.state_dict())

        self.actor_loss_fn = ActorLoss()
        self.critic_loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model_A.parameters(), lr=LR_ARRAY[0])

        self.time = 0

        self.losses = []
        self.game_losses = []
        self.scores = []
        self.times = []
        self.actions_count = []
        self.actions = np.zeros((POLICIES_NUM,))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KirbyLearningController):
            return self.first_name == other.first_name
        return False

    def __hash__(self) -> int:
        return hash(self.first_name)

    def exploration_status(self):
        return self.seen.sum() / self.map.numel()

    def update_visible_items(self, visible_tiles: dict, my_position: tuple):
        for coords, tile in visible_tiles.items():
            self.seen[coords] = 1

            if tile.type == "menhir":
                self.menhir = coords
                self.found_menhir = True
            elif coords == self.menhir and not self.found_menhir:
                self.random_menhir()

            if tile.type == "forest":
                self.trees.append(coords)

            if tile.consumable:
                self.consumables.add(coords)
            else:
                self.consumables.discard(coords)

            if tile.loot:
                self.loot[coords] = tile.loot
            elif coords in self.loot:
                self.loot.pop(coords)

            if tile.effects:
                self.effects.add(coords)
                if any(isinstance(effect, Mist) for effect in tile.effects):
                    self.mist[coords] = 1

            if tile.character is not None and coords != my_position:
                character_name = tile.character.controller_name
                self.characters[character_name] = tile.character
                old_pos = self.characters_to_positions.get(character_name, None)
                self.characters_to_positions[character_name] = coords
                if old_pos and old_pos != coords:
                    self.positions_to_characters.pop(old_pos, None)
                self.positions_to_characters[coords] = character_name
            elif tile.character is None and coords in self.positions_to_characters:
                character_name = self.positions_to_characters.pop(coords)
                self.characters_to_positions.pop(character_name)

        # self.mist = tuple((i / mist_tiles if mist_tiles else 0) for i in mist_position)

    def random_menhir(self):
        self.menhir = (
            np.random.randint(self.map.shape[0] - 2 * MAP_PADDING),
            np.random.randint(self.map.shape[1] - 2 * MAP_PADDING),
        )
        while (
            not self.map[self.menhir[0] + MAP_PADDING, self.menhir[1] + MAP_PADDING]
            or self.seen[self.menhir]
            or self.mist[self.menhir]
        ):
            self.menhir = (
                np.random.randint(self.map.shape[0] - 2 * MAP_PADDING),
                np.random.randint(self.map.shape[1] - 2 * MAP_PADDING),
            )

    def a_star_setup(self):
        return (
            np.full(
                (
                    self.map.shape[0] - 2 * MAP_PADDING,
                    self.map.shape[1] - 2 * MAP_PADDING,
                ),
                fill_value=float("inf"),
            ),
            Queue(),
        )

    def travel(
        self, my_position: tuple[int, int], my_direction: Facing
    ) -> characters.Action:
        distances, queue = self.a_star_setup()
        distances[self.menhir] = 0
        queue.put((self.menhir, 0))
        return self.path_finding(queue, distances, my_position, my_direction)

    def hide(
        self, my_position: tuple[int, int], my_direction: Facing
    ) -> characters.Action:
        distances, queue = self.a_star_setup()
        for tree in self.trees:
            distances[tree] = 0
            queue.put((tree, 0))
        return self.path_finding(queue, distances, my_position, my_direction)

    def bigger_weapons(
        self, my_position: tuple[int, int], my_direction: Facing
    ) -> characters.Action:
        distances, queue = self.a_star_setup()
        for coord in self.loot.keys():
            weapon_value = weapons_hierarchy[self.loot[coord]]
            if (
                weapon_value >= weapons_hierarchy[self.weapon]
                and self.weapon.name != "scroll"
            ):
                distances[coord] = -weapon_value * 10
                queue.put((coord, -weapon_value * 10))

        return self.path_finding(queue, distances, my_position, my_direction)

    def get_consumables(
        self, my_position: tuple[int, int], my_direction: Facing
    ) -> characters.Action:
        distances, queue = self.a_star_setup()
        for consumable in self.consumables:
            distances[consumable] = 0
            queue.put((consumable, 0))
        return self.path_finding(queue, distances, my_position, my_direction)

    def attack(
        self, my_position: tuple[int, int], my_direction: Facing
    ) -> characters.Action:
        my_weapon_hits = weapons_names_dict[self.weapon].cut_positions(self.terrain, my_position, my_direction)
        if any(i in my_weapon_hits for i in self.positions_to_characters.keys()):
            return characters.Action.ATTACK
        attacking_positions = []
        for character_position, attack_position in product(self.positions_to_characters.keys(), my_weapon_hits):
            goal_position = (character_position[0] - attack_position[0] + my_position[0],
                             character_position[1] - attack_position[1] + my_position[1])
            attacking_positions.append(goal_position)  # The position I need to be at in order to attack opponent

        distances, queue = self.a_star_setup()
        for coord in attacking_positions:
            distances[coord] = 0
            queue.put((coord, 0))
        return self.path_finding(queue, distances, my_position, my_direction)

    def reconnaissance(
        self,
        my_position: tuple[int, int],  # noqa
        my_direction: Facing,  # noqa
    ) -> characters.Action:
        return characters.Action.TURN_LEFT

    def run(self, my_position: tuple[int, int], my_direction: Facing):
        truncated_map = self.map[MAP_PADDING:-MAP_PADDING, MAP_PADDING:-MAP_PADDING]
        distances, queue = self.a_star_setup()
        hits_map = self.opponents_hit_dict()
        for position in chain(
            self.positions_to_characters, hits_map.keys(), self.effects
        ):
            is_in = (
                0 < position[0] < truncated_map.shape[0]
                and 0 < position[1] < truncated_map.shape[1]
            )
            if is_in:
                distances[position] = 0
                queue.put((position, 0))

        distances[self.menhir] = 20_000
        queue.put((self.menhir, 20_000))
        while not queue.empty():
            tile, distance = queue.get()
            for dir_vector in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                next_tile = tile[0] + dir_vector[0], tile[1] + dir_vector[1]
                is_in = (
                    0 < next_tile[0] < truncated_map.shape[0]
                    and 0 < next_tile[1] < truncated_map.shape[1]
                )
                if (
                    is_in
                    and not self.mist[next_tile]
                    and truncated_map[next_tile]
                    and distances[next_tile] > distance + 1
                ):
                    distances[next_tile] = distance + 1
                    queue.put((next_tile, distance + 1))
        next_action_id = np.nan_to_num(
            np.array(
                [
                    distances[i, j]
                    if 0 < i < truncated_map.shape[0] and 0 < j < truncated_map.shape[1]
                    else 0
                    for i, j in neighbourhood_4(my_position)
                ]
            ),
            posinf=0,
        ).argmax()

        if (
            distances[my_position]
            > distances[neighbourhood_4(my_position)[next_action_id]]
        ):
            return characters.Action.TURN_RIGHT
        else:
            direction_diff = next_action_id - directions_to_indices[my_direction]
            match (direction_diff + DIRECTIONS_NUM) % DIRECTIONS_NUM:
                case 0:
                    return characters.Action.STEP_FORWARD
                case 1:
                    return characters.Action.STEP_LEFT
                case 2:
                    return characters.Action.STEP_BACKWARD
                case 3:
                    return characters.Action.STEP_RIGHT

    def opponents_hit_dict(self):
        opponents_hits = [
            (coord, weapon_power(character.weapon))
            for character in self.characters.values()
            if character.controller_name in self.characters_to_positions
            for coord in weapons_names_dict[character.weapon].cut_positions(
                self.terrain,
                Coords(*self.characters_to_positions[character.controller_name]),
                character.facing,
            )
        ]
        opponents_hits_dict = defaultdict(lambda: 0)
        for coord, damage in opponents_hits:
            opponents_hits_dict[coord] += damage
        return opponents_hits_dict

    def path_finding(
        self,
        queue: Queue,
        distances: np.ndarray,
        my_position: tuple[int, int],
        my_direction: Facing,
        additional_impassable=None,
    ):
        truncated_map = self.map[MAP_PADDING:-MAP_PADDING, MAP_PADDING:-MAP_PADDING]
        hits_map = self.opponents_hit_dict()
        while not queue.empty():
            tile, distance = queue.get()
            for dir_vector in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                next_tile = tile[0] + dir_vector[0], tile[1] + dir_vector[1]
                is_in = (
                    0 < next_tile[0] < truncated_map.shape[0]
                    and 0 < next_tile[1] < truncated_map.shape[1]
                )
                next_tile_effect = 1000 if next_tile in self.effects else 1
                next_tile_effect += hits_map[next_tile] * 100
                if (
                    is_in
                    and not self.mist[next_tile]
                    and truncated_map[next_tile]
                    and next_tile not in self.positions_to_characters
                    and distances[next_tile] > distance + next_tile_effect
                    and (
                        additional_impassable is None
                        or next_tile not in additional_impassable
                    )
                ):
                    distances[next_tile] = distance + next_tile_effect
                    queue.put((next_tile, distance + next_tile_effect))
        next_action_id = np.array(
            [
                distances[i, j]
                if 0 < i < truncated_map.shape[0] and 0 < j < truncated_map.shape[1]
                else float("inf")
                for i, j in neighbourhood_4(my_position)
            ]
        ).argmin()
        if distances[my_position] == 0:
            return characters.Action.TURN_LEFT
        else:
            direction_diff = next_action_id - directions_to_indices[my_direction]
            match (direction_diff + DIRECTIONS_NUM) % DIRECTIONS_NUM:
                case 0:
                    return characters.Action.STEP_FORWARD
                case 1:
                    return characters.Action.STEP_LEFT
                case 2:
                    return characters.Action.STEP_BACKWARD
                case 3:
                    return characters.Action.STEP_RIGHT

    def get_neighbourhood_from(
        self, my_position: tuple[int, int], my_direction: Facing, my_map: torch.Tensor
    ) -> torch.Tensor:
        neighbourhood = my_map[
            my_position[0]: my_position[0] + 2 * MAP_PADDING + 1,
            my_position[1]: my_position[1] + 2 * MAP_PADDING + 1,
        ]

        neighbourhood = directions_to_rotations[my_direction](neighbourhood)
        coords_list = [(i + 2, j + 2) for i, j in neighbourhood_coords_list]
        return torch.tensor([neighbourhood[coords] for coords in coords_list])

    def get_neighbourhood(
        self, my_position: tuple[int, int], my_direction: Facing
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ##2##
        #212#
        21012
        #212#
        ##2##
        Używam takiego sąsiedztwa
        """
        neighbourhood = self.map[
            my_position[0]: my_position[0] + 2 * MAP_PADDING + 1,
            my_position[1]: my_position[1] + 2 * MAP_PADDING + 1,
        ]

        neighbourhood = directions_to_rotations[my_direction](neighbourhood)
        f1 = neighbourhood[1, 2]
        f2 = f1 and neighbourhood[0, 2]
        r1 = neighbourhood[2, 3]
        r2 = r1 and neighbourhood[2, 4]
        rf = neighbourhood[1, 3] and (f1 or r1)
        l1 = neighbourhood[2, 1]
        l2 = l1 and neighbourhood[2, 0]
        lf = neighbourhood[1, 1] and (f1 or l1)
        b1 = neighbourhood[3, 2]
        b2 = neighbourhood[4, 2] and b1
        lb = neighbourhood[3, 1] and (b1 or l1)
        rb = neighbourhood[3, 3] and (b1 or r1)
        neighbourhood = torch.tensor([f1, f2, l1, l2, lf, r1, r2, rf, b1, b2, lb, rb])
        return neighbourhood, f1

    def opponents_hits_vector(
        self, opponents_hits: list[tuple[tuple[int, int], int]]
    ) -> list[int]:
        hit_effects = defaultdict(lambda: 0)
        for tile in opponents_hits:
            relative_coord = tile[0]
            if relative_coord in neighbourhood_coords_list:
                hit_effects[relative_coord] += tile[1]

        return [hit_effects[i] for i in neighbourhood_coords_list]

    def analyse_knoledge(self, knowledge: characters.ChampionKnowledge):

        relative_coords = lambda x: dir_to_coords_change[my_direction](
            x[0] - knowledge.position.x, knowledge.position.y - x[1]
        )
        scaled_coords = lambda x: (
            x[0] / (self.map.shape[0] - 2 * MAP_PADDING),
            x[1] / (self.map.shape[1] - 2 * MAP_PADDING),
        )
        if self.time == 0:
            self.characters_no = knowledge.no_of_champions_alive
        my_position = knowledge.position
        my_tile: TileDescription = knowledge.visible_tiles[knowledge.position]
        my_effects = 1 if my_tile.effects else 0
        my_health = my_tile.character.health / 16

        my_direction = my_tile.character.facing

        self.weapon = my_tile.character.weapon
        my_weapon_hits = {
            relative_coords(i)
            for i in weapons_names_dict[self.weapon].cut_positions(
                self.terrain, my_position, my_direction
            )
        }
        my_weapon_power = weapon_power(self.weapon)
        my_vector = torch.tensor([my_health, my_effects])

        self.update_visible_items(knowledge.visible_tiles, my_position)
        closest_consumables = torch.zeros((20,))
        nonzero_consumables = torch.tensor(
            sorted(
                [scaled_coords(relative_coords(i)) for i in self.consumables],
                key=distance_x_y,
            )[:10]
        ).reshape(-1)
        closest_consumables[: len(nonzero_consumables)] = nonzero_consumables

        closest_loot = torch.zeros((20,))
        nonzero_loot = torch.tensor(
            sorted(
                [scaled_coords(relative_coords(i)) for i in self.loot.keys()],
                key=distance_x_y,
            )[:10]
        ).reshape(-1)
        closest_loot[: len(nonzero_loot)] = nonzero_loot

        closest_effects = torch.zeros((20,))
        effects_relative_coords = sorted(
            [relative_coords(i) for i in self.effects], key=distance_x_y
        )[:10]
        effects_relative_scaled_coords = [
            scaled_coords(j) for j in effects_relative_coords
        ]
        nonzero_effects = torch.tensor(effects_relative_scaled_coords).reshape(-1)
        closest_effects[: len(nonzero_effects)] = nonzero_effects

        effect_in_front = 1 if (-1, 0) in effects_relative_coords else 0

        characters_seen = []
        for i, (character_name, coords) in enumerate(
            self.characters_to_positions.items()
        ):
            character = self.characters[character_name]
            direction = directions_values_relative[(character.facing, my_direction)]
            characters_seen.append(
                [
                    *direction,  # 2
                    character.health / 8,  # 1
                    *relative_coords(coords),  # 2
                ]
            )
        characters_seen.sort(key=lambda x: distance_x_y(x[-3:]))
        attack_effects = (
            sum(
                [
                    my_weapon_power
                    for i in self.positions_to_characters.keys()
                    if relative_coords(i) in my_weapon_hits
                ]
            )
            / 8
        )
        opponents_hits: list[tuple[tuple[int, int], int]] = [
            (relative_coords(i), weapon_power(character.weapon))
            for character in self.characters.values()
            if character.controller_name in self.characters_to_positions
            for i in weapons_names_dict[character.weapon].cut_positions(
                self.terrain,
                Coords(*self.characters_to_positions[character.controller_name]),
                character.facing,
            )
        ]

        hits_vector = torch.tensor(
            self.opponents_hits_vector(opponents_hits), dtype=torch.float32
        )

        characters_vector = torch.zeros((3 * 5,))
        characters_seen = torch.tensor(
            [scaled_coords(i) for i in characters_seen[:3]]
        ).reshape(-1)
        characters_vector[: len(characters_seen)] = characters_seen

        neighbourhood, can_go_forward = self.get_neighbourhood(
            my_position, my_direction
        )

        transparent = self.get_neighbourhood_from(
            my_position, my_direction, self.transparent
        )
        #prev_actions = [[int(i) for i in f'{i:03b}'] for i in self.prev_actions[-5:]]
        seen = self.get_neighbourhood_from(my_position, my_direction, self.seen)
        menhir_coords = scaled_coords(relative_coords(self.menhir))
        meta = torch.tensor(
            [
                knowledge.no_of_champions_alive / self.characters_no,
                *menhir_coords,
                self.exploration_status(),
                self.time / 1000,
                effect_in_front,
                attack_effects,
                self.map.shape[0] / 100,
                self.map.shape[1] / 100,
                # *scaled_coords(relative_coords(self.mist)),
            ]
        )
        result_vector = torch.hstack(
            [
                my_vector,  # 2
                closest_consumables,  # 20
                closest_loot,  # 20
                closest_effects,  # 20
                characters_vector,  # 15
                neighbourhood,  # 12
                transparent,  # 12
                meta,  # 9
                hits_vector,  # 13
                seen,  # 12
                # prev_actions,  # 15
            ]
        )
        return result_vector.reshape(1, -1).type(torch.float32), attack_effects

    def learn(self, current_reward, expected_reward, policy_log):
        actor_loss = self.actor_loss_fn(
            current_reward, expected_reward.detach(), policy_log
        )
        critic_loss = self.critic_loss_fn(
            torch.tensor(current_reward).reshape(1, 1).to(device), expected_reward
        )

        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model_A.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.losses.append(critic_loss.item())
        tau = 0.1
        for target_param, param in zip(
            self.model_B.parameters(), self.model_A.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def decide(self, knowledge: characters.ChampionKnowledge) -> characters.Action:
        my_position = tuple(knowledge.position)
        my_tile: TileDescription = knowledge.visible_tiles[knowledge.position]
        my_direction: Facing = my_tile.character.facing
        policies: list[Callable[[tuple[int, int], Facing], characters.Action]] = [
            self.travel,
            self.run,
            self.hide,
            self.attack,
            self.bigger_weapons,
            self.get_consumables,
            self.reconnaissance,
        ]

        new_map, attack_effects = self.analyse_knoledge(knowledge)
        my_tile = knowledge.visible_tiles[knowledge.position]
        my_health = my_tile.character.health

        with torch.no_grad():
            policy_b, expected_value_b = self.model_B(
                new_map.to(device)
            )  # przewidujemy przyszłość

        if self.prev_map is not None:
            policy_a, expected_value_a = self.model_A(
                self.prev_map.to(device)
            )  # przewidujemy teraźniejszość na podstawie przeszłości
            prev_policy_log = torch.log(policy_a[0, self.prev_actions[-1]])

            reward = (
                DISCOUNT_FACTOR * expected_value_b[0, 0].detach()
                + min(8, my_health) / 20
                + self.exploration_status() / 10
                + self.prev_attack_effects
                - knowledge.no_of_champions_alive / self.characters_no / 10
            )
            self.learn(reward, expected_value_a, prev_policy_log)

        epsilon_greedy_probs = (
            np.ones((POLICIES_NUM,)) / POLICIES_NUM * EPSILON
            + (1 - EPSILON) * policy_b.cpu().detach().numpy()[0]
        )
        epsilon_greedy_probs /= epsilon_greedy_probs.sum()
        choice_idx = np.random.choice(
            [i for i in range(POLICIES_NUM)],
            p=epsilon_greedy_probs if self.prev_map is not None else None,
        )

        self.time += 1
        self.prev_map = new_map.clone()
        self.prev_attack_effects = 0 if choice_idx != 3 else attack_effects / 10
        self.actions[choice_idx] += 1
        self.prev_actions.append(choice_idx)

        return policies[choice_idx](my_position, my_direction)

    def praise(self, score: int) -> None:
        self.scores.append(score)
        self.times.append(self.time)
        if score < self.characters_no:
            policy_a, expected_value_a = self.model_A(self.prev_map.to(device))
            prev_policy_log = torch.log(policy_a[0, self.prev_actions[-1]])
            self.learn(
                min(8, 0) / 20
                + self.exploration_status() / 10
                + self.prev_attack_effects,
                expected_value_a,
                prev_policy_log,
            )

    def log_progress(self, game_no):
        rewards = [i / BOTS_NO for i in self.scores]
        last_50_cumsum = [
            sum(rewards[max(0, i - 50): i + 1]) / min(i + 1, 50)
            for i in range(20, len(rewards))
        ]
        last_50_times = [
            sum(self.times[max(0, i - 50): i + 1]) / min(i + 1, 50)
            for i in range(20, len(self.times))
        ]
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        ax[0, 0].plot(
            [i for i, _ in enumerate(self.game_losses)],
            [np.log(i) for i in self.game_losses],
        )
        ax[0, 1].plot(
            [i for i in range(20, len(rewards))],
            last_50_cumsum,
            color="orange",
        )

        ax[1, 1].plot(
            [i for i in range(20, len(self.times))],
            last_50_times,
            color="green",
        )
        for i, color in enumerate(("red", "green", "blue", "pink", "cyan", "purple", "yellow")):
            upper = [episode[i] for episode in self.actions_count]
            lower = (
                [episode[i - 1] for episode in self.actions_count]
                if i > 0
                else np.zeros((len(self.actions_count),))
            )
            ax[1, 0].fill_between(
                [i for i in range(len(self.actions_count))],
                lower,
                upper,
                color=color,
            )
        plt.show()
        plt.savefig(os.path.join("plots", f"all_rounds_{game_no}.png"))

    def reset(self, game_no: int, arena_description: arenas.ArenaDescription) -> None:
        global DISCOUNT_FACTOR, EPSILON
        DISCOUNT_FACTOR = DISCOUNT_FACTOR_ARRAY[game_no]
        EPSILON = EPSILON_ARRAY[game_no]
        if game_no == 0:
            if os.path.exists("best_weights.pth"):
                checkpoint = torch.load("best_weights.pth", weights_only=False)
                self.model_A.load_state_dict(checkpoint["model"])
                self.model_B.load_state_dict(self.model_A.state_dict())
                self.optimizer.load_state_dict(checkpoint["optimizer"])
        for g in self.optimizer.param_groups:
            g["lr"] = LR_ARRAY[game_no]
        if game_no > 0:
            self.game_losses.append(sum(self.losses) / len(self.losses))
            self.actions_count.append(np.cumsum(self.actions / self.actions.sum()))

        arena = Arena.load(arena_description.name)
        self.terrain = arena.terrain
        checkpoint1_freq = 50
        if game_no % checkpoint1_freq == 0 and game_no:
            self.log_progress(game_no)

            checkpoint = {
                "model": self.model_A.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join("weights", f"weights{game_no}.pth"))

        """checkpoint2_freq = 10
        if game_no % checkpoint2_freq == 0 and game_no:
            checkpoint = {
                "model": self.model_A.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, "learned_weights.pth")"""

        self.map = torch.zeros(arena.size)
        self.transparent = torch.zeros(arena.size)

        self.time = 0

        self.consumables = set()
        self.loot = {}
        self.effects = set()
        self.trees = []

        self.characters = {}
        self.characters_to_positions = {}
        self.positions_to_characters = {}

        for coords, tile in self.terrain.items():
            self.map[coords] += int(tile.passable)
            self.transparent[coords] += int(tile.transparent)

        self.mist = np.zeros_like(self.map)
        self.map = torch.tensor(
            np.pad(
                self.map,
                ((MAP_PADDING, MAP_PADDING), (MAP_PADDING, MAP_PADDING)),
                "constant",
                constant_values=(0, 0),
            )
        )

        self.transparent = torch.tensor(
            np.pad(
                self.transparent,
                ((MAP_PADDING, MAP_PADDING), (MAP_PADDING, MAP_PADDING)),
                "constant",
                constant_values=(0, 0),
            )
        )
        self.seen = torch.zeros_like(self.map)
        self.random_menhir()
        self.prev_map = None
        self.found_menhir = False
        self.prev_actions = []
        self.actions = np.zeros((POLICIES_NUM,))
        self.weapon = Knife().description()

    @property
    def name(self) -> str:
        return f"{self.first_name}"

    @property
    def preferred_tabard(self) -> characters.Tabard:
        return characters.Tabard.KIRBY


POTENTIAL_CONTROLLERS = [
    KirbyLearningController("Kirby"),
]
