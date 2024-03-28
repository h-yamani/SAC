from typing import Dict, List
import numpy as np

from pyboy_environment.environments.pokemon import pokemon_constants as pkc
from pyboy_environment.environments.environment import PyboyEnvironment

from pyboy import WindowEvent


class PokemonEnvironment(PyboyEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:
        super().__init__(
            task="pokemon",
            rom_name="PokemonRed.gb",
            init_name="has_pokedex.state",
            act_freq=act_freq,
            emulation_speed=emulation_speed,
            headless=headless,
        )

        self.valid_actions: List[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        self.release_button: List[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        """
        TODO: Clean up this entire environment
        Currently not in a working state I believe
        """

        self.seen_locations = set()
        self.stuck_count = 0
        self.outside_count = 0
        self.inside_count = 0
        self.inside_map_constants = []  # These don't do anything
        self.outside_map_constants = []  # THese don't do anything
        self.initial_positions = {}
        self.last_distance_travelled = 0
        self.buffer_frame = 0

    def _stats_to_state(self, game_stats: Dict[str, any]) -> List[any]:
        state = [
            game_stats["location"]["x"],
            game_stats["location"]["y"],
            game_stats["location"]["map_id"],
            # game_stats["party_size"],
            # game_stats["ids"],
            # game_stats["levels"],
            # game_stats["type_id"],
            # game_stats["xp"],
            # game_stats["status"],
            # game_stats["badges"],
            # game_stats["caught_pokemon"],
            # game_stats["seen_pokemon"],
            # game_stats["money"],
            # game_stats["hp"]["current"],
            # game_stats["hp"]["max"],
            # game_stats["events"],
            game_stats["stucked"],
            game_stats["outin"][0],
            game_stats["outin"][1],
        ]

        # Generate the non-walkable matrix for the current location
        obstacle_matrix = self._get_screen_walkable_matrix()

        # Flatten the obstacle matrix and add it to the state
        flattened_obstacle_matrix = obstacle_matrix.flatten().tolist()
        state.extend(flattened_obstacle_matrix)

        return np.array(state, dtype=np.float32)

    def _generate_game_stats(self) -> Dict[str, any]:
        return {
            "location": self._get_location(),
            "party_size": self._get_party_size(),
            "ids": self._read_party_id(),
            "pokemon": [pkc.get_pokemon(id) for id in self._read_party_id()],
            "levels": self._read_party_level(),
            "type_id": self._read_party_type(),
            "type": [pkc.get_type(id) for id in self._read_party_type()],
            "hp": self._read_party_hp(),
            "xp": self._read_party_xp(),
            "status": self._read_party_status(),
            "badges": self._get_badge_count(),
            "caught_pokemon": self._read_caught_pokemon_count(),
            "seen_pokemon": self._read_seen_pokemon_count(),
            "money": self._read_money(),
            "events": self._read_events(),
            "outin": self.get_outside_inside(self._get_location()),
            "stucked": self.get_if_stucked(self._get_location()),
        }

    def _reward_stats_to_reward(self, reward_stats: Dict[str, any]) -> int:
        reward_total = 0
        for _, reward in reward_stats.items():
            reward_total += reward
        return reward_total

    def _calculate_reward_stats(self, new_state: Dict[str, any]) -> Dict[str, int]:
        return {
            "caught_reward": self._caught_reward(new_state),
            "seen_reward": self._seen_reward(new_state),
            "health_reward": self._health_reward(new_state),
            "xp_reward": self._xp_reward(new_state),
            "levels_reward": self._levels_reward(new_state),
            "badges_reward": self._badges_reward(new_state),
            "money_reward": self._money_reward(new_state),
            "event_reward": self._event_reward(new_state),
            "stuck_reward": self._stuck_reward(new_state),
            "location_reward": self._location_reward(new_state),
            "distance_reward": self._distance_travelled_reward(new_state),
            "grass_reward": self._grass_reward(new_state),
            "outside_reward": self._outside_reward(new_state),
        }

    def _caught_reward(self, new_state: Dict[str, any]) -> int:
        return new_state["caught_pokemon"] - self.prior_game_stats["caught_pokemon"]

    def _seen_reward(self, new_state: Dict[str, any]) -> int:
        return new_state["seen_pokemon"] - self.prior_game_stats["seen_pokemon"]

    def _health_reward(self, new_state: Dict[str, any]) -> int:
        return sum(new_state["hp"]["current"]) - sum(
            self.prior_game_stats["hp"]["current"]
        )

    def _xp_reward(self, new_state: Dict[str, any]) -> int:
        return sum(new_state["xp"]) - sum(self.prior_game_stats["xp"])

    def _levels_reward(self, new_state: Dict[str, any]) -> int:
        return sum(new_state["levels"]) - sum(self.prior_game_stats["levels"])

    def _badges_reward(self, new_state: Dict[str, any]) -> int:
        return new_state["badges"] - self.prior_game_stats["badges"]

    def _money_reward(self, new_state: Dict[str, any]) -> int:
        return new_state["money"] - self.prior_game_stats["money"]

    def _event_reward(self, new_state: Dict[str, any]) -> int:
        return sum(new_state["events"]) - sum(self.prior_game_stats["events"])

    def _stuck_reward(self, new_state: Dict[str, any]) -> int:
        """
        Calculates the reward for being stuck in the game environment.

        Args:
            new_state (Dict[str, any]): The new state of the game environment.

        Returns:
            int: The reward value for being stuck. Returns -5 if the agent has been stuck for more than 10 steps, otherwise returns 0.
        """
        if new_state["location"] == self.prior_game_stats["location"]:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count >= 10:
            self.stuck_count = 0
            return -5
        else:
            return 0

    def _location_reward(self, new_state: Dict[str, any]) -> int:
        """
        Calculates the reward based on the location of the agent in the game.

        Args:
            new_state (Dict[str, any]): The new state of the game.

        Returns:
            int: The reward based on the location.
        """
        # print(new_state["location"]["map_id"])
        if new_state["location"]["map_id"] not in self.seen_locations:
            self.seen_locations.add(new_state["location"]["map_id"])
            return 50  # Increased reward for new locations
        return 0

    @staticmethod
    def euclidean_distance(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    # new reward function
    def _distance_travelled_reward(self, new_state: Dict[str, any]) -> int:
        """
        Calculates the reward based on the distance travelled in the game environment.

        Args:
            new_state (Dict[str, any]): The new state of the game environment.

        Returns:
            int: The reward based on the distance travelled.
        """
        map_id = new_state["location"]["map_id"]
        current_position = (new_state["location"]["x"], new_state["location"]["y"])

        while self.buffer_frame > 0:
            self.buffer_frame -= 1
            return 0

        # if map id is different from previous map id
        if self.prior_game_stats["location"]["map_id"] != map_id:
            self.last_distance_travelled = 0
            # delete the initial position of the previous map
            try:
                del self.initial_positions[self.prior_game_stats["location"]["map_id"]]
            except:
                pass
            self.buffer_frame = 3
            return 0

        # Check if it's a new map or if we haven't recorded the initial position for this map yet
        if map_id not in self.initial_positions:
            self.initial_positions[map_id] = current_position
            return 0  # No reward for the first step in a new map

        # Calculate Euclidean distance from the initial position
        initial_position = self.initial_positions[map_id]
        distance = self.euclidean_distance(initial_position, current_position)

        if (
            distance <= self.last_distance_travelled
            and self.prior_game_stats["location"]["map_id"] == map_id
        ):
            return 0

        # if self.stuck_count >= 1:
        #     return 0
        # if self.get_if_stucked(self._get_location()) == 1:
        #     return 0

        # Optionally, reset the initial position to encourage exploration from the new point
        # self.initial_positions[map_id] = current_position  # Uncomment to reset on each reward calculation

        # Define the reward; for simplicity, the reward is just the distance moved
        reward = distance
        self.last_distance_travelled = distance

        return reward

    def _outside_reward(self, game_stats: Dict[str, any]) -> int:
        """
        Calculates the reward for being outside based on the tileset type.

        Args:
            game_stats (Dict[str, any]): The game statistics.

        Returns:
            int: The reward value. 2 for being outside, -2 for being indoors, 0 for other cases.
        """
        tileset_type = game_stats["location"]["map_id"]  # Read the tileset type
        if (
            tileset_type in self.outside_map_constants
        ):  # Value 2 indicates outside with flower animation
            self.outside_count += 1
        else:
            self.outside_count = 0

        if tileset_type in self.inside_map_constants:  # Indoors
            self.inside_count += 1
        else:
            self.inside_count = 0

        if self.outside_count >= 10:
            self.outside_count = 0
            # print("outside")
            return 2
        elif self.inside_count >= 10:
            self.inside_count = 0
            # print("inside")
            return -2

        return 0  # No reward or penalty for other cases (e.g., caves)

    def get_outside_inside(self, game_stats: Dict[str, any]) -> int:
        """
        Determines whether the player is currently outside or inside based on the map id.

        Args:
            game_stats (Dict[str, any]): The game statistics containing the map ID.

        Returns:
            int: A list representing whether the player is outside or inside. [1, 0] indicates outside, [0, 1] indicates inside.
        """
        tileset_type = game_stats["map_id"]  # Read the tileset type
        try:
            if (
                tileset_type in self.outside_map_constants
            ):  # Value 2 indicates outside with flower animation
                return [1, 0]

            if tileset_type in self.inside_map_constants:  # Indoors
                return [0, 1]
        except:
            return [0, 1]
        return [0, 1]

    def get_if_stucked(self, game_stats: Dict[str, any]) -> int:
        """
        Checks if the game is stucked based on the current game stats.

        Args:
            game_stats (Dict[str, any]): The current game stats.

        Returns:
            int: Returns 1 if the game is stucked, otherwise returns 0.
        """
        try:
            if game_stats == self.prior_game_stats["location"]:
                return 1
            return 0
        except:
            return 0

    def _check_if_done(self, game_stats: Dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return self.prior_game_stats["badges"] > 0

    def _get_location(self) -> Dict[str, any]:
        x_pos = self._read_m(0xD362)
        y_pos = self._read_m(0xD361)
        map_n = self._read_m(0xD35E)

        return {
            "x": x_pos,
            "y": y_pos,
            "map_id": map_n,
            "map": pkc.get_map_location(map_n),
        }

    def _get_party_size(self) -> int:
        return self._read_m(0xD163)

    def _get_badge_count(self) -> int:
        return self._bit_count(self._read_m(0xD356))

    def _is_grass_tile(self) -> bool:
        grass_tile_index = self._read_m(0xD535)
        player_sprite_status = self._read_m(0xC207)  # Assuming player is sprite 0
        return player_sprite_status == 0x80

    # in grass reward function that returns reward
    def _grass_reward(self, new_state: Dict[str, any]) -> int:
        if self._is_grass_tile():
            return 19
        return 0

    def _read_party_id(self) -> List[int]:
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/pokemon_constants.asm
        return [
            self._read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def _read_party_type(self) -> List[int]:
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/type_constants.asm
        return [
            self._read_m(addr)
            for addr in [
                0xD170,
                0xD171,
                0xD19C,
                0xD19D,
                0xD1C8,
                0xD1C9,
                0xD1F4,
                0xD1F5,
                0xD220,
                0xD221,
                0xD24C,
                0xD24D,
            ]
        ]

    def _read_party_level(self) -> List[int]:
        return [
            self._read_m(addr)
            for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]

    def _read_party_status(self) -> List[int]:
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/status_constants.asm
        return [
            self._read_m(addr)
            for addr in [0xD16F, 0xD19B, 0xD1C7, 0xD1F3, 0xD21F, 0xD24B]
        ]

    def _read_party_hp(self) -> Dict[str, List[int]]:
        hp = [
            self._read_hp(addr)
            for addr in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ]
        max_hp = [
            self._read_hp(addr)
            for addr in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ]
        return {"current": hp, "max": max_hp}

    def _read_party_xp(self) -> List[List[int]]:
        return [
            self._read_triple(addr)
            for addr in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
        ]

    def _read_hp(self, start: int) -> int:
        return 256 * self._read_m(start) + self._read_m(start + 1)

    def _read_caught_pokemon_count(self) -> int:
        return sum(
            list(self._bit_count(self._read_m(i)) for i in range(0xD2F7, 0xD30A))
        )

    def _read_seen_pokemon_count(self) -> int:
        return sum(
            list(self._bit_count(self._read_m(i)) for i in range(0xD30A, 0xD31D))
        )

    def _read_money(self) -> int:
        return (
            100 * 100 * self._read_bcd(self._read_m(0xD347))
            + 100 * self._read_bcd(self._read_m(0xD348))
            + self._read_bcd(self._read_m(0xD349))
        )

    def _read_events(self) -> List[int]:
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        # museum_ticket = (0xD754, 0)
        # base_event_flags = 13
        return [
            self._bit_count(self._read_m(i))
            for i in range(event_flags_start, event_flags_end)
        ]

    def _get_screen_background_tilemap(self):
        ### SIMILAR TO CURRENT pyboy.game_wrapper()._game_area_np(), BUT ONLY FOR BACKGROUND TILEMAP, SO NPC ARE SKIPPED
        bsm = self.pyboy.botsupport_manager()
        ((scx, scy), (wx, wy)) = bsm.screen().tilemap_position()
        tilemap = np.array(bsm.tilemap_background()[:, :])
        return np.roll(np.roll(tilemap, -scy // 8, axis=0), -scx // 8, axis=1)[:18, :20]

    def _get_screen_walkable_matrix(self):
        walkable_tiles_indexes = []
        collision_ptr = self.pyboy.get_memory_value(0xD530) + (
            self.pyboy.get_memory_value(0xD531) << 8
        )
        tileset_type = self.pyboy.get_memory_value(0xFFD7)
        if tileset_type > 0:
            grass_tile_index = self.pyboy.get_memory_value(0xD535)
            if grass_tile_index != 0xFF:
                walkable_tiles_indexes.append(grass_tile_index + 0x100)
        for i in range(0x180):
            tile_index = self.pyboy.get_memory_value(collision_ptr + i)
            if tile_index == 0xFF:
                break
            else:
                walkable_tiles_indexes.append(tile_index + 0x100)
        screen_tiles = self._get_screen_background_tilemap()
        bottom_left_screen_tiles = screen_tiles[1 : 1 + screen_tiles.shape[0] : 2, ::2]
        walkable_matrix = np.isin(
            bottom_left_screen_tiles, walkable_tiles_indexes
        ).astype(np.uint8)
        return walkable_matrix

    def game_area_collision(self):
        shape = (20, 18)
        game_area_section = (0, 0) + shape
        width = game_area_section[2]
        height = game_area_section[3]

        game_area = np.ndarray(shape=(height, width), dtype=np.uint32)
        _collision = self._get_screen_walkable_matrix()
        for i in range(height // 2):
            for j in range(width // 2):
                game_area[i * 2][j * 2 : j * 2 + 2] = _collision[i][j]
                game_area[i * 2 + 1][j * 2 : j * 2 + 2] = _collision[i][j]
        return game_area
