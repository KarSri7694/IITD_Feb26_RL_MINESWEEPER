#!/usr/bin/python3
"""
Minesweeper Agent
This agent plays Minesweeper by analyzing game states and generating actions
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from .minesweeper_model import MinesweeperAgent


class MinesweeperPlayer:
    """Agent responsible for playing Minesweeper"""

    def __init__(self, **kwargs):
        self.agent = MinesweeperAgent(**kwargs)

    def build_prompt(self, game_state: Dict[str, Any]) -> tuple[str, str]:
        """
        Generate prompt for LLM from game state.

        Args:
            game_state: Dictionary containing board, rows, cols, mines, etc.

        Returns:
            (prompt, system_prompt)
        """
        sys_prompt = "You output JSON actions for Minesweeper. No text, only JSON."

        board = game_state.get("board", [])
        if isinstance(board, str):
            board = json.loads(board)
            
        # 2. Calculate flags and revealed if missing
        flags_placed = game_state.get("flags_placed")
        if flags_placed is None:
            # Count 'F's in the board
            flags_placed = sum(row.count('F') for row in board)
            
        cells_revealed = game_state.get("cells_revealed")
        if cells_revealed is None:
            # Count anything that is not '.' or 'F'
            cells_revealed = sum(1 for row in board for cell in row if cell not in ['.', 'F'])
        move_history = game_state.get("move_history", [])
        state = {
            "board": board,
            "rows": game_state.get("rows", len(board)),
            "cols": game_state.get("cols", len(board[0]) if board else 0),
            "mines": game_state.get("mines", 0),
            "flags_placed": flags_placed,
            "cells_revealed": cells_revealed,
            "move_history": move_history
        }

        # Ultra-minimal prompt with example showing direct JSON output
        prompt = f"""
        You are an expert Minesweeper solver. Your goal is to maximize your score by playing optimally.

## Scoring System
- **Win Game**: +100 points (Flag all mines + Reveal all safe cells)
- **Flag Mine**: +15 points
- **Reveal Safe Cell (Logical)**: +15 points
- **Reveal Safe Cell (Guess)**: +10 points
- **Flag Incorrectly**: -10 points
- **Reveal Mine**: -25 points (Game Over)
- **Redundant Move (Reveal/Flag again)**: -12 / -8 points
- **Invalid Move (Out of bounds/JSON)**: -15 / -10 points
- **Excessive Flags**: -10 points

## Strategy
1. **Prioritize Safety**: Hitting a mine ends the round with a heavy penalty (-25).
2. **Maximize Points**: 
   - logically deduced safe cells give +15.
   - Flagging confirmed mines gives +15.
   - Guessing gives +10, but carries risk of hitting a mine.
3. **Avoid Penalties**: Never repeat a move. Ensure coordinates are valid. Do not flag more than remaining mines.

## Board State
{json.dumps(state)}

## Output Format
Respond with ONLY a JSON object (no other text outside the JSON):
```json
{{"type": "reveal", "row": <r>, "col": <c>}}
```
Or to flag a mine:
```json
{{"type": "flag", "row": <r>, "col": <c>}}
```

## Critical Rules
- DO NOT CLICK THE SAME GRID AS PREVIOUSLY CLICKED 
- DO NOT WRITE ANYTHING OTHER THAN DESCRIBED JSON. NO TEXT BEFORE OR AFTER, NO REASONING
- NEVER reveal a cell you've deduced is a mine.
- NEVER flag a cell you've deduced is safe.
- Prefer actions with CERTAINTY over guesses.
- When guessing, minimize risk — choose cells with the fewest adjacent unrevealed mines.
        
        """
        return prompt, sys_prompt

    def play_action(
        self, game_state: Dict[str, Any], **gen_kwargs
    ) -> tuple[Optional[Dict], Optional[int], Optional[float]]:
        """
        Generate a single action for the given game state.

        Args:
            game_state: Current game state
            **gen_kwargs: Generation parameters

        Returns:
            (action_dict, token_count, generation_time)
        """
        prompt, sys_prompt = self.build_prompt(game_state)
        response, tl, gt = self.agent.generate_response(prompt, sys_prompt, **gen_kwargs)

        # Parse the action from response
        action = self.parse_action(response)

        return action, tl, gt

    def parse_action(self, response: str) -> Optional[Dict]:
        """
        Extract JSON action from LLM response.
        Handles cases where multiple JSON objects are present (e.g., function calling format).

        Expected format:
        {"type": "reveal", "row": 2, "col": 3}
        or
        {"type": "flag", "row": 1, "col": 4}
        """
        try:
            # Find all potential JSON objects in response
            potential_jsons = []
            i = 0
            while i < len(response):
                start = response.find("{", i)
                if start == -1:
                    break

                # Try to find matching closing brace
                brace_count = 0
                end = start
                while end < len(response):
                    if response[end] == '{':
                        brace_count += 1
                    elif response[end] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            json_str = response[start:end+1]
                            try:
                                obj = json.loads(json_str)
                                potential_jsons.append(obj)
                            except:
                                pass
                            break
                    end += 1

                i = end + 1 if end < len(response) else len(response)

            # Find the first valid action object
            for obj in potential_jsons:
                if (isinstance(obj, dict) and
                    "type" in obj and
                    "row" in obj and
                    "col" in obj and
                    obj["type"] in ["reveal", "flag"]):
                    # Ensure row/col are integers
                    obj["row"] = int(obj["row"])
                    obj["col"] = int(obj["col"])
                    return obj

        except Exception as e:
            print(f"Failed to parse action: {e}")
            return None

        return None

    @staticmethod
    def save_action(action: Dict, file_path: str | Path) -> None:
        """Save action to JSON file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(action, f, indent=2)


# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # Command: python -m agents.minesweeper_agent --game_state_file game_state.json --output_file action.json
    argparser = argparse.ArgumentParser(
        description="Play Minesweeper using fine-tuned LLM."
    )
    argparser.add_argument(
        "--game_state_file",
        type=str,
        required=True,
        help="Input JSON file containing game state",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/action.json",
        help="Output file to save the action",
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging."
    )
    args = argparser.parse_args()

    # Load game state
    with open(args.game_state_file, "r") as f:
        game_state = json.load(f)

    # Initialize agent
    player = MinesweeperPlayer()

    # Load generation config
    gen_kwargs = {"tgps_show": args.verbose}
    config_file = Path("minesweeper_config.yaml")
    if config_file.exists():
        with open(config_file, "r") as f:
            gen_kwargs.update(yaml.safe_load(f))

    # Generate action
    action, tl, gt = player.play_action(game_state, **gen_kwargs)

    if args.verbose:
        print(f"Game State:")
        print(json.dumps(game_state, indent=2))
        print(f"\nGenerated Action:")
        print(json.dumps(action, indent=2))
        if tl and gt:
            print(f"\nStats: Tokens={tl}, Time={gt:.2f}s, TGPS={tl/gt:.2f}")

    if action:
        player.save_action(action, args.output_file)
        print(f"Action saved to {args.output_file}")
    else:
        print("ERROR: Failed to generate valid action!")
        # Save error indicator
        player.save_action({"error": "parse_failed"}, args.output_file)
