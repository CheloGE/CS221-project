class CustomPrint():
    def __init__(self, env):
        self.swapped_dict = {value: key for key, value in env.action_type.actions_indexes.items()}
        self.prev_output_len_action = 0
        self.prev_output_len_obs = 0
    def print_action(self, action):
        action_str = f"action: {self.swapped_dict[action]}"
        print(" " * self.prev_output_len_action, end='\r')
        print(action_str, end="\r")
        self.prev_output_len_action = len(action_str)
    def print_obs(self, obs):
        print()
        for i, row in enumerate(obs):
            presence, x, y, vx, vy = row.round(2)
            vehicle_name = f"vehicle {i}" if i!=0 else f"ego-vehicle"
            print(f"{vehicle_name}: presence: {presence}, x: {x:.4f}, y: {y:.4f}, vx: {vx:.4f}, vy: {vy:.4f}")