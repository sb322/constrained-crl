from envs.ant_maze import AntMaze
from envs.humanoid_maze import HumanoidMaze
from brax import envs

# Register custom environments with Brax
envs.register_environment('ant_u_maze', AntMaze, maze_layout_name='u_maze', maze_size_scaling=4.0)
envs.register_environment('ant_u_maze_eval', AntMaze, maze_layout_name='u_maze_eval', maze_size_scaling=4.0)
envs.register_environment('ant_u_maze_single_eval', AntMaze, maze_layout_name='u_maze_single_eval', maze_size_scaling=4.0)
envs.register_environment('ant_u2_maze', AntMaze, maze_layout_name='u2_maze', maze_size_scaling=4.0)
envs.register_environment('ant_u2_maze_eval', AntMaze, maze_layout_name='u2_maze_eval', maze_size_scaling=4.0)
envs.register_environment('ant_u3_maze', AntMaze, maze_layout_name='u3_maze', maze_size_scaling=4.0)
envs.register_environment('ant_u3_maze_eval', AntMaze, maze_layout_name='u3_maze_eval', maze_size_scaling=4.0)
envs.register_environment('ant_u3_maze_single_eval', AntMaze, maze_layout_name='u3_maze_single_eval', maze_size_scaling=4.0)
envs.register_environment('ant_u4_maze', AntMaze, maze_layout_name='u4_maze', maze_size_scaling=4.0)
envs.register_environment('ant_u4_maze_eval', AntMaze, maze_layout_name='u4_maze_eval', maze_size_scaling=4.0)
envs.register_environment('ant_u5_maze', AntMaze, maze_layout_name='u5_maze', maze_size_scaling=4.0)
envs.register_environment('ant_u5_maze_eval', AntMaze, maze_layout_name='u5_maze_eval', maze_size_scaling=4.0)
envs.register_environment('ant_u5_maze_single_eval', AntMaze, maze_layout_name='u5_maze_single_eval', maze_size_scaling=4.0)
envs.register_environment('ant_u6_maze', AntMaze, maze_layout_name='u6_maze', maze_size_scaling=4.0)
envs.register_environment('ant_u6_maze_eval', AntMaze, maze_layout_name='u6_maze_eval', maze_size_scaling=4.0)
envs.register_environment('ant_u7_maze', AntMaze, maze_layout_name='u7_maze', maze_size_scaling=4.0)
envs.register_environment('ant_u7_maze_eval', AntMaze, maze_layout_name='u7_maze_eval', maze_size_scaling=4.0)
envs.register_environment('ant_big_maze', AntMaze, maze_layout_name='big_maze', maze_size_scaling=4.0)
envs.register_environment('ant_big_maze_eval', AntMaze, maze_layout_name='big_maze_eval', maze_size_scaling=4.0)
envs.register_environment('ant_hardest_maze', AntMaze, maze_layout_name='hardest_maze', maze_size_scaling=4.0)

envs.register_environment('humanoid_u_maze', HumanoidMaze, maze_layout_name='u_maze', maze_size_scaling=2.0)
envs.register_environment('humanoid_u_maze_eval', HumanoidMaze, maze_layout_name='u_maze_eval', maze_size_scaling=2.0)
envs.register_environment('humanoid_big_maze', HumanoidMaze, maze_layout_name='big_maze', maze_size_scaling=2.0)
envs.register_environment('humanoid_big_maze_eval', HumanoidMaze, maze_layout_name='big_maze_eval', maze_size_scaling=2.0)
envs.register_environment('humanoid_hardest_maze', HumanoidMaze, maze_layout_name='hardest_maze', maze_size_scaling=2.0)
