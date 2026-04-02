import functools
from brax import envs
from envs.ant_maze import AntMaze
from envs.humanoid_maze import HumanoidMaze

def _ant(layout, scale=4.0):
    return functools.partial(AntMaze, maze_layout_name=layout, maze_size_scaling=scale)

def _hum(layout, scale=2.0):
    return functools.partial(HumanoidMaze, maze_layout_name=layout, maze_size_scaling=scale)

envs.register_environment('ant_u_maze',               _ant('u_maze'))
envs.register_environment('ant_u_maze_eval',           _ant('u_maze_eval'))
envs.register_environment('ant_u_maze_single_eval',    _ant('u_maze_single_eval'))
envs.register_environment('ant_u2_maze',               _ant('u2_maze'))
envs.register_environment('ant_u2_maze_eval',          _ant('u2_maze_eval'))
envs.register_environment('ant_u3_maze',               _ant('u3_maze'))
envs.register_environment('ant_u3_maze_eval',          _ant('u3_maze_eval'))
envs.register_environment('ant_u3_maze_single_eval',   _ant('u3_maze_single_eval'))
envs.register_environment('ant_u4_maze',               _ant('u4_maze'))
envs.register_environment('ant_u4_maze_eval',          _ant('u4_maze_eval'))
envs.register_environment('ant_u5_maze',               _ant('u5_maze'))
envs.register_environment('ant_u5_maze_eval',          _ant('u5_maze_eval'))
envs.register_environment('ant_u5_maze_single_eval',   _ant('u5_maze_single_eval'))
envs.register_environment('ant_u6_maze',               _ant('u6_maze'))
envs.register_environment('ant_u6_maze_eval',          _ant('u6_maze_eval'))
envs.register_environment('ant_u7_maze',               _ant('u7_maze'))
envs.register_environment('ant_u7_maze_eval',          _ant('u7_maze_eval'))
envs.register_environment('ant_big_maze',              _ant('big_maze'))
envs.register_environment('ant_big_maze_eval',         _ant('big_maze_eval'))
envs.register_environment('ant_hardest_maze',          _ant('hardest_maze'))

envs.register_environment('humanoid_u_maze',           _hum('u_maze'))
envs.register_environment('humanoid_u_maze_eval',      _hum('u_maze_eval'))
envs.register_environment('humanoid_big_maze',         _hum('big_maze'))
envs.register_environment('humanoid_big_maze_eval',    _hum('big_maze_eval'))
envs.register_environment('humanoid_hardest_maze',     _hum('hardest_maze'))
