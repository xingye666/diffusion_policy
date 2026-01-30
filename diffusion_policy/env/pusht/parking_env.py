import gym
from gym import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import cv2
from collections import OrderedDict
import math  # 添加这行
import shapely.geometry as sg

def pymunk_to_shapely(body, shapes):
  geoms = list()
  for shape in shapes:
      if isinstance(shape, pymunk.shapes.Poly):
          verts = [body.local_to_world(v) for v in shape.get
          verts += [verts[0]]
          geoms.append(sg.Polygon(verts))
      else:
        raise RuntimeError(f'Unsupported shape type {type(shape)}')
  geom = sg.MultiPolygon(geoms)
  return geom

class ParkingEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)
  def __init__(self,
               legacy=False, block_cog=None, damping=None,
               render_size=500,#96
               render_action=True,
               success_threshold=0.95,
               reset_to_state=None): # 启用精确碰撞检测
      self._seed = None
      self.seed()

      self.window_size = 500#512  # The size of the PyGame window
      self.render_size = render_size
      self.sim_hz = 100
      self.control_hz = self.metadata['video.frames_per_second']
      self.k_p, self.k_v =25, 1#0.2# 8, 4#5, 1 #80, 25 #更柔和的参数
      self.legacy = legacy
      # 状态空间：车辆位置(x,y), 车辆角度, 目标位置(x,y), 目标角度
      # 使用与原始PushT相同的结构
      self.observation_space = spaces.Box(
          low=np.array([0, 0, 0, 0, -np.pi], dtype=np.float64),
          high=np.array([self.window_size, self.window_size,
                        self.window_size, self.window_size, np.pi*2], dtype=np.float64),
          shape=(5,),dtype=np.float64)
      # 动作空间：目标位置(x,y)
      
