import pygame
import math, random
import params

BALL_RADIUS = int(params.HEIGHT * 0.04)
BALL_COLOR  = (0, 255, 0)
PUSH_ACC = 0.01
MAX_V_X = 7
MAX_ANGLE_SPEED = 0.17
GRAV_ACC = 1.0
JUMP_ACC = BALL_RADIUS * 0.89
DRAG = 0.007
HIT_DRAG = 0.70 ## (0.0, 1.0]

DIAMOND_RADIUS = int(params.HEIGHT * 0.04)
DIAMOND_COLOR  = (0, 41, 107)

OBSTACLE_COLOR = (0, 0, 0)

class Obstacle(object):

	def __init__(self, surface, color, rect):
		self.surface = surface
		self.color = color
		self.x = rect[0]
		self.y = rect[1]
		self.w = rect[2]
		self.h = rect[3]

	def draw(self):
		pygame.draw.rect(self.surface, self.color, (self.x, self.y, self.w, self.h), 0)

class Ball(object):

	def __init__(self, surface, color, x, y, radius, dotAngle, obstacles):
		self.surface = surface
		self.color = color
		self.x = int(x)
		self.y = int(y)
		self.radius = radius
		self.dotRadius = int(self.radius * 0.5)
		self.dotAngle = dotAngle
		self.dotX = int(self.x + self.dotRadius * math.cos(self.dotAngle))
		self.dotY = int(self.y + self.dotRadius * math.sin(self.dotAngle))
		self.minX = 0 + self.radius
		self.maxX = params.WIDTH - self.radius
		self.minY = 0 + self.radius
		self.maxY = params.HEIGHT - self.radius
		self.angleSpeed = 0.0
		self.angleAcc   = 0.0
		self.v_x = 0.0
		self.v_y = 0.0
		self.yAcc = 0.0
		self.pushedRight = False
		self.pushedLeft = False
		self.pushedY = False
		self.isPushable = False

		self.obstacles = obstacles

	def isOnFloor(self):
		return (self.y == self.maxY)

	def setOnFloor(self):
		self.y = self.maxY

	def draw(self):
		pygame.draw.circle(self.surface, (0, 0, 0), (int(self.x), int(self.y)), int(self.radius + 2), 1)					## Border of ball
		pygame.draw.circle(self.surface, self.color, (int(self.x), int(self.y)), int(self.radius), 0)
		pygame.draw.circle(self.surface, (0, 0, 0),  (int(self.dotX), int(self.dotY)), int(self.radius * 0.1), 0)		## Black dot on the ball

	def check(self):
		self.minX = 0 + self.radius
		self.maxX = params.WIDTH - self.radius
		self.minY = 0 + self.radius
		self.maxY = params.HEIGHT - self.radius
		for obstacle in self.obstacles:
			if obstacle is not None:
				if (self.x + self.radius >= obstacle.x) and (self.x - self.radius <= obstacle.x + obstacle.w):
					if self.y + self.radius <= obstacle.y + 1:
						if obstacle.y - self.radius < self.maxY:
							self.maxY = obstacle.y - self.radius
					elif self.y - self.radius >= obstacle.y + obstacle.h - 1:
						if obstacle.y + obstacle.h + self.radius > self.minY:
							self.minY = obstacle.y + obstacle.h + self.radius
				if (self.y + self.radius >= obstacle.y) and (self.y - self.radius <= obstacle.y + obstacle.h):
					if self.x + self.radius <= obstacle.x + 1:
						if obstacle.x - self.radius < self.maxX:
							self.maxX = obstacle.x - self.radius
					elif self.x - self.radius >= obstacle.x + obstacle.w - 1:
						if obstacle.x + obstacle.w + self.radius > self.minX:
							self.minX = obstacle.x + obstacle.w + self.radius

		if int(self.y + self.v_y) < self.minY:		## Out from top
			self.y = self.minY						## Set to top
			self.v_y *= -HIT_DRAG
		if int(self.y + self.v_y) > self.maxY:		## Out from bottom
			self.y = self.maxY						## Set to bottom
			self.v_y *= -HIT_DRAG
		if int(self.x + self.v_x) < self.minX:		## Out from left
			self.x = self.minX						## Set to left
			self.v_y *= HIT_DRAG
			self.angleSpeed *= -1.0
			self.v_x = self.radius * self.angleSpeed
		if int(self.x + self.v_x) > self.maxX:		## Out from right
			self.x = self.maxX						## Set to right
			self.v_y *= HIT_DRAG
			self.angleSpeed *= -1.0
			self.v_x = self.radius * self.angleSpeed

	def rotate(self):
		if math.fabs(self.angleSpeed + self.angleAcc) <= MAX_ANGLE_SPEED:
			self.angleSpeed += self.angleAcc
		# if math.fabs(self.radius * (self.angleSpeed + self.angleAcc)) <= MAX_V_X:
		# 	self.angleSpeed += self.angleAcc
		self.dotAngle += self.angleSpeed
		self.dotX = int(self.x + self.dotRadius * math.cos(self.dotAngle))
		self.dotY = int(self.y + self.dotRadius * math.sin(self.dotAngle))

	def roll(self):
		if self.pushedRight and (not self.pushedLeft):
			self.angleAcc = PUSH_ACC
		elif self.pushedLeft and (not self.pushedRight):
			self.angleAcc = -PUSH_ACC
		if (not self.pushedRight) and (not self.pushedLeft):
			self.angleAcc = -(DRAG * self.angleSpeed)
		if math.fabs(self.radius * (self.angleSpeed + self.angleAcc)) <= MAX_V_X:
			self.angleSpeed += self.angleAcc
			self.v_x = self.radius * self.angleSpeed
		if int(self.v_x) != 0:		## Prevent sliding (no translation, but rotation)
			self.x += int(self.v_x)
			self.dotAngle += self.angleSpeed
			self.dotX = int(self.x + self.dotRadius * math.cos(self.dotAngle))
			self.dotY = int(self.y + self.dotRadius * math.sin(self.dotAngle))

	def fall(self):
		self.yAcc = GRAV_ACC
		self.v_y += self.yAcc
		self.y += int(self.v_y)
		self.x += int(self.v_x)
		self.dotY = int(self.y + self.dotRadius * math.sin(self.dotAngle))
		self.dotX = int(self.x + self.dotRadius * math.cos(self.dotAngle))

	def jump(self):
		if self.pushedY:
			self.yAcc = -JUMP_ACC
		else:
			self.yAcc = GRAV_ACC
		if math.fabs(self.v_y) < self.yAcc:
			self.v_y = 0.0
			self.setOnFloor()
		else:
			self.v_y += self.yAcc
			self.y += int(self.v_y)
			self.x += int(self.v_x)
			self.dotY = int(self.y + self.dotRadius * math.sin(self.dotAngle))
			self.dotX = int(self.x + self.dotRadius * math.cos(self.dotAngle))

	def grow(self):
		pass

	def shrink(self):
		pass

	def act(self):
		self.check()
		if self.isOnFloor():
			if self.v_y == 0.0:
				self.isPushable = True
			self.roll()
			self.jump()
		if not self.isOnFloor():
			self.isPushable = False
			self.rotate()
			self.fall()
		# self.grow()
		# self.shrink()
		self.draw()

class Diamond(object):

	def __init__(self, surface, color, x, y, radius):
		self.surface = surface
		self.color = color
		self.x = x
		self.y = y
		self.radius = radius
		self.setVertices()
		self.exists = True

	def setVertices(self):
		self.vertices =[(self.x + self.radius, self.y), (self.x, self.y + self.radius), \
						(self.x - self.radius, self.y), (self.x, self.y - self.radius)]

	def draw(self):
		if self.exists:
			pygame.draw.polygon(self.surface, self.color, self.vertices, 0)
			pygame.draw.polygon(self.surface, (0, 0, 0), self.vertices, 2)		## Border

	def getDistance(self, agent):
		return math.sqrt((self.x - agent.x)**2 + (self.y - agent.y)**2)

	def isCollectedBy(self, agent):
		return self.getDistance(agent) <= self.radius + agent.radius
