import argparse, time, math, sys, random, os
import pygame
import cv2
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas
import imageio
from skimage.exposure import rescale_intensity
import params, objects, game_setup
import objects

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

WIDTH, HEIGHT = params.WIDTH, params.HEIGHT
SHAPE_REDUCE_RATE = params.SHAPE_REDUCE_RATE
BACKGROUND = (116, 86, 255)
FPS = 60
FRAME_PER_ACTION = 8

MAX_MEMORY = 1000
N_ACTIONS = params.N_ACTIONS  # [nothing, push or release left, push or release right, up]
ACTIONS = {0:"nothing", 1:("push left", "release left"), 2:("push right", "release right"), 3:"jump"}
INPUT_STACK = params.INPUT_STACK

GAMMA = 0.99
EPSILON_INIT = 1.0
EPSILON_LOW = 0.1
EPSILON_DECAY = 0.9999
ACTION_PER_TRAIN = 1

BATCH_SIZE = 64
SAVE_PER_ITER = 100
MAX_REFRESH = 2
REWARD_OUT = 2.0

GIF = False

parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', help="Train ai")
parser.add_argument("--test", action='store_true', help="Test ai")
parser.add_argument("-s", "--save", type=str, help="Save path of model")
parser.add_argument("-l", "--load", type=str, help="Load path of model")
parser.add_argument("--setup", type=int, help="Game setup number")
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("-g", "--game_over", type=int, help="Game over score")
parser.add_argument("-v", "--verbose", action='store_true', help="Show more info")
args = vars(parser.parse_args())
TRAIN = args["train"]
TEST = args["test"]
SAVE = args["save"]
LOAD = args["load"]
SETUP = args["setup"]
LR = args["lr"]
GAME_OVER = args["game_over"]
VERBOSE = args["verbose"]

def processSurface(image):
	image = np.where(image == objects.DIAMOND_COLOR, np.array([255, 255, 255], dtype=np.uint8), image)
	grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)		## Convert to gray
	grayImage = np.fliplr(cv2.rotate(grayImage, cv2.ROTATE_90_CLOCKWISE))
	resized = cv2.resize(grayImage, (int(SHAPE_REDUCE_RATE * params.WIDTH), int(SHAPE_REDUCE_RATE * params.HEIGHT)), interpolation=cv2.INTER_NEAREST)
	final = rescale_intensity(resized, out_range=(0, 255))
	# plt.imshow(grayImage, cmap='Greys_r')
	# plt.show()
	return final

def getModelProps(modelName):
	df = pandas.read_csv('models.csv', sep=",", index_col=0)
	data = df.values
	modelNames = data[:, 0]
	if modelName in modelNames:
		row = np.where(data[:, 0] == modelName)[0][0]
		return data[row]
	else:
		return False

def saveModelProps(modelName, conv, fc, lr, epsilon, loss, fpa, apt, ntrain):
	df = pandas.read_csv('models.csv', sep=",", index_col=0)
	data = df.values
	modelNames = data[:, 0]
	if modelName in modelNames:
		row = np.where(data[:, 0] == modelName)[0][0]
		data[row][3] = lr
		data[row][4] = epsilon
		data[row][5] = loss
		data[row][6] = fpa
		data[row][7] = apt
		data[row][8] = ntrain
	else:
		new_row = np.array([[modelName, conv, fc, lr, epsilon, loss, fpa, apt, ntrain]])
		data = np.append(data, new_row, axis=0)
	res = pandas.DataFrame(data, columns=df.columns)
	res.to_csv("models.csv")


def play_as_ai():
	import network

	pygame.init()
	surface = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
	pygame.display.set_caption('Geometry Friends')
	clock = pygame.time.Clock()
	font = pygame.font.SysFont("Lato", 29)
	font.set_bold(True)

	gameQuit = False
	if TEST:
		setup = game_setup.setups[SETUP](surface)

	start_ticks = pygame.time.get_ticks() ## Start tick

	def stepFrame(actions_t, setup, collectTime):
		action_index = np.argmax(actions_t)	## Highest action index
		obstacles, ball, diamonds = (setup.obstacles, setup.ball, setup.diamonds)

		## Do nothing
		if action_index == 0:
			pass
		## Push or release left
		elif action_index == 1:
			ball.pushedLeft = not ball.pushedLeft
			ball.pushedRight = False
		## Push or release right
		elif action_index == 2:
			ball.pushedRight = not ball.pushedRight
			ball.pushedLeft = False
		## Jump
		elif action_index == 3 and ball.isPushable:
			ball.pushedY = True

		reward = max(-REWARD_OUT, REWARD_OUT * (collectTime / (setup.maxCollectTime * -1.0)))
		rewardPerDiamond = +1.0
		#rewardPerDiamond = 1.0 / len(diamonds)
		#rewardPerDiamond = +10.0
		## For given action update frames by FRAME_PER_ACTION
		for i in xrange(FRAME_PER_ACTION):
			surface.fill(BACKGROUND)
			ball.act()
			ball.pushedY = False

			for diamond in diamonds:
				if diamond.exists and diamond.isCollectedBy(ball):
					diamond.exists = False
					setup.scoreCounter += 1

					reward = rewardPerDiamond

				diamond.draw()

			for obstacle in obstacles:
				if obstacle is not None:
					obstacle.draw()


			setup.printScoreAndTime(surface=surface, start_ticks=setup.start_ticks, font=font)
			pygame.display.update()
			clock.tick(FPS)
			if reward > 0.0:
				break

		while not ball.isOnFloor():
			surface.fill(BACKGROUND)
			ball.act()
			ball.pushedY = False

			for diamond in diamonds:
				if diamond.exists and diamond.isCollectedBy(ball):
					diamond.exists = False
					setup.scoreCounter += 1

					reward = rewardPerDiamond

				diamond.draw()

			for obstacle in obstacles:
				if obstacle is not None:
					obstacle.draw()

			setup.printScoreAndTime(surface=surface, start_ticks=setup.start_ticks, font=font)
			pygame.display.update()
			clock.tick(FPS)

		image = pygame.surfarray.array3d(pygame.display.get_surface())
		termination = (setup.scoreCounter / len(diamonds)) >= GAME_OVER

		return image, reward, termination

	def train(model):
		if LOAD is None:
			modelName = SAVE.split("/")[1].rstrip(".h5")
			print "Model {} is created.".format(modelName)
			SAVE_PATH = SAVE
		else:
			modelName = LOAD.split("/")[1].rstrip(".h5")
			print "Model {} is restored.".format(modelName)
			SAVE_PATH = LOAD

		if getModelProps(modelName) is False:
			iteration = 0
			loss = 0.0
			epsilon = EPSILON_INIT
		else:
			modelProps = getModelProps(modelName)
			iteration = modelProps[7] * modelProps[8]
			loss = modelProps[5]
			epsilonLast = modelProps[4]
			epsilon = epsilonLast

		memory = deque()

		setup = random.choice(game_setup.setups)(surface)
		# Get the first state by doing nothing
		no_actions = np.zeros(N_ACTIONS)
		no_actions[0] = 1
		image_t, reward_t, termination = stepFrame(no_actions, setup, 1)

		image_t = processSurface(image_t)	## Convert gray, resize to 1/10

		s_t = np.hstack([image_t] * INPUT_STACK)
		s_t = s_t.reshape(1, s_t.shape[0], image_t.shape[1], INPUT_STACK) ## 1x(HEIGHT * 0.1)x(WIDTH * 0.1)xINPUT_STACK

		t = 0	## TOTAL ACTIONS MADE
		gameQuit = False
		collectTime = 0
		totalCollect = 0
		train_start_ticks = pygame.time.get_ticks() ## Start tick
		while not gameQuit:

			## If all diamonds collected start new random setup, set new s_t
			if all([not diamond.exists for diamond in setup.diamonds]) or (setup.nRefresh >= MAX_REFRESH):
				totalCollect += setup.scoreCounter
				setup = random.choice(game_setup.setups)(surface)
				no_actions = np.zeros(N_ACTIONS)
				no_actions[0] = 1
				image_t, reward_t, termination = stepFrame(no_actions, setup, 1)
				image_t = processSurface(image_t)	## Convert gray, resize to 1/10
				s_t = np.hstack([image_t] * INPUT_STACK)
				s_t = s_t.reshape(1, s_t.shape[0], image_t.shape[1], INPUT_STACK) ## 1x(HEIGHT * 0.1)x(WIDTH * 0.1)xINPUT_STACK
				collectTime = 0
				print "\n", "Total score:", totalCollect, "Total time:", (pygame.time.get_ticks() - train_start_ticks) // 1000, "\n"

			## If no diamond collected refresh the setup, set new s_t
			if reward_t == -REWARD_OUT:
				setup.refresh()
				setup.start_ticks = pygame.time.get_ticks()
				totalCollect += setup.scoreCounter
				setup.scoreCounter = 0
				setup.nRefresh += 1
				no_actions = np.zeros(N_ACTIONS)
				no_actions[0] = 1
				image_t, reward_t, termination = stepFrame(no_actions, setup, 1)
				image_t = processSurface(image_t)	## Convert gray, resize to 1/10
				s_t = np.hstack([image_t] * INPUT_STACK)
				s_t = s_t.reshape(1, s_t.shape[0], image_t.shape[1], INPUT_STACK) ## 1x(HEIGHT * 0.1)x(WIDTH * 0.1)xINPUT_STACK
				collectTime = 0
				print "\n", "Total score:", totalCollect, "Total time:", (pygame.time.get_ticks() - train_start_ticks) // 1000, "\n"

			else:
				collectTime += FRAME_PER_ACTION / (1.0 * FPS)

			## Interruptions
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					gameQuit = True

				## Keyboard interruptions
				if event.type == pygame.KEYDOWN:
					## 'q' to exit
					if event.key == pygame.K_q:
						gameQuit = True
					## 'r' to refresh
					if event.key == pygame.K_r:
						setup.refresh()

			actions_t = np.zeros([N_ACTIONS])
			targets = np.zeros([N_ACTIONS])

			## Choose an action epsilon greedy
			if random.random() <= epsilon:
				if VERBOSE:
					print("RANDOM ACTION")
				action_index = random.randrange(N_ACTIONS)
			else:
				q = model.predict(s_t)       ## Input a stack of 4 images, get the prediction
				action_index = np.argmax(q)
			if action_index == 0 and setup.ball.v_x <= 0.1:
				action_index = random.choice([1, 2, 3])
			actions_t[action_index] = 1

			## Run the selected action and observed next state and reward
			image_t1, reward_t, termination = stepFrame(actions_t, setup, collectTime)

			## Determine action name
			if action_index == 1:
				if setup.ball.pushedLeft:
					actionName = ACTIONS[action_index][0]
				else:
					actionName = ACTIONS[action_index][1]
			elif action_index == 2:
				if setup.ball.pushedRight:
					actionName = ACTIONS[action_index][0]
				else:
					actionName = ACTIONS[action_index][1]
			else:
				actionName = ACTIONS[action_index]

			if VERBOSE:
				print "Action:", actionName , "Reward:", reward_t

			image_t1 = processSurface(image_t1)

			prevStack = s_t.reshape(s_t.shape[1], s_t.shape[2] * s_t.shape[3])
			prevStack = prevStack[:, s_t.shape[2]:]
			image_t1 = np.hstack((prevStack, image_t1))
			s_t1 = image_t1.reshape(1, image_t1.shape[0], image_t1.shape[1] / INPUT_STACK, INPUT_STACK) ## 1x(HEIGHT * 0.1)x(WIDTH * 0.1)xINPUT_STACK


			### SEE STACKS
			# plt.imshow(s_t1.reshape(s_t1.shape[1], s_t1.shape[2] * s_t1.shape[3]), cmap='Greys_r')
			# plt.show()


			## Store the transition in memory
			memory.append((s_t, action_index, reward_t, s_t1, termination))

			if len(memory) > MAX_MEMORY:
				memory.popleft()

			## Get batch and update
			if t % ACTION_PER_TRAIN == 0:
				loss = 0.0

				if t >= BATCH_SIZE:
					batch = random.sample(memory, BATCH_SIZE)

					inputs = np.zeros((BATCH_SIZE, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
					targets = np.zeros((BATCH_SIZE, N_ACTIONS))

					#Now we do the experience replay
					for i in xrange(BATCH_SIZE):
						s = batch[i][0]
						a_index = batch[i][1]
						r = batch[i][2]
						s1 = batch[i][3]
						over = batch[i][4]

						inputs[i:i + 1] = s				## Input is state 's'
						targets[i] = model.predict(s)			## Output 'actions_t' for input 's'
						Q_s1a1 = model.predict(s1)			## Output 'actions_t1' for input 's1'

						if over:
							targets[i, a_index] = r
						else:
							targets[i, a_index] = r + GAMMA * np.max(Q_s1a1)

					# print("Actions_0", "\t", targets[0])
					loss += model.train_on_batch(inputs, targets)

				if epsilon * EPSILON_DECAY > EPSILON_LOW:
					epsilon *= EPSILON_DECAY	## Decay epsilon

				print "Loss:", loss, "Epsilon:", epsilon, "Iteration:", iteration

			## Termination
			if memory[-1][-1]:
				print("Train finished!")
				model.save(SAVE)
				saveModelProps(modelName, params.N_CONV, params.N_FC, LR, epsilon, loss, \
				 				FRAME_PER_ACTION, ACTION_PER_TRAIN, iteration / ACTION_PER_TRAIN)
				print("\nModel {} saved.\n".format(SAVE))
				break
			else:
				s_t = s_t1
				t += 1
				if iteration % 10000 == 0:
					SAVE_PATH = SAVE.rstrip(".h5") + "_" + str(iteration) + ".h5"
					modelName = SAVE_PATH.split("/")[1].rstrip(".h5")
				iteration += 1

				# Save progress
				if t % SAVE_PER_ITER == 0:
					model.save(SAVE_PATH)
					saveModelProps(modelName, params.N_CONV, params.N_FC, LR, epsilon, loss, \
					 				FRAME_PER_ACTION, ACTION_PER_TRAIN, iteration / ACTION_PER_TRAIN)
					print("\nModel {} saved.\n".format(SAVE_PATH))

	def test(model):
		# from keras.models import load_model
		# one = load_model("models/huger_deeper.h5")

		no_actions = np.zeros(N_ACTIONS)
		no_actions[0] = 1
		image_t, reward_0, termination = stepFrame(no_actions, setup, 0)

		image_t = processSurface(image_t)	## Convert gray, resize to 1/10

		s_t = np.hstack([image_t] * INPUT_STACK)
		s_t = s_t.reshape(1, s_t.shape[0], image_t.shape[1], INPUT_STACK) ## 1x(HEIGHT * 0.1)x(WIDTH * 0.1)xINPUT_STACK

		epsilon = EPSILON_LOW
		gameQuit = False
		totalCollect = 0
		test_start_ticks = pygame.time.get_ticks() ## Start tick
		while not gameQuit:
			## If all diamonds collected start refresh setup
			if all([not diamond.exists for diamond in setup.diamonds]):
				setup.refresh()
				totalCollect += setup.scoreCounter
				setup.start_ticks = pygame.time.get_ticks()
				setup.scoreCounter = 0

				print "\n", "Total score:", totalCollect, "Total time:", (pygame.time.get_ticks() - test_start_ticks) // 1000, "\n"

			actions_t = np.zeros(N_ACTIONS)

			## Interruptions
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					gameQuit = True

				## Keyboard interruptions
				if event.type == pygame.KEYDOWN:
					## 'q' to exit
					if event.key == pygame.K_q:
						totalCollect += setup.scoreCounter
						gameQuit = True
					## 'r' to refresh
					if event.key == pygame.K_r:
						setup.refresh()
						setup.start_ticks = pygame.time.get_ticks()
						totalCollect += setup.scoreCounter
						setup.scoreCounter = 0

			if random.random() <= epsilon:
				if VERBOSE:
					print("RANDOM ACTION")
				action_index = random.randrange(N_ACTIONS)
			else:
				q = model.predict(s_t)       ## Input a stack of 4 images, get the prediction
				action_index = np.argmax(q)
				# ####
				# q_one = one.predict(s_t)
				# if max(np.max(q), np.max(q_one)) == np.max(q):
				# 	action_index = np.argmax(q)
				# else:
				# 	action_index = np.argmax(q_one)
				# ####

			if action_index == 0 and setup.ball.v_x < 1.0:
				action_index = random.choice([1, 2, 3])
			actions_t[action_index] = 1

			image_t1, reward_t, termination = stepFrame(actions_t, setup, 1)

			## Determine action name
			if action_index == 1:
				if setup.ball.pushedLeft:
					actionName = ACTIONS[action_index][0]
				else:
					actionName = ACTIONS[action_index][1]
			elif action_index == 2:
				if setup.ball.pushedRight:
					actionName = ACTIONS[action_index][0]
				else:
					actionName = ACTIONS[action_index][1]
			else:
				actionName = ACTIONS[action_index]

			if VERBOSE:
				print "Action:", actionName , "Reward:", reward_t

			image_t1 = processSurface(image_t1)

			prevStack = s_t.reshape(s_t.shape[1], s_t.shape[2] * s_t.shape[3])
			prevStack = prevStack[:, s_t.shape[2]:]
			image_t1 = np.hstack((prevStack, image_t1))
			s_t1 = image_t1.reshape(1, image_t1.shape[0], image_t1.shape[1] / INPUT_STACK, INPUT_STACK) ## 1x(HEIGHT * 0.1)x(WIDTH * 0.1)xINPUT_STACK

			## Termination
			if termination:
				print("Test finished!")
				break
			else:
				s_t = s_t1

		name, score = setup.saveScore(start_ticks=test_start_ticks, totalCollect=setup.scoreCounter, player="AI")
		print("PLAYER {}, TIME PER SCORE {}, FOR SETUP {}".format(name, score, SETUP))

	if TRAIN:
		print("TRAINING")
		train(network.model)
	elif TEST:
		print("TESTING")
		test(network.model)

def play_as_human():

	pygame.init()
	surface = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
	pygame.display.set_caption('Geometry Friends')
	clock = pygame.time.Clock()
	font = pygame.font.SysFont("Lato", 22)
	font.set_bold(True)

	gameQuit = False
	setup = game_setup.setups[SETUP](surface)
	obstacles, ball, diamonds = (setup.obstacles, setup.ball, setup.diamonds)


	if GIF:
		t = 0
	start_ticks = pygame.time.get_ticks() ## Start tick
	while not gameQuit:
		for event in pygame.event.get():
			# print(event)
			if event.type == pygame.QUIT:
				gameQuit = True

			## Press the arrows
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_RIGHT:
					ball.pushedRight = True
				if event.key == pygame.K_LEFT:
					ball.pushedLeft = True
				if event.key == pygame.K_UP and ball.isPushable:
					ball.pushedY = True

				## 'q' to exit
				if event.key == pygame.K_q:
					gameQuit = True

				## 'r' to reset
				if event.key == pygame.K_r:
					play_as_human()

			## Release the arrows
			if event.type == pygame.KEYUP:
				if (event.key == pygame.K_RIGHT):
					ball.pushedRight = False
				if (event.key == pygame.K_LEFT):
					ball.pushedLeft = False

		surface.fill(BACKGROUND)
		ball.act()
		ball.pushedY = False
		for diamond in diamonds:
			if diamond.exists and diamond.isCollectedBy(ball):
				diamond.exists = False
				setup.scoreCounter += 1
			diamond.draw()

		if all([not diamond.exists for diamond in diamonds]):
			setup.refresh()

		for obstacle in obstacles:
			if obstacle is not None:
				obstacle.draw()

		setup.printScoreAndTime(surface=surface, start_ticks=start_ticks, font=font)

		if GIF:
			image = pygame.surfarray.array3d(pygame.display.get_surface())
			image = np.fliplr(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
			image = cv2.resize(image, (int(0.75 * params.WIDTH), int(0.75 * params.HEIGHT)), interpolation=cv2.INTER_NEAREST)
			plt.imsave("images/frame_{:0>3}.jpg".format(t), image)
			t += 1

		pygame.display.update()
		clock.tick(FPS)

		if (setup.scoreCounter / len(diamonds)) == GAME_OVER:
			break

	name, score = setup.saveScore(start_ticks=start_ticks, totalCollect=setup.scoreCounter, player="human")
	print("PLAYER {}, TIME PER SCORE {}, FOR SETUP {}".format(name, score, SETUP))

if __name__ == "__main__":
	if TRAIN or TEST:
		play_as_ai()
	else:
		play_as_human()
		if GIF:
			images = []
			for root, dirs, filenames in os.walk("images/"):
				filenames = sorted(filenames)
				t = 0
				for filename in filenames:
					filepath = os.path.join(root, filename)
					if t % 2 == 0:
						images.append(imageio.imread(filepath))
					os.remove(filepath)
					t += 1
			imageio.mimsave('images/human_game.gif', images)
	pygame.quit()
	quit()
