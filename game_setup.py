import pygame
import math, random, time
import params
import main, objects

class Setup(object):

    def __init__(self, surface):
        self.surface = surface
        self.scoreCounter = 0
        self.nRefresh = 0
        self.start_ticks = pygame.time.get_ticks()

    def printScoreAndTime(self, surface, start_ticks, font):
        time = (pygame.time.get_ticks() - start_ticks) // 1000  ## Calculate seconds
        text1 = font.render("SCORE:{}".format(self.scoreCounter), True, (255, 255, 255))
        text2 = font.render("TIME:{} s".format(time), True, (255, 255, 255))
        surface.blit(text1, (params.WIDTH * 0.1 - text1.get_width() // 2, params.HEIGHT * 0.1 - text1.get_height() // 2))
        surface.blit(text2, (params.WIDTH * 0.9 - text2.get_width() // 2, params.HEIGHT * 0.1 - text2.get_height() // 2))

    def getTimePerScore(self, start_ticks, totalCollect):
        time = (pygame.time.get_ticks() - start_ticks) / 1000.0  ## Calculate seconds
        if totalCollect == 0:
            return 0
        else:
            return time / totalCollect

    def saveScore(self, start_ticks, totalCollect, player):
        score = self.getTimePerScore(start_ticks, totalCollect)
        bestPlayers = {}
        human_numbers = []
        ai_numbers = []
        with open("output/scores{}.txt".format(main.SETUP)) as f:
            lines = f.readlines()
            for line in lines:
                nameBefore, scoreBefore = line.split(" ")
                scoreBefore = float(scoreBefore.rstrip("\n"))
                bestPlayers[nameBefore] = scoreBefore
                if nameBefore.startswith("h"):
                    human_numbers.append(int(nameBefore[-4:]))
                else:
                    ai_numbers.append(int(nameBefore[-4:]))
        if player == "human":
            try:
                currentNum = max(human_numbers) + 1
            except ValueError:
                currentNum = 1
            playerName = "human" + str(currentNum).zfill(4)
        else:
            try:
                currentNum = max(ai_numbers) + 1
            except ValueError:
                currentNum = 1
            playerName = "AI_" + main.LOAD.split("/")[1].rstrip(".h5") + "_" + str(currentNum).zfill(4)

        bestPlayers[playerName] = score

        with open("output/scores{}.txt".format(main.SETUP), "w") as f:
            for name, sc in sorted(bestPlayers.items(), key=lambda x : x[1]):
                f.write(name + " " + str(sc) + "\n")

        return playerName, score

class Setup0(Setup):
    '''
    Basic setup. No obstacles. One diamond appear on random positions (at reachable positions).
    '''

    def __init__(self, surface):
        Setup.__init__(self, surface)

        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.9)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        dotAngle = 0.0
        ball = objects.Ball(surface=self.surface, color=objects.BALL_COLOR, x=ballX, y=ballY, radius=objects.BALL_RADIUS, dotAngle=dotAngle, obstacles=[None])

        diamondX = random.randint(2 * objects.DIAMOND_RADIUS, params.WIDTH - objects.DIAMOND_RADIUS)    ## Initial diamond x
        diamondY = random.randint(params.HEIGHT * 0.5 + 2 * objects.DIAMOND_RADIUS, params.HEIGHT - objects.DIAMOND_RADIUS) ## Initial diamond y
        diamond = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=diamondX, y=diamondY, radius=objects.DIAMOND_RADIUS)

        self.obstacles, self.ball, self.diamonds = ([None], ball, [diamond])

        self.maxCollectTime = 3  ## Ideal max completion time in seconds to collect all diamond

    def refresh(self):
        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.9)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        self.ball.angleSpeed = 0.0
        self.ball.angleAcc   = 0.0
        self.ball.v_x = 0.0
        self.ball.v_y = 0.0
        self.ball.yAcc = 0.0
        self.ball.pushedRight = False
        self.ball.pushedLeft = False
        self.ball.pushedY = False
        self.ball.isPushable = False
        self.ball.x, self.ball.y = (ballX, ballY)   ## Initial ball coordinates
        for diamond in self.diamonds:
            diamond.exists = True
            diamond.x = random.randint(2 * diamond.radius, params.WIDTH - 2 * diamond.radius)
            diamond.y = random.randint(params.HEIGHT * 0.6 + 2 * diamond.radius, params.HEIGHT - 2 * diamond.radius)
            diamond.setVertices()

class Setup1(Setup):
    '''
    One platform to reach diamond.
    '''

    def __init__(self, surface):
        Setup.__init__(self, surface)

        platformW = int(params.WIDTH * 0.3)
        platformH = int(params.HEIGHT * 0.05)
        platformX = int(params.WIDTH * 0.5 - platformW * 0.5)
        platformY = int(params.HEIGHT * 0.75 - platformH * 0.5)
        platform = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))
        obstacles = [platform]

        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.9)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        dotAngle = 0.0
        ball = objects.Ball(surface=self.surface, color=objects.BALL_COLOR, x=ballX, y=ballY, radius=objects.BALL_RADIUS, dotAngle=dotAngle, obstacles=obstacles)

        self.diamondX =     platformX + platformW * 0.5     ## Initial diamond x
        self.diamondY =     platformY - params.HEIGHT * 0.07 ## Initial diamond y
        diamond = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)
        diamonds = [diamond]

        self.obstacles, self.ball, self.diamonds = (obstacles, ball, diamonds)

        self.maxCollectTime = 5  ## Ideal max completion time in seconds to collect all diamond

    def refresh(self):
        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.9)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        self.ball.angleSpeed = 0.0
        self.ball.angleAcc   = 0.0
        self.ball.v_x = 0.0
        self.ball.v_y = 0.0
        self.ball.yAcc = 0.0
        self.ball.pushedRight = False
        self.ball.pushedLeft = False
        self.ball.pushedY = False
        self.ball.isPushable = False
        self.diamonds[0].exists = True
        self.ball.x, self.ball.y = (ballX, ballY)   ## Initial ball coordinates

class Setup2(Setup):
    '''
    Two diamonds, one platform at right.
    '''

    def __init__(self, surface):
        Setup.__init__(self, surface)

        platformW = int(params.WIDTH * 0.23)
        platformH = int(params.HEIGHT * 0.05)
        platformX = int(params.WIDTH * 0.8 - platformW * 0.5)
        platformY = int(params.HEIGHT * 0.80 - platformH * 0.5)
        platform = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))
        obstacles = [platform]

        self.diamondX =     platformX + platformW * 1.1 + objects.DIAMOND_RADIUS     ## Initial diamond x
        self.diamondY =     platformY + objects.DIAMOND_RADIUS ## Initial diamond y
        diamond1 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        ballX = random.randint(params.WIDTH * 0.4, params.WIDTH * 0.4)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        dotAngle = 0.0
        ball = objects.Ball(surface=self.surface, color=objects.BALL_COLOR, x=ballX, y=ballY, radius=objects.BALL_RADIUS, dotAngle=dotAngle, obstacles=obstacles)

        self.diamondX =     platformX + platformW * 0.9     ## Initial diamond x
        self.diamondY =     platformY - params.HEIGHT * 0.2 ## Initial diamond y
        diamond2 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        self.diamondX =     params.WIDTH * 0.2                               ## Initial diamond x
        self.diamondY =     params.HEIGHT * 0.7 + 4 * objects.DIAMOND_RADIUS ## Initial diamond y
        diamond3 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        self.diamondX =     params.WIDTH * 0.9                               ## Initial diamond x
        self.diamondY =     params.HEIGHT - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond4 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        diamonds = [diamond1, diamond2, diamond3, diamond4]

        self.obstacles, self.ball, self.diamonds = (obstacles, ball, diamonds)

        self.maxCollectTime = 9  ## Ideal max completion time in seconds to collect all diamond

    def refresh(self):
        time.sleep(0.5)
        ballX = random.randint(params.WIDTH * 0.4, params.WIDTH * 0.5)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        self.ball.angleSpeed = 0.0
        self.ball.angleAcc   = 0.0
        self.ball.v_x = 0.0
        self.ball.v_y = 0.0
        self.ball.yAcc = 0.0
        self.ball.pushedRight = False
        self.ball.pushedLeft = False
        self.ball.pushedY = False
        self.ball.isPushable = False
        for diamond in self.diamonds:
            diamond.exists = True
        self.ball.x, self.ball.y = (ballX, ballY)   ## Initial ball coordinates

class Setup3(Setup):
    '''
    One hidden diamond.
    '''

    def __init__(self, surface):
        Setup.__init__(self, surface)

        platformW = int(params.WIDTH * 0.57)
        platformH = int(params.HEIGHT * 0.05)
        platformX = int(params.WIDTH * 0.6 - platformW * 0.5)
        platformY = int(params.HEIGHT * 0.75 - platformH * 0.5)
        platform1 = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))

        platformW = int(params.WIDTH * 0.1)
        platformH = int(params.HEIGHT * 0.3)
        platformX = int(params.WIDTH * 0.7 - platformW * 0.5)
        platformY = int(params.HEIGHT * 0.9 - platformH * 0.5)
        platform2 = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))

        obstacles = [platform1, platform2]

        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.5)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        dotAngle = 0.0
        ball = objects.Ball(surface=self.surface, color=objects.BALL_COLOR, x=ballX, y=ballY, radius=objects.BALL_RADIUS, dotAngle=dotAngle, obstacles=obstacles)

        self.diamondX =     params.WIDTH * 0.9     ## Initial diamond x
        self.diamondY =     params.HEIGHT - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond1 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)
        self.diamondX =     params.WIDTH * 0.6     ## Initial diamond x
        self.diamondY =     params.HEIGHT - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond2 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)
        self.diamondX =     params.WIDTH * 0.75     ## Initial diamond x
        self.diamondY =     int(params.HEIGHT * 0.75 - params.HEIGHT * 0.05 * 0.5) - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond3 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        diamonds = [diamond1, diamond2, diamond3]

        self.obstacles, self.ball, self.diamonds = (obstacles, ball, diamonds)

        self.maxCollectTime = 9  ## Ideal max completion time in seconds to collect all diamond

    def refresh(self):
        time.sleep(0.5)
        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.4)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        self.ball.angleSpeed = 0.0
        self.ball.angleAcc   = 0.0
        self.ball.v_x = 0.0
        self.ball.v_y = 0.0
        self.ball.yAcc = 0.0
        self.ball.pushedRight = False
        self.ball.pushedLeft = False
        self.ball.pushedY = False
        self.ball.isPushable = False
        for diamond in self.diamonds:
            diamond.exists = True
        self.ball.x, self.ball.y = (ballX, ballY)   ## Initial ball coordinates

class Setup4(Setup):
    '''
    Two platforms.
    '''

    def __init__(self, surface):
        Setup.__init__(self, surface)

        platformW = int(params.WIDTH * 0.37)
        platformH = int(params.HEIGHT * 0.05)
        platformX = int(params.WIDTH * 0.2 - platformW * 0.5)
        platformY = int(params.HEIGHT * 0.7 - platformH * 0.5)
        platform1 = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))

        self.diamondX =     params.WIDTH * 0.2     ## Initial diamond x
        self.diamondY =     platformY - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond1 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        platformW = int(params.WIDTH * 0.31)
        platformH = int(params.HEIGHT * 0.05)
        platformX = int(params.WIDTH * 0.73 - platformW * 0.5)
        platformY = int(params.HEIGHT * 0.75 - platformH * 0.5)
        platform2 = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))

        self.diamondX =     params.WIDTH * 0.73     ## Initial diamond x
        self.diamondY =     platformY - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond2 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        obstacles = [platform1, platform2]

        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.5)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        dotAngle = 0.0
        ball = objects.Ball(surface=self.surface, color=objects.BALL_COLOR, x=ballX, y=ballY, radius=objects.BALL_RADIUS, dotAngle=dotAngle, obstacles=obstacles)

        self.diamondX =     params.WIDTH * 0.5 - objects.DIAMOND_RADIUS     ## Initial diamond x
        self.diamondY =     platformY - params.HEIGHT * 0.3 ## Initial diamond y
        diamond3 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        self.diamondX =     params.WIDTH * 0.8     ## Initial diamond x
        self.diamondY =     params.HEIGHT - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond4 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        diamonds = [diamond1, diamond2, diamond3, diamond4]

        self.obstacles, self.ball, self.diamonds = (obstacles, ball, diamonds)

        self.maxCollectTime = 9  ## Ideal max completion time in seconds to collect all diamond

    def refresh(self):
        time.sleep(0.5)
        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.5)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        self.ball.angleSpeed = 0.0
        self.ball.angleAcc   = 0.0
        self.ball.v_x = 0.0
        self.ball.v_y = 0.0
        self.ball.yAcc = 0.0
        self.ball.pushedRight = False
        self.ball.pushedLeft = False
        self.ball.pushedY = False
        self.ball.isPushable = False
        for diamond in self.diamonds:
            diamond.exists = True
        self.ball.x, self.ball.y = (ballX, ballY)   ## Initial ball coordinates

class Setup5(Setup):
    '''
    One platform left, three diamonds.
    '''

    def __init__(self, surface):
        Setup.__init__(self, surface)

        platformW = int(params.WIDTH * 0.37)
        platformH = int(params.HEIGHT * 0.05)
        platformX = int(params.WIDTH * 0.2 - platformW * 0.5)
        platformY = int(params.HEIGHT * 0.7 - platformH * 0.5)
        platform = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))

        self.diamondX =     params.WIDTH * 0.2     ## Initial diamond x
        self.diamondY =     platformY - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond1 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        obstacles = [platform]

        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.5)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        dotAngle = 0.0
        ball = objects.Ball(surface=self.surface, color=objects.BALL_COLOR, x=ballX, y=ballY, radius=objects.BALL_RADIUS, dotAngle=dotAngle, obstacles=obstacles)

        self.diamondX =     params.WIDTH * 0.5 - objects.DIAMOND_RADIUS     ## Initial diamond x
        self.diamondY =     platformY - params.HEIGHT * 0.3 ## Initial diamond y
        diamond2 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        self.diamondX =     params.WIDTH * 0.8     ## Initial diamond x
        self.diamondY =     params.HEIGHT - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond3 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        diamonds = [diamond1, diamond2, diamond3]

        self.obstacles, self.ball, self.diamonds = (obstacles, ball, diamonds)

        self.maxCollectTime = 9  ## Ideal max completion time in seconds to collect all diamond

    def refresh(self):
        time.sleep(0.5)
        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.5)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        self.ball.angleSpeed = 0.0
        self.ball.angleAcc   = 0.0
        self.ball.v_x = 0.0
        self.ball.v_y = 0.0
        self.ball.yAcc = 0.0
        self.ball.pushedRight = False
        self.ball.pushedLeft = False
        self.ball.pushedY = False
        self.ball.isPushable = False
        for diamond in self.diamonds:
            diamond.exists = True
        self.ball.x, self.ball.y = (ballX, ballY)   ## Initial ball coordinates

class Setup6(Setup):
    '''
    One platform right, three diamonds.
    '''

    def __init__(self, surface):
        Setup.__init__(self, surface)

        platformW = int(params.WIDTH * 0.31)
        platformH = int(params.HEIGHT * 0.05)
        platformX = int(params.WIDTH * 0.73 - platformW * 0.5)
        platformY = int(params.HEIGHT * 0.75 - platformH * 0.5)
        platform = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))

        self.diamondX =     params.WIDTH * 0.73     ## Initial diamond x
        self.diamondY =     platformY - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond1 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        obstacles = [platform]

        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.5)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        dotAngle = 0.0
        ball = objects.Ball(surface=self.surface, color=objects.BALL_COLOR, x=ballX, y=ballY, radius=objects.BALL_RADIUS, dotAngle=dotAngle, obstacles=obstacles)

        self.diamondX =     params.WIDTH * 0.5 - objects.DIAMOND_RADIUS     ## Initial diamond x
        self.diamondY =     platformY - params.HEIGHT * 0.3 ## Initial diamond y
        diamond2 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        self.diamondX =     params.WIDTH * 0.8     ## Initial diamond x
        self.diamondY =     params.HEIGHT - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond3 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        diamonds = [diamond1, diamond2, diamond3]

        self.obstacles, self.ball, self.diamonds = (obstacles, ball, diamonds)

        self.maxCollectTime = 9  ## Ideal max completion time in seconds to collect all diamond

    def refresh(self):
        time.sleep(0.5)
        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.5)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        self.ball.angleSpeed = 0.0
        self.ball.angleAcc   = 0.0
        self.ball.v_x = 0.0
        self.ball.v_y = 0.0
        self.ball.yAcc = 0.0
        self.ball.pushedRight = False
        self.ball.pushedLeft = False
        self.ball.pushedY = False
        self.ball.isPushable = False
        for diamond in self.diamonds:
            diamond.exists = True
        self.ball.x, self.ball.y = (ballX, ballY)   ## Initial ball coordinates

class Setup7(Setup):
    '''
    One platform middle, one diamond other side.
    '''

    def __init__(self, surface):
        Setup.__init__(self, surface)

        platformW = int(params.WIDTH * 0.20)
        platformH = int(params.HEIGHT * 0.30)
        platformX = int(params.WIDTH * 0.5 - platformW * 0.5)
        platformY = int(params.HEIGHT - platformH)
        platform = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))

        self.diamondX =     platformX + platformW * 0.5      ## Initial diamond x
        self.diamondY =     platformY - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond1 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        self.diamondX =     params.WIDTH * 0.93     ## Initial diamond x
        self.diamondY =     params.HEIGHT - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond2 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        obstacles = [platform]

        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.2)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        dotAngle = 0.0
        ball = objects.Ball(surface=self.surface, color=objects.BALL_COLOR, x=ballX, y=ballY, radius=objects.BALL_RADIUS, dotAngle=dotAngle, obstacles=obstacles)

        diamonds = [diamond1, diamond2]

        self.obstacles, self.ball, self.diamonds = (obstacles, ball, diamonds)

        self.maxCollectTime = 5  ## Ideal max completion time in seconds to collect all diamond

    def refresh(self):
        time.sleep(0.5)
        ballX = random.randint(params.WIDTH * 0.1, params.WIDTH * 0.2)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        self.ball.angleSpeed = 0.0
        self.ball.angleAcc   = 0.0
        self.ball.v_x = 0.0
        self.ball.v_y = 0.0
        self.ball.yAcc = 0.0
        self.ball.pushedRight = False
        self.ball.pushedLeft = False
        self.ball.pushedY = False
        self.ball.isPushable = False
        for diamond in self.diamonds:
            diamond.exists = True
        self.ball.x, self.ball.y = (ballX, ballY)   ## Initial ball coordinates

class Setup8(Setup):
    '''
    Two platform left and right, two diamonds at sides.
    '''

    def __init__(self, surface):
        Setup.__init__(self, surface)

        platformW = int(params.WIDTH * 0.10)
        platformH = int(params.HEIGHT * 0.30)
        platformX = int(params.WIDTH * 0.3 - platformW * 0.5)
        platformY = int(params.HEIGHT - platformH)
        platform1 = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))

        self.diamondX =     platformX - objects.DIAMOND_RADIUS      ## Initial diamond x
        self.diamondY =     params.HEIGHT - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond1 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        platformW = int(params.WIDTH * 0.10)
        platformH = int(params.HEIGHT * 0.30)
        platformX = int(params.WIDTH * 0.7 - platformW * 0.5)
        platformY = int(params.HEIGHT - platformH)
        platform2 = objects.Obstacle(surface=self.surface, color=objects.OBSTACLE_COLOR, rect=(platformX, platformY, platformW, platformH))

        self.diamondX =     platformX + platformW + objects.DIAMOND_RADIUS      ## Initial diamond x
        self.diamondY =     params.HEIGHT - objects.DIAMOND_RADIUS ## Initial diamond y
        diamond2 = objects.Diamond(surface=self.surface, color=objects.DIAMOND_COLOR, x=self.diamondX, y=self.diamondY, radius=objects.DIAMOND_RADIUS)

        obstacles = [platform1, platform2]

        ballX = random.randint(params.WIDTH * 0.5, params.WIDTH * 0.55)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        dotAngle = 0.0
        ball = objects.Ball(surface=self.surface, color=objects.BALL_COLOR, x=ballX, y=ballY, radius=objects.BALL_RADIUS, dotAngle=dotAngle, obstacles=obstacles)

        diamonds = [diamond1, diamond2]

        self.obstacles, self.ball, self.diamonds = (obstacles, ball, diamonds)

        self.maxCollectTime = 9  ## Ideal max completion time in seconds to collect all diamond

    def refresh(self):
        time.sleep(0.5)
        ballX = random.randint(params.WIDTH * 0.5, params.WIDTH * 0.55)
        ballY = params.HEIGHT * 0.95 - objects.BALL_RADIUS
        self.ball.angleSpeed = 0.0
        self.ball.angleAcc   = 0.0
        self.ball.v_x = 0.0
        self.ball.v_y = 0.0
        self.ball.yAcc = 0.0
        self.ball.pushedRight = False
        self.ball.pushedLeft = False
        self.ball.pushedY = False
        self.ball.isPushable = False
        for diamond in self.diamonds:
            diamond.exists = True
        self.ball.x, self.ball.y = (ballX, ballY)   ## Initial ball coordinates

setups = [Setup0, Setup1, Setup2, Setup3, Setup4, Setup5, Setup6, Setup7, Setup8]
