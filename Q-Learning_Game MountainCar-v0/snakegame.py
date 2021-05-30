# Import Libraries
import pygame
import time
import random
import pandas as pd
from os import path

# Const Variable
WHITE = (255, 255, 255)
YELLOW = (255, 255, 102)
BLACK = (0, 0, 0)
RED = (255,0,0)
GREEN = (0, 255, 0)
BLUE = (0,0,255)
SNAKESIZE = [20,20]

class Link():
    def __init__(self):
        self.spriteSnake = "spire/snake.png"
        self.spriteFood = "spire/apple.png"
        self.backgroundPlaying = "spire/grass.png"
        self.backgroundLose = "spire/gameover.jpg"
        self.backgroundGoodbye = "spire/goodbye.jpg"
        self.filesave = "output/result.csv"

class Snake():
    def __init__(self,screenSize):
        self.headColor = RED
        self.bodyColor = GREEN
        self.borderColor = WHITE
        self.size = [20,20]
        self.length = 1
        self.screenSize = screenSize
        self.position = [[screenSize[0]/2,screenSize[1]/2]]
        self.prePosition = self.position
        self.direct="left"
        link = Link().spriteSnake

        self.origSprite = pygame.image.load(link)
        self.origSprite = pygame.transform.smoothscale(self.origSprite, (self.size[0], self.size[1]))
        self.sprite = self.origSprite
        self.spriteSize = self.sprite.get_rect().size
        self.rotation = 0
    
    def predictPosition(self,direct): # Predict Move of Snake. Use later
        change_position = [0,0]
        temp = self.position[:]
        predict = temp[:]
        if direct == "left" :
            change_position[0] = -self.size [0]
        elif direct == "right":
            change_position[0] = self.size [0]
        elif direct == "up":
            change_position[1] = -self.size [1]
        elif direct == "down" :
            change_position[1] = self.size [1]
        
        if change_position != [0,0]: 
            predict[0] = [sum(x) for x in zip(temp[0], change_position)]
            predict[1:] = temp[:-1]
        return predict[:]

    def predictLoseMove(self,direct):
        isLose = False
        position = self.predictPosition(direct)[:]
        if position[0][0] >= self.screenSize[0] or position[0][0] < 0 or position[0][1] >= self.screenSize[1] or position[0][1] < 0:
            isLose = True
        elif (position[0]) in (position[1:]):
                    isLose = True
        return isLose
    
    def updatePosition(self,direct):
        predict = self.predictPosition(direct)
        self.direct = direct
        self.prePosition = self.position[:]
        self.position = predict[:]

    def updateLengthSnake(self,foodPosition):
        head = [foodPosition[:]]
        body = self.prePosition
        head.extend(body)
        self.position = head[:]
        self.length = len(self.position)        

    def drawHead(self,screen):
        temp = self.position[0]+self.size
        #self.lineStart = [self.position[0][0]+self.size[0]//2,self.position[0][1]+self.size[1]//2]
        # Draw a rectangle with rounded corners
        if self.direct == "left" or self.direct == "" :
            pygame.draw.rect(screen, self.headColor, temp, 0, border_radius=self.size[0]//2, border_top_right_radius=0,
                            border_bottom_right_radius=0)
            pygame.draw.rect(screen, self.borderColor, temp, 1, border_radius=self.size[0]//2, border_top_right_radius=0,
                            border_bottom_right_radius=0)
            #self.lineEnd = [self.lineStart[0]-self.size[0],self.lineStart[1]]
        elif self.direct == "right" :
            pygame.draw.rect(screen, self.headColor, temp, 0, border_radius=self.size[0]//2, border_top_left_radius=0,
                            border_bottom_left_radius=0)
            pygame.draw.rect(screen, self.borderColor, temp, 1, border_radius=self.size[0]//2, border_top_left_radius=0,
                            border_bottom_left_radius=0)
            #self.lineEnd = [self.lineStart[0]+self.size[0],self.lineStart[1]]
        elif self.direct == "down" :
            pygame.draw.rect(screen, self.headColor, temp, 0, border_radius=self.size[0]//2, border_top_left_radius=0,
                            border_top_right_radius=0)
            pygame.draw.rect(screen, self.borderColor, temp, 1, border_radius=self.size[0]//2, border_top_left_radius=0,
                            border_top_right_radius=0)
            #self.lineEnd = [self.lineStart[0],self.lineStart[1]+self.size[1]]
        elif self.direct == "up" :
            pygame.draw.rect(screen, self.headColor, temp, 0, border_radius=self.size[0]//2, border_bottom_left_radius=0,
                            border_bottom_right_radius=0)
            pygame.draw.rect(screen, self.borderColor, temp, 1, border_radius=self.size[0]//2, border_bottom_left_radius=0,
                            border_bottom_right_radius=0)
            #self.lineEnd = [self.lineStart[0],self.lineStart[1]-self.size[1]]
        #pygame.draw.aaline(screen, GREEN, self.lineStart,self.lineEnd, True)

    def drawTail(self,screen):
        temp = self.position[-1]+self.size
        # Draw a rectangle with rounded corners
        if self.position[-1][0]<self.position[-2][0] and self.position[-1][1]==self.position[-2][1]:
            pygame.draw.rect(screen, self.bodyColor, temp, 0, border_radius=self.size[0]//2, border_top_right_radius=0,
                            border_bottom_right_radius=0)
            pygame.draw.rect(screen, self.borderColor, temp, 1, border_radius=self.size[0]//2, border_top_right_radius=0,
                            border_bottom_right_radius=0)
        elif self.position[-1][0]>self.position[-2][0] and self.position[-1][1]==self.position[-2][1] :
            pygame.draw.rect(screen, self.bodyColor, temp, 0, border_radius=self.size[0]//2, border_top_left_radius=0,
                            border_bottom_left_radius=0)
            pygame.draw.rect(screen, self.borderColor, temp, 1, border_radius=self.size[0]//2, border_top_left_radius=0,
                            border_bottom_left_radius=0)
        elif self.position[-1][0]==self.position[-2][0] and self.position[-1][1]>self.position[-2][1] :
            pygame.draw.rect(screen, self.bodyColor, temp, 0, border_radius=self.size[0]//2, border_top_left_radius=0,
                            border_top_right_radius=0)
            pygame.draw.rect(screen, self.borderColor, temp, 1, border_radius=self.size[0]//2, border_top_left_radius=0,
                            border_top_right_radius=0)
        elif self.position[-1][0]==self.position[-2][0] and self.position[-1][1]<self.position[-2][1] :
            pygame.draw.rect(screen, self.bodyColor, temp, 0, border_radius=self.size[0]//2, border_bottom_left_radius=0,
                            border_bottom_right_radius=0)
            pygame.draw.rect(screen, self.borderColor, temp, 1, border_radius=self.size[0]//2, border_bottom_left_radius=0,
                            border_bottom_right_radius=0)

    def drawBody(self,screen,index):
        temp = self.position[index]+self.size
        x = self.position[index-1][:]
        y = self.position[index][:]
        z = self.position[index+1][:]
        idx = list()
        idx.append([y[0],y[1]-self.size[1]]) #up
        idx.append([y[0],y[1]+self.size[1]]) #down
        idx.append([y[0]-self.size[0],y[1]]) #left
        idx.append([y[0]+self.size[0],y[1]]) #right
        check = [True,True,True,True]

        for i,j in enumerate(idx) :
            if x == j or z == j:
                check[i]=False
        
        # Draw a rectangle with rounded corners
        if check[0]==check[1] or check[2]==check[3]:
            pygame.draw.rect(screen,self.bodyColor,temp)
            pygame.draw.rect(screen,self.borderColor,temp,width=1)
        elif check[0]:
            if check[2] :
                pygame.draw.rect(screen, self.bodyColor, temp, 0, border_top_left_radius=self.size[0]//2)
                pygame.draw.rect(screen, self.borderColor, temp, 1, border_top_left_radius=self.size[0]//2)
            elif check[3] :
                pygame.draw.rect(screen, self.bodyColor, temp, 0, border_top_right_radius=self.size[0]//2)
                pygame.draw.rect(screen, self.borderColor, temp, 1, border_top_right_radius=self.size[0]//2)
        elif check[1]:
            if check[2] :
                pygame.draw.rect(screen, self.bodyColor, temp, 0,border_bottom_left_radius=self.size[0]//2)
                pygame.draw.rect(screen, self.borderColor, temp, 1,border_bottom_left_radius=self.size[0]//2)
            elif check[3] :
                pygame.draw.rect(screen, self.bodyColor, temp, 0,border_bottom_right_radius=self.size[0]//2)
                pygame.draw.rect(screen, self.borderColor, temp, 1,border_bottom_right_radius=self.size[0]//2)

class Food():
    def __init__(self,screenSize,sizePlayable):
        self.color = WHITE
        self.size = [20,20]
        self.sizePlayable = sizePlayable
        self.screenSize = screenSize
        self.addFood()
        link = Link().spriteFood

        self.origSprite = pygame.image.load(link).convert_alpha()
        self.origSprite = pygame.transform.smoothscale(self.origSprite, (self.size[0], self.size[1]))
        self.sprite = self.origSprite
        self.spriteSize = self.sprite.get_rect().size
    
    def addFood(self):
        foodx = round(random.randrange(self.sizePlayable[0], self.screenSize[0] - self.size [0]) / self.size[0]) * self.size[0]
        foody = round(random.randrange(self.sizePlayable[1], self.screenSize[1] - self.size [1]) / self.size[1]) * self.size[1]
        self.position = [foodx, foody]

class FileSave():
    def __init__(self):
        self.fileName = Link().filesave
        self.highestScore = 0
        if path.exists(self.fileName) :
            with open(self.fileName,'r') as f:
                try :
                    df = pd.read_csv(f, usecols= ['Score'])
                    lst = df.values.tolist()
                    self.highestScore = max(lst)[0]
                except:
                    print("No column")
                f.close()

    def saveFileData(self,score):
        #df = pd.DataFrame([[list_direction,list_food,score]], columns = ['Direction', 'Food Position', 'Score'])
        df = pd.DataFrame([[score]], columns = ['Score'])
        if not path.exists(self.fileName) :
            with open(self.fileName,'w', newline='') as f:
                df.to_csv(f, index=False, encoding='utf-8',header=True)
                f.close()
        else:
            with open(self.fileName, 'a+', newline='') as f:
                df.to_csv(f, index=False, encoding='utf-8', mode='a',header=False)
                f.close()

class Display():
    def __init__(self,scale,mode="normal"):
        self.mode = mode
        self.link = Link()
        # Screen
        self.color = BLACK
        self.scale = scale
        self.size = (SNAKESIZE[0]*self.scale[0],SNAKESIZE[1]*self.scale[1])
        self.fps = 60
        self.caption = 'Snake game by TXDien'
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(self.caption)

        # Font
        self.fontSizeScore = round(self.size[0]/15)
        self.fontSize =  round(self.size[0]/13)
        self.fontStyle = pygame.font.SysFont("bahnschrift", self.fontSize)
        self.fontStyleScore = pygame.font.SysFont("comicsansms", self.fontSizeScore )
        value = self.fontStyleScore .render("Your Score: " + str(0), True, YELLOW)
        self.sizePlayable = [value.get_width(),value.get_height()]

        self.displayBackground()

    def displayScore(self,score):
        # Display Score
        value = self.fontStyleScore.render("Your Score: " + str(score), True, YELLOW)
        self.screen.blit(value, [0, 0])

    def displayFood(self,food):
        # Display Food
        # temp_food = food.position+food.size
        radius = food.size[0]/2
        center = [food.position[0]+radius,food.position[1]+radius]
        if self.mode == "normal":
            # pygame.draw.rect(self.screen,food.color,temp_food)
            pygame.draw.circle(self.screen,food.color,center,radius)
        elif self.mode == "beautiful":
            self.screen.blit(food.sprite, (food.position[0], food.position[1]))

    def displaySnake(self,snake):
        # Display Snake
        snake.drawHead(self.screen)
        for i in range(1,snake.length-1):
            snake.drawBody(self.screen,i)
            '''
            temp_snake = snake.position[i]+snake.size
            temp_color = snake.headColor if i==0 else snake.bodyColor
            temp_color_border = WHITE
            pygame.draw.rect(self.screen,temp_color,temp_snake)
            pygame.draw.rect(self.screen,temp_color_border,temp_snake,width=1)
            '''
        if snake.length>1:
            snake.drawTail(self.screen)

    def displayBackground(self,screenName="playing"):
        self.screen.fill(BLACK)
        if screenName == 'playing':
            color = self.color
            file = self.link.backgroundPlaying
        elif screenName == 'goodbye':
            color = self.color
            file = self.link.backgroundGoodbye
        elif screenName == 'gameover':
            color = BLUE
            file = self.link.backgroundLose
        if self.mode == "normal" :
            self.screen.fill(color)
        elif self.mode == "beautiful" :
            screenbackground = pygame.image.load(file)
            screenbackground = pygame.transform.smoothscale(screenbackground.convert_alpha(), (self.size[0], self.size[1]))
            self.screen.blit(screenbackground, (0,0))
            
    def displayElement(self,snake,food,score):
        # Erase Screen
        self.displayBackground()
        self.displayScore(score)
        self.displayFood(food)
        self.displaySnake(snake)
        pygame.display.update()

    def displayMessage(self,screenName,color,*args):
        self.displayBackground(screenName)
        numberOfText = len(args)
        heightOfText = 0
        for index, text in enumerate(args) :
            mesg = self.fontStyle.render(text, True, color)
            x = (self.size[0]-mesg.get_width())/2
            if index==0 :
                heightOfText = mesg.get_height()
                y = self.size[1]/2 - heightOfText*(numberOfText//2)
                if numberOfText%2 > 0 :
                    y = y - heightOfText/2
            else:
                y = y + heightOfText
            textPosition = [x,y]
            self.screen.blit(mesg,textPosition)
        pygame.display.update()

class Enviroment():
    def __init__(self,scale,mode = "normal"):
        self.mode = mode
        # Initalizing Screen for Game
        pygame.init()
        self.clock = pygame.time.Clock()
        # Screen
        self.screen = Display(scale,self.mode)
        # Load from file
        self.fileName = FileSave()
        self.highestScore = self.fileName.highestScore
        # Setting new game
        self.reset()

    def direction(self,keyPressed):
        direct = ""
        if keyPressed == pygame.K_LEFT:
            direct = "left"
        elif keyPressed == pygame.K_RIGHT:
            direct = "right"
        elif keyPressed == pygame.K_UP:
            direct = "up"
        elif keyPressed == pygame.K_DOWN:
            direct = "down"
        return direct

    def updateScore(self):
        self.score += 1
        self.snake.updateLengthSnake(self.food.position)
        while self.food.position in self.snake.position:
            self.food.addFood()

    def checkLose(self):
        isLose = False
        if self.snake.position[0][0] >= self.screen.size[0] or self.snake.position[0][0] < 0 or self.snake.position[0][1] >= self.screen.size[1] or self.snake.position[0][1] < 0:
            isLose = True
            print(f'out of windown: {self.snake.position[0]}')
        elif (self.snake.position[0]) in (self.snake.position[1:]):
            isLose = True
            print(f'Oop eat yourself: {self.snake.position}')
        self.isLose = isLose
        return isLose

    def askToQuit(self):
        self.highestScore = max(self.score, self.highestScore)
        color = YELLOW if (self.mode =="normal") else BLACK
        self.screen.displayMessage('gameover',color,"You Lost!",f"Your score: {self.score}",f"Hightest Score: {self.highestScore}","Press C-Play Again or Q-Quit")
        
        right_key = False
        isOver = False
        while not right_key :
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    isOver = True
                    right_key = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        isOver = True
                        right_key = True
                    elif event.key == pygame.K_c:
                        right_key = True
                        self.isNewgame= True
                        self.fileName.saveFileData(self.score)
        return isOver
        
    def reset(self):
        self.isNewgame = False # it already new game
        self.isLose = False
        self.food = Food(self.screen.size,self.screen.sizePlayable)
        self.snake = Snake(self.screen.size)
        self.score = 0
        self.screen.displayElement(self.snake,self.food,self.score)

    def playGame(self):
        isOver = False
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                isOver = True
            elif event.type == pygame.KEYDOWN:
                direct = self.direction(event.key)
                self.snake.updatePosition(direct)
                if self.snake.position[0] == self.food.position :
                    self.updateScore()
                self.screen.displayElement(self.snake,self.food,self.score)
                if self.checkLose() :
                    isOver = self.askToQuit()
                self.clock.tick(self.screen.fps)
        if isOver :
            self.fileName.saveFileData(self.score)
            color = GREEN if (self.mode == "normal") else BLACK
            self.screen.displayMessage('goodbye',GREEN, "See you again")
            pygame.display.update()
            time.sleep(2)
            pygame.quit()
        elif self.isNewgame :
            self.reset()
        return isOver

if __name__ == "__main__":
    env = Enviroment([20,20])
    isOver = False
    while not isOver:
        isOver = env.playGame()
