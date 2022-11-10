import pickle

import matplotlib.pyplot as plt
import pygame as pg
import pygame.image
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage.transform import resize

#Chargement du modele
global model
model = pickle.load(open('Neural_Network_MNIST.sav', 'rb'))

#fonction de calcul de la prediction
def predict():
    img = imread('image.png','rb')
    im = resize(img, (28,28))
    plt.figure(1)
    plt.imshow(im)
    plt.show()
    x=np.zeros((784,1))
    x = x.reshape(x.shape[0],)
    cpt=0
    for i in range(28):
        for j in range(28):
            x[cpt] = (im[i][j][0] + im[i][j][1] + im[i][j][2])/3
            cpt += 1
    return model.predict([x])


pg.init()

#WIDGETS
screen = pg.display.set_mode((400,400))
screen.fill((153,255,255))
canvas = pygame.Surface([280,280])
canvas.fill((0,0,0))
w, h = screen.get_size()
canva_x,canva_y = w/2 - canvas.get_size()[0]/2 , h/2 - canvas.get_size()[1]/2
font = pygame.font.Font('freesansbold.ttf', 25)
text = font.render('Draw a number between 0-9', True, (0,0,0))
textRect = text.get_rect()
textRect.center= ( w//2, 30)

#MAIN LOOP
loop = 1
while loop:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            loop=0
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_s:
                pg.image.save(canvas, "image.png")
                print(predict())
                canvas.fill((0,0,0))

    screen.blit(canvas, [canva_x,canva_y] )
    screen.blit(text, textRect)

    if pygame.mouse.get_pressed()[0]:
        mx, my = pygame.mouse.get_pos()
        dx = mx - canva_x
        dy = my - canva_y

        pygame.draw.circle(canvas, (255,255,255),[dx,dy],10)


    pygame.display.flip()

pg.quit()