
from Detection import Detection

import pygame as pg
import pygame.image
import cv2

def listToString(l):
    out=""
    for e in l:
        out+=str(e)+" "
    return out

pg.init()

#MODEL
detection = Detection('best_model.pt')

#WIDGETS
screen = pg.display.set_mode((1000,800))
screen.fill((153,255,255))
canvas = pygame.Surface([800,600])
canvas.fill((0,0,0))
w, h = screen.get_size()
canva_x,canva_y = w/2 - canvas.get_size()[0]/2 , h/2 - canvas.get_size()[1]/2
font = pygame.font.Font('freesansbold.ttf', 25)
text = font.render('Draw a single-line equation (e.g. "4-3")', True, (0,0,0))
textRect = text.get_rect()
textRect.center = ( w//2, 30)

#MAIN LOOP
loop = 1
i=0
while loop:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            loop=0
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_s:
                pg.image.save(canvas, "image.png")
                canvas.fill((0,0,0))
                frame = cv2.imread("image.png")
                frame = cv2.resize(frame, (640,640))
                calcul = detection.predict(frame)
                result = detection.resolve(calcul)
                print(calcul, "=", result)

                text = font.render(listToString(calcul) + " = " + str(result), True, (0, 0, 0))
                textRect = text.get_rect()
                textRect.center = (w // 2, 30)

            if event.key == pg.K_a:
                pg.image.save(canvas, "img/op/op_" + str(i) +".png")
                canvas.fill((0,0,0))
                i+=1

    screen.fill((153,255,255))
    screen.blit(canvas, [canva_x,canva_y] )
    screen.blit(text, textRect)

    if pygame.mouse.get_pressed()[0]:
        mx, my = pygame.mouse.get_pos()
        dx = mx - canva_x
        dy = my - canva_y

        pygame.draw.circle(canvas, (255,255,255),[dx,dy],10)


    pygame.display.flip()

pg.quit()