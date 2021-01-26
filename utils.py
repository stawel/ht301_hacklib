import cv2




def drawTemperature(img, point, T, color = (0,0,0)):
    d1, d2 = 2, 5
    dsize = 1
    font = cv2.FONT_HERSHEY_PLAIN
    (x, y) = point
    t = '%.2fC' % T
    cv2.line(img,(x+d1, y),(x+d2,y),color, dsize)
    cv2.line(img,(x-d1, y),(x-d2,y),color, dsize)
    cv2.line(img,(x, y+d1),(x,y+d2),color, dsize)
    cv2.line(img,(x, y-d1),(x,y-d2),color, dsize)

    text_size = cv2.getTextSize(t, font, 1, dsize)[0]
    tx, ty = x+d1, y+d1+text_size[1]
    if tx + text_size[0] > img.shape[1]: tx = x-d1-text_size[0]
    if ty                > img.shape[0]: ty = y-d1

    cv2.putText(img, t, (tx,ty), font, 1, color, dsize, cv2.LINE_8)