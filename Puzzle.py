import pygame
import sys

# Dimensiones del tablero
ANCHO, ALTO = 400, 400
FILAS, COLUMNAS = 3, 3
TAMANO_CASILLA = ANCHO // COLUMNAS

# Dimensiones de la ventana de movimientos
ANCHO_VENTANA_MOVIMIENTOS, ALTO_VENTANA_MOVIMIENTOS = 200, ALTO
POS_VENTANA_MOVIMIENTOS = (ANCHO, 0)

# Colores
NEGRO = (0, 0, 0)
BLANCO = (255, 255, 255)

# Inicialización de Pygame
pygame.init()
pantalla = pygame.display.set_mode((ANCHO + ANCHO_VENTANA_MOVIMIENTOS, ALTO))
pygame.display.set_caption('Puzzle Deslizante')

# Ventana para mostrar movimientos
ventana_movimientos = pygame.Surface((ANCHO_VENTANA_MOVIMIENTOS, ALTO_VENTANA_MOVIMIENTOS))

# Números en el rompecabezas (disposición inicial)
numeros = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]

# Movimientos realizados
movimientos = []

def dibujar_tablero():
    for fila in range(FILAS):
        for columna in range(COLUMNAS):
            numero = numeros[fila][columna]
            if numero == 0:
                color = NEGRO
            else:
                color = BLANCO
            pygame.draw.rect(pantalla, color, (columna * TAMANO_CASILLA, fila * TAMANO_CASILLA, TAMANO_CASILLA, TAMANO_CASILLA))
            if numero != 0:
                fuente = pygame.font.Font(None, 36)
                texto = fuente.render(str(numero), True, NEGRO)
                rectangulo_texto = texto.get_rect(center=(columna * TAMANO_CASILLA + TAMANO_CASILLA // 2, fila * TAMANO_CASILLA + TAMANO_CASILLA // 2))
                pantalla.blit(texto, rectangulo_texto)

def deslizar(fila_pieza, columna_pieza):
    # Comprueba si el espacio en blanco está adyacente a la pieza y, en caso afirmativo, realiza el deslizamiento
    fila_blanco, columna_blanco = encontrar_blanco()
    if (fila_pieza == fila_blanco and abs(columna_pieza - columna_blanco) == 1) or (columna_pieza == columna_blanco and abs(fila_pieza - fila_blanco) == 1):
        numeros[fila_blanco][columna_blanco], numeros[fila_pieza][columna_pieza] = numeros[fila_pieza][columna_pieza], numeros[fila_blanco][columna_blanco]
        movimientos.append((fila_pieza, columna_pieza))

def encontrar_blanco():
    # Encuentra la posición del espacio en blanco en el tablero
    for fila in range(FILAS):
        for columna in range(COLUMNAS):
            if numeros[fila][columna] == 0:
                return fila, columna

while True:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if evento.type == pygame.MOUSEBUTTONDOWN:
            click_x, click_y = pygame.mouse.get_pos()
            columna_pieza = click_x // TAMANO_CASILLA
            fila_pieza = click_y // TAMANO_CASILLA
            deslizar(fila_pieza, columna_pieza)

    pantalla.fill(NEGRO)
    dibujar_tablero()
    
    # Mostrar los movimientos realizados en la ventana de movimientos
    ventana_movimientos.fill(NEGRO)
    fuente = pygame.font.Font(None, 20)
    for i, movimiento in enumerate(movimientos):
        texto = fuente.render(f"Movimiento {i + 1}: ({movimiento[0]}, {movimiento[1]})", True, BLANCO)
        rectangulo_texto = texto.get_rect(topleft=(10, i * 20))
        ventana_movimientos.blit(texto, rectangulo_texto)
    
    # Copia la ventana de movimientos a la pantalla principal
    pantalla.blit(ventana_movimientos, POS_VENTANA_MOVIMIENTOS)
    
    pygame.display.update()