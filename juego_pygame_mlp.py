import os
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
# Usamos ExtraTrees porque es mucho más preciso para memorizar y sobreajustar (overfitting) 
# intencionalmente combinaciones exactas de datos que un MLP tradicional.
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

# Opcional: para graficar los datos en 2D y 3D
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt

plt.ion()

# Ventana base y factor de escala
BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1

# Acciones del modelo (multiclase)
ACCION_NADA    = 0
ACCION_SALTO   = 1
ACCION_AGACHAR = 2

# Tipo de bala — 3 alturas
BALA_SUELO      = 0   
BALA_MEDIA_BAJA = 1   
BALA_ALTA       = 2   

BALA_BAJA = BALA_SUELO  

@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    tipo_bala: int
    # Añadimos memoria al dataset: las acciones previas del jugador
    accion_t_minus_1: int
    accion_t_minus_2: int
    accion: int       


class Juego:
    def __init__(self) -> None:
        pygame.init()

        self._flags = 0
        self._fullscreen = False

        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Imitador Secuencial usando Machine Learning")

        # Colores
        self.BLANCO  = (255, 255, 255)
        self.NEGRO   = (0,   0,   0)
        self.GRIS    = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)
        self.CYAN    = (100, 220, 255)

        # Estado global
        self.corriendo = True
        self.modo_auto = False

        # Datos / modelo
        self.datos_modelo: List[Sample] = []
        self.modelo: Optional[ExtraTreesClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.modelo_entrenado = False
        self.ultima_proba: Optional[list] = None   

        # --- MEMORIA A CORTO PLAZO (Para el comportamiento del jugador) ---
        self.historial_acciones = [ACCION_NADA, ACCION_NADA]

        # Geometría / física 
        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size       = (32, 48)
        self.player_size_agach = (32, 24)   
        self.bullet_size = (16, 16)
        self.ship_size   = (64, 64)
        self.fondo_speed = 3

        # Estado de salto
        self.salto = False
        self.en_suelo = True
        self.salto_vel_inicial = 15.0
        self.gravedad  = 1.0
        self.salto_vel = self.salto_vel_inicial

        # Estado de agacharse
        self.agachado = False
        self.agachado_timer = 0
        self.AGACHADO_FRAMES = 25   

        # Animación del personaje
        self.current_frame = 0
        self.frame_speed   = 10
        self.frame_count   = 0

        # Animación del agachado
        self.agach_frame       = 0
        self.agach_frame_count = 0
        self.agach_frame_speed = 4

        # Bala
        self.velocidad_bala = -12
        self.bala_disparada = False
        self.tipo_bala_actual = BALA_SUELO

        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)
        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin       = int(50 * self.scale)
        ground_offset     = int(100 * self.scale)
        self.ground_y     = self.h - ground_offset

        self.player_size       = (int(32 * self.scale), int(48 * self.scale))
        self.player_size_agach = (int(32 * self.scale), int(24 * self.scale))
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size   = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 12 * self.scale
        self.gravedad  = 1  * self.scale
        self.salto_vel = self.salto_vel_inicial

        self.fuente       = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(self.margin, self.ground_y, self.player_size[0], self.player_size[1])
            self.bala = pygame.Rect(self.w - self.margin, self.ground_y + int(10 * self.scale), self.bullet_size[0], self.bullet_size[1])
            self.nave = pygame.Rect(self.w - int(100 * self.scale), self.ground_y, self.ship_size[0], self.ship_size[1])

    def _cargar_assets(self) -> None:
        def safe_load(path: str, size: Tuple[int, int], fallback_color=(200, 200, 200, 255)) -> pygame.Surface:
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill(fallback_color)
                return surf

        base = os.path.dirname(__file__)
        self.jugador_frames = [
            safe_load(os.path.join(base, "assets/sprites/sonic1.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/sonic2.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/sonic3.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/sonic4.png"), self.player_size),
        ]
        self.jugador_frames_agach = [
            safe_load(os.path.join(base, "assets/sprites/spin1.png"), self.player_size_agach),
            safe_load(os.path.join(base, "assets/sprites/spin2.png"), self.player_size_agach),
            safe_load(os.path.join(base, "assets/sprites/spin3.png"), self.player_size_agach),
            safe_load(os.path.join(base, "assets/sprites/spin4.png"), self.player_size_agach),
        ]
        self.bala_imgs = [
            safe_load(os.path.join(base, "assets/sprites/purple_ball.png"), self.bullet_size, (100, 180, 255, 255)),
            safe_load(os.path.join(base, "assets/sprites/purple_ball.png"), self.bullet_size, (160, 120, 255, 255)),
            safe_load(os.path.join(base, "assets/sprites/purple_ball.png"), self.bullet_size, (255,  80,  80, 255)),
        ]
        self.fondo_img = safe_load(os.path.join(base, "assets/game/fondo2.PNG"), (self.w, self.h), (40, 40, 40, 255))
        self.nave_img = safe_load(os.path.join(base, "assets/game/ufo.png"), self.ship_size, (140, 255, 200, 255))

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            self.pantalla = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
            self._apply_resolution(info.current_w, info.current_h, reset_positions=True)
        else:
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.jugador.width, self.jugador.height = self.player_size
        self.bala.x = self.w - self.margin
        self.bala_disparada = False
        self.salto    = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial
        self.agachado = False
        self.historial_acciones = [ACCION_NADA, ACCION_NADA] # Reset de memoria a corto plazo

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False

    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            self.bala_disparada = True
            self.tipo_bala_actual = random.choice([BALA_SUELO, BALA_MEDIA_BAJA, BALA_ALTA])

            ph = self.player_size[1]
            if self.tipo_bala_actual == BALA_SUELO:
                self.bala.y = self.ground_y + int(18 * self.scale)
            elif self.tipo_bala_actual == BALA_MEDIA_BAJA:
                self.bala.y = self.ground_y - int(ph * 0.25)
            else:
                self.bala.y = self.ground_y - int(ph * 0.75)
            self.bala.x = self.w - self.margin

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self) -> None:
        if self.en_suelo and not self.agachado:
            self.salto    = True
            self.en_suelo = False

    def manejar_salto(self) -> None:
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.salto     = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo  = True

    def iniciar_agache(self) -> None:
        if self.en_suelo and not self.salto and not self.agachado:
            self.agachado       = True
            self.jugador.height = self.player_size_agach[1]
            self.jugador.y      = self.ground_y + (self.player_size[1] - self.player_size_agach[1])

    def terminar_agache(self) -> None:
        if self.agachado:
            self.agachado       = False
            self.jugador.height = self.player_size[1]
            self.jugador.y      = self.ground_y

    # ----------------- MACHINE LEARNING SECUENCIAL -----------------
    def registrar_decision_manual(self) -> None:
        if not self.bala_disparada:
            return
        distancia = abs(self.jugador.x - self.bala.x)

        if self.agachado:
            accion_actual = ACCION_AGACHAR
        elif not self.en_suelo:
            accion_actual = ACCION_SALTO
        else:
            accion_actual = ACCION_NADA

        # Guardamos la muestra incluyendo el historial de lo que pasó hace 1 y 2 frames
        self.datos_modelo.append(
            Sample(
                velocidad_bala=float(self.velocidad_bala),
                distancia=float(distancia),
                tipo_bala=int(self.tipo_bala_actual),
                accion_t_minus_1=self.historial_acciones[0],
                accion_t_minus_2=self.historial_acciones[1],
                accion=accion_actual
            )
        )

        # Desplazamos la memoria temporal
        self.historial_acciones[1] = self.historial_acciones[0]
        self.historial_acciones[0] = accion_actual

    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 60:
            return False, "Necesitas más datos (>= 60). Juega un poco más en MANUAL."

        # Features enriquecidas con variables de tiempo/secuencia
        X = [[s.velocidad_bala, s.distancia, float(s.tipo_bala), float(s.accion_t_minus_1), float(s.accion_t_minus_2)] for s in samples]
        y = [s.accion for s in samples]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ExtraTreesClassifier es increíble para "memorizar" secuencias exactas sin perder
        # la capacidad de reaccionar si cambian ligeramente las velocidades.
        clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
        clf.fit(X_scaled, y)

        self._reset_modelo()
        self.scaler = scaler
        self.modelo = clf
        self.modelo_entrenado = True
        return True, f"Modelo de ML Secuencial entrenado con {len(samples)} estados."

    def decision_auto(self) -> int:
        if not self.modelo_entrenado or not self.bala_disparada or self.modelo is None or self.scaler is None:
            return ACCION_NADA

        distancia = abs(self.jugador.x - self.bala.x)

        # La IA predice basándose en el entorno + lo que ella misma hizo en los frames anteriores
        X = [[float(self.velocidad_bala), float(distancia), float(self.tipo_bala_actual), float(self.historial_acciones[0]), float(self.historial_acciones[1])]]
        Xs = self.scaler.transform(X)

        if hasattr(self.modelo, "predict_proba"):
            probas = self.modelo.predict_proba(Xs)[0]
            clases = list(self.modelo.classes_)
            proba_vec = [0.0, 0.0, 0.0]
            for i, c in enumerate(clases):
                if c < 3: proba_vec[c] = float(probas[i])
            self.ultima_proba = proba_vec
            accion = int(clases[probas.argmax()])
        else:
            accion = int(self.modelo.predict(Xs)[0])

        # Actualizar la memoria de la IA con su propia decisión de cara al siguiente frame
        self.historial_acciones[1] = self.historial_acciones[0]
        self.historial_acciones[0] = accion

        return accion

    # ----------------- menú y bucle principal -----------------
    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ - IA CON INGENIERÍA DE SECUENCIAS", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(40 * self.scale)))

        opciones = [
            "M - Manual (Juega presionando repetidamente ABAJO para crear el patrón)",
            "T - Entrenar IA (Aprenderá la secuencia temporal)",
            "A - Auto (La IA intentará imitar tu ritmo y repeticiones)",
            "F - Fullscreen (toggle)",
            "Q - Salir",
        ]
        x0, y = int(80 * self.scale), int(140 * self.scale)
        for op in opciones:
            t = self.fuente.render(op, True, self.BLANCO)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente.get_linesize() + 8

        info_txt = f"Datos en memoria: {len(self.datos_modelo)} | IA Lista: {'SÍ' if self.modelo_entrenado else 'NO'}"
        self.pantalla.blit(self.fuente_chica.render(info_txt, True, self.GRIS), (x0, y + 20))
        if msg:
            self.pantalla.blit(self.fuente_chica.render(msg, True, self.AMARILLO), (x0, y + 50))
        pygame.display.flip()

    def mostrar_menu(self) -> None:
        msg = ""
        esperando = True
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT: self.corriendo = False; esperando = False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos_modelo.clear()
                        self._reset_modelo()
                        self._reset_estado_juego()
                        esperando = False
                    if e.key == pygame.K_a:
                        if not self.modelo_entrenado: msg = "Primero necesitas entrenar el modelo (T)."
                        else: self.modo_auto = True; self._reset_estado_juego(); esperando = False
                    if e.key == pygame.K_t:
                        ok, info = self.train_result = self.entrenar_modelo()
                        msg = info
                    if e.key == pygame.K_f: self._toggle_fullscreen()
                    if e.key == pygame.K_q: self.corriendo = False; esperando = False

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: self.corriendo = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q: self.corriendo = False
                    elif e.key in (pygame.K_ESCAPE, pygame.K_p): self._reset_estado_juego(); self.mostrar_menu()
                    elif e.key == pygame.K_f: self._toggle_fullscreen()
                    elif not self.modo_auto:
                        if e.key == pygame.K_SPACE and self.en_suelo and not self.agachado: self.iniciar_salto()
                        elif e.key in (pygame.K_DOWN, pygame.K_s) and self.en_suelo and not self.salto: self.iniciar_agache()
                elif e.type == pygame.KEYUP and not self.modo_auto:
                    if e.key in (pygame.K_DOWN, pygame.K_s): self.terminar_agache()

            if not self.corriendo: break

            if self.modo_auto:
                accion = self.decision_auto()
                if accion == ACCION_SALTO: self.terminar_agache(); self.iniciar_salto()
                elif accion == ACCION_AGACHAR: self.iniciar_agache()
                else: self.terminar_agache()
            else:
                self.registrar_decision_manual()

            if self.salto: self.manejar_salto()
            if not self.bala_disparada: self.disparar_bala()

            # Renderizado básico de frames
            self.fondo_x1 -= self.fondo_speed; self.fondo_x2 -= self.fondo_speed
            if self.fondo_x1 <= -self.w: self.fondo_x1 = self.w
            if self.fondo_x2 <= -self.w: self.fondo_x2 = self.w
            self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0)); self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))
            
            if self.agachado: self.pantalla.blit(self.jugador_frames_agach[0], (self.jugador.x, self.jugador.y))
            else: self.pantalla.blit(self.jugador_frames[0], (self.jugador.x, self.jugador.y))
            
            self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))
            if self.bala_disparada: self.bala.x += self.velocidad_bala
            if self.bala.x < -self.bullet_size[0]: self.reset_bala()
            self.pantalla.blit(self.bala_imgs[self.tipo_bala_actual], (self.bala.x, self.bala.y))

            if self.jugador.colliderect(self.bala): self._reset_estado_juego()
            
            pygame.display.flip()
            reloj.tick(45)
        pygame.quit()

if __name__ == "__main__":
    Juego().loop()