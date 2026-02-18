import pygame
import cv2
import mediapipe as mp   #Version <- v0.0.9
import numpy as np
from collections import deque
import time
import json
import os
import random

class ImprovedHandGestureController:
    """Controlador de gestos mejorado con detección robusta y confirmación temporal."""
    
    def __init__(self, detection_confidence=0.6, tracking_confidence=0.6, smoothing_factor=0.4):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        self.position_buffer = deque(maxlen=8)
        self.smoothing_factor = smoothing_factor
        self.last_position = None
        self.movement_buffer = deque(maxlen=15)
        self.gesture_confirmation_buffer = deque(maxlen=5)
        self.gesture_confirmation_threshold = 3
        
        self.gesture_states = {
            'current_gesture': 'none', 'gesture_confidence': 0.0,
            'last_gesture': 'none', 'gesture_timer': 0
        }
        
        self.last_palm_position = None
        self.detection_stats = {
            'total_frames': 0, 'detected_frames': 0, 'detection_rate': 0.0,
            'consecutive_failures': 0, 'last_detection_time': time.time()
        }
        
        self.use_preprocessing = True
        self.enhance_contrast = True
        self.target_brightness = 128
        self.max_consecutive_failures = 30
        
    def preprocess_frame(self, frame):
        if not self.use_preprocessing:
            return frame
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        current_brightness = np.mean(gray)
        
        if current_brightness < 80:
            alpha = min(self.target_brightness / (current_brightness + 1e-6), 2.0)
            rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=alpha, beta=0)
        elif current_brightness > 180:
            alpha = max(self.target_brightness / (current_brightness + 1e-6), 0.5)
            rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=alpha, beta=0)
        
        if self.enhance_contrast:
            lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            rgb_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        rgb_frame = cv2.bilateralFilter(rgb_frame, 5, 50, 50)
        return rgb_frame
    
    def reinitialize_detector(self):
        print("🔄 Re-inicializando detector...")
        self.hands.close()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.detection_stats['consecutive_failures'] = 0
    
    def _confirm_gesture(self, gesture):
        self.gesture_confirmation_buffer.append(gesture)
        return self.gesture_confirmation_buffer.count(gesture) >= self.gesture_confirmation_threshold
    
    def calculate_hand_landmarks(self, frame):
        self.detection_stats['total_frames'] += 1
        rgb_frame = self.preprocess_frame(frame)
        results = self.hands.process(rgb_frame)
        
        landmarks_data = {
            'hands_detected': False, 'palm_center': None, 'static_gesture': 'none',
            'dynamic_gesture': 'none', 'gesture_confidence': 0.0,
            'movement_velocity': 0.0, 'movement_direction': None, 'detection_quality': 0.0
        }
        
        if results.multi_hand_landmarks:
            self.detection_stats['detected_frames'] += 1
            self.detection_stats['consecutive_failures'] = 0
            self.detection_stats['last_detection_time'] = time.time()
            landmarks_data['hands_detected'] = True
            
            hand_landmarks = results.multi_hand_landmarks[0]
            visibility_scores = [lm.visibility if hasattr(lm, 'visibility') else 1.0 
                               for lm in hand_landmarks.landmark]
            landmarks_data['detection_quality'] = np.mean(visibility_scores)
            
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            palm_center = np.mean(landmarks_array[[0, 5, 9, 13, 17]], axis=0)
            finger_extensions = self._analyze_finger_extensions(landmarks_array)
            
            static_gesture_raw = self._classify_static_gesture(finger_extensions, landmarks_array)
            if self._confirm_gesture(static_gesture_raw):
                landmarks_data['static_gesture'] = static_gesture_raw
            
            landmarks_data['palm_center'] = palm_center
            
            if self.last_palm_position is not None:
                movement = palm_center[:2] - self.last_palm_position[:2]
                velocity = np.linalg.norm(movement)
                self.movement_buffer.append((movement, velocity))
                
                if velocity > 0.025:
                    dynamic_gesture, confidence = self._detect_dynamic_gesture()
                    if dynamic_gesture != 'none' and confidence > 0.85:
                        landmarks_data['dynamic_gesture'] = dynamic_gesture
                        landmarks_data['gesture_confidence'] = confidence
                        landmarks_data['movement_velocity'] = velocity
            
            self.last_palm_position = palm_center
            
            self.mp_draw.draw_landmarks(
                rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            self.detection_stats['consecutive_failures'] += 1
            self.movement_buffer.clear()
            self.last_palm_position = None
            self.gesture_confirmation_buffer.clear()
            
            if self.detection_stats['consecutive_failures'] >= self.max_consecutive_failures:
                self.reinitialize_detector()
        
        if self.detection_stats['total_frames'] > 0:
            self.detection_stats['detection_rate'] = (
                self.detection_stats['detected_frames'] / self.detection_stats['total_frames']
            )
        
        return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), landmarks_data
    
    def _analyze_finger_extensions(self, landmarks):
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        finger_mcps = [2, 5, 9, 13, 17]
        
        extensions = {}
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, (tip, pip, mcp, name) in enumerate(zip(finger_tips, finger_pips, finger_mcps, finger_names)):
            tip_to_pip = np.linalg.norm(landmarks[tip][:2] - landmarks[pip][:2])
            pip_to_mcp = np.linalg.norm(landmarks[pip][:2] - landmarks[mcp][:2])
            
            if i == 0:
                wrist_x = landmarks[0][0]
                thumb_tip_x = landmarks[tip][0]
                thumb_mcp_x = landmarks[mcp][0]
                extensions[name] = abs(thumb_tip_x - wrist_x) > abs(thumb_mcp_x - wrist_x) * 1.3
            else:
                extensions[name] = tip_to_pip > pip_to_mcp * 0.8
            
        return extensions
    
    def _classify_static_gesture(self, finger_extensions, landmarks):
        extended_count = sum(finger_extensions.values())
        
        if extended_count == 0:
            return 'fist'
        elif extended_count == 5:
            return 'open_palm'
        elif extended_count == 1 and finger_extensions['index']:
            return 'point'
        elif extended_count == 2 and finger_extensions['index'] and finger_extensions['middle'] and not finger_extensions['thumb']:
            return 'peace'
        elif extended_count == 1 and finger_extensions['thumb']:
            return 'thumb_up'
        elif finger_extensions['thumb'] and finger_extensions['index']:
            thumb_tip = landmarks[4][:2]
            index_tip = landmarks[8][:2]
            distance = np.linalg.norm(thumb_tip - index_tip)
            if distance < 0.06:
                return 'pinch'
            elif extended_count == 2:
                return 'l_shape'
        return 'custom'
    
    def _detect_dynamic_gesture(self):
        if len(self.movement_buffer) < 10:
            return 'none', 0.0
        
        recent_movements = list(self.movement_buffer)[-10:]
        movements = np.array([m[0] for m in recent_movements])
        velocities = np.array([m[1] for m in recent_movements])
        
        total_movement = np.sum(movements, axis=0)
        avg_velocity = np.mean(velocities)
        
        if avg_velocity < 0.02:
            return 'none', 0.0
        
        angle = np.arctan2(total_movement[1], total_movement[0])
        magnitude = np.linalg.norm(total_movement)
        
        if magnitude > 0.15:
            if abs(angle) < np.pi / 4:
                return 'swipe_right', 0.95
            elif abs(angle) > 3 * np.pi / 4:
                return 'swipe_left', 0.95
            elif np.pi / 4 < angle < 3 * np.pi / 4:
                return 'swipe_down', 0.95
            elif -3 * np.pi / 4 < angle < -np.pi / 4:
                return 'swipe_up', 0.95
        
        if len(self.movement_buffer) >= 15:
            angles = [np.arctan2(movements[i][1], movements[i][0]) for i in range(len(movements) - 1)]
            if len(angles) > 5:
                angle_changes = np.diff(angles)
                angle_changes = np.where(angle_changes > np.pi, angle_changes - 2*np.pi, angle_changes)
                angle_changes = np.where(angle_changes < -np.pi, angle_changes + 2*np.pi, angle_changes)
                total_rotation = np.sum(angle_changes)
                if abs(total_rotation) > np.pi * 1.8:
                    return ('circle_ccw', 0.90) if total_rotation > 0 else ('circle_cw', 0.90)
        
        return 'none', 0.0
    
    def get_smoothed_position(self, raw_position, frame_width):
        if raw_position is None:
            return self.last_position if self.last_position else frame_width // 2
        
        screen_pos = raw_position[0] * frame_width
        self.position_buffer.append(screen_pos)
        
        if len(self.position_buffer) >= 3:
            weights = np.exp(np.linspace(-1, 0, len(self.position_buffer)))
            weights /= weights.sum()
            smoothed_pos = np.average(list(self.position_buffer), weights=weights)
        elif self.last_position is not None:
            smoothed_pos = self.smoothing_factor * screen_pos + (1 - self.smoothing_factor) * self.last_position
        else:
            smoothed_pos = screen_pos
        
        self.last_position = smoothed_pos
        return int(smoothed_pos)
    
    def get_detection_stats(self):
        return self.detection_stats.copy()


class RecordsManager:
    """Gestor de récords con persistencia JSON."""
    
    def __init__(self, filename="breakout_records.json"):
        self.filename = filename
        self.records = self._load_records()
    
    def _load_records(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except:
                return self._get_default_records()
        return self._get_default_records()
    
    def _get_default_records(self):
        return {
            'best_score': 0, 'best_time': float('inf'), 'best_score_time': float('inf'),
            'games_played': 0, 'total_time_played': 0.0, 'history': []
        }
    
    def _save_records(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.records, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error guardando récords: {e}")
    
    def update_records(self, score, game_time, victory=False):
        self.records['games_played'] += 1
        self.records['total_time_played'] += game_time
        
        new_record = False
        records_broken = []
        
        if score > self.records['best_score']:
            self.records['best_score'] = score
            self.records['best_score_time'] = game_time
            new_record = True
            records_broken.append('score')
        
        if victory and game_time < self.records['best_time']:
            self.records['best_time'] = game_time
            new_record = True
            records_broken.append('time')
        
        game_entry = {
            'score': score, 'time': game_time, 'victory': victory,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.records['history'].insert(0, game_entry)
        self.records['history'] = self.records['history'][:10]
        self._save_records()
        
        return new_record, records_broken
    
    def get_records(self):
        return self.records.copy()


class PowerUp:
    """Clase para power-ups que caen de los ladrillos."""
    
    def __init__(self, x, y, powerup_type):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 15
        self.speed = 3
        self.type = powerup_type
        self.active = True
        
        self.colors = {
            'multi_2': (0, 255, 255),
            'multi_5': (255, 0, 255),
        }
        self.color = self.colors.get(powerup_type, (255, 255, 255))
    
    def update(self):
        self.y += self.speed
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (255, 255, 255), (self.x, self.y, self.width, self.height), 2)
        
        font = pygame.font.Font(None, 20)
        if self.type == 'multi_2':
            text = font.render("x2", True, (0, 0, 0))
        else:
            text = font.render("x5", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.x + self.width//2, self.y + self.height//2))
        screen.blit(text, text_rect)


class Ball:
    """Clase para manejar múltiples bolas con física mejorada."""
    
    def __init__(self, x, y, speed_x, speed_y, radius=10):
        self.x = float(x)
        self.y = float(y)
        self.speed_x = float(speed_x)
        self.speed_y = float(speed_y)
        self.radius = radius
        self.color = (255, 100, 100)
        self.base_speed = 6
        self.launched = False
        self.active = True
        
        self.velocity_buffer = deque(maxlen=3)
    
    def update_velocity(self, new_vx, new_vy):
        """Actualiza velocidad directamente sin suavizado excesivo."""
        self.speed_x = new_vx
        self.speed_y = new_vy


class EnhancedBreakoutGame:
    """Juego Breakout con power-ups, turbo mejorado y puntuación por colores."""
    
    # Puntuación por color de ladrillo
    BLOCK_SCORES = {
        (255, 0, 0): 50,      # Rojo - más puntos (arriba)
        (255, 165, 0): 40,    # Naranja
        (255, 255, 0): 30,    # Amarillo
        (0, 255, 0): 20,      # Verde
        (0, 0, 255): 10       # Azul - menos puntos (abajo)
    }
    
    # Configuración del turbo
    TURBO_DURATION = 10.0      # 10 segundos de duración
    TURBO_COOLDOWN = 20.0      # 20 segundos de cooldown
    TURBO_MULTIPLIER = 2.5     # Multiplicador de velocidad
    
    def __init__(self, width=1000, height=800):
        pygame.init()
        
        self.WIDTH = width
        self.HEIGHT = height
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Breakout - Power-ups y Turbo Mejorado")
        self.clock = pygame.time.Clock()
        
        self.gesture_controller = ImprovedHandGestureController(
            detection_confidence=0.6, tracking_confidence=0.6, smoothing_factor=0.4
        )
        
        self.records_manager = RecordsManager()
        
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.game_state = 'playing'
        self.score = 0
        self.lives = 3
        
        self.start_time = time.time()
        self.pause_time = 0
        self.total_pause_duration = 0
        self.game_time = 0
        
        self.final_time = 0
        self.final_score = 0
        self.victory = False
        self.new_record = False
        self.records_broken = []
        
        self.gesture_cooldown = 0
        self.gesture_cooldown_duration = 45
        self.last_processed_gesture = 'none'
        
        # Sistema de Turbo mejorado
        self.turbo_active = False
        self.turbo_start_time = 0
        self.turbo_cooldown_start = 0
        self.turbo_on_cooldown = False
        
        # Power-ups y múltiples bolas
        self.powerups = []
        self.balls = []
        self.powerup_spawn_chance = 0.25  # 25% probabilidad de spawn
        self.powerups_spawned = 0  # Contador de power-ups generados
        self.max_powerups_per_game = 6  # Máximo 6 power-ups por partida
        
        self.current_detected_gesture = 'none'
        self.gesture_display_timer = 0
        self.show_diagnostics = True
        
        self._initialize_game_elements()
        
    def _initialize_game_elements(self):
        """Inicializa elementos del juego."""
        self.paddle = {
            'width': 150, 'height': 20,
            'x': self.WIDTH // 2 - 75, 'y': self.HEIGHT - 100,
            'speed': 0, 'color': (100, 200, 255), 'base_width': 150
        }
        
        # Bola principal
        main_ball = Ball(
            x=self.WIDTH // 2, y=self.HEIGHT // 2,
            speed_x=6, speed_y=-6, radius=10
        )
        self.balls = [main_ball]
        
        self.blocks = []
        block_rows, block_cols = 5, 10
        block_width = self.WIDTH // block_cols
        block_height = 30
        
        colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
        
        for row in range(block_rows):
            for col in range(block_cols):
                self.blocks.append({
                    'x': col * block_width, 'y': 50 + row * block_height,
                    'width': block_width - 2, 'height': block_height - 2,
                    'color': colors[row], 'active': True
                })
        
        self.powerups = []
    
    def get_elapsed_time(self):
        if self.game_state == 'paused':
            return self.pause_time - self.start_time - self.total_pause_duration
        return time.time() - self.start_time - self.total_pause_duration
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 100)
        return f"{minutes:02d}:{secs:02d}.{ms:02d}"
    
    def get_turbo_status(self):
        """Retorna estado del turbo: tiempo restante o cooldown."""
        current_time = time.time()
        
        if self.turbo_active:
            elapsed = current_time - self.turbo_start_time
            remaining = max(0, self.TURBO_DURATION - elapsed)
            return 'active', remaining
        elif self.turbo_on_cooldown:
            elapsed = current_time - self.turbo_cooldown_start
            remaining = max(0, self.TURBO_COOLDOWN - elapsed)
            if remaining <= 0:
                self.turbo_on_cooldown = False
                return 'ready', 0
            return 'cooldown', remaining
        return 'ready', 0
    
    def activate_turbo(self):
        """Activa el turbo si está disponible."""
        status, _ = self.get_turbo_status()
        if status == 'ready':
            self.turbo_active = True
            self.turbo_start_time = time.time()
            
            for ball in self.balls:
                if ball.active:
                    ball.speed_x *= self.TURBO_MULTIPLIER
                    ball.speed_y *= self.TURBO_MULTIPLIER
                    ball.color = (255, 255, 0)
            
            print("⚡ TURBO ACTIVADO!")
            return True
        return False
    
    def deactivate_turbo(self):
        """Desactiva el turbo e inicia cooldown."""
        if self.turbo_active:
            self.turbo_active = False
            self.turbo_on_cooldown = True
            self.turbo_cooldown_start = time.time()
            
            for ball in self.balls:
                if ball.active:
                    ball.speed_x /= self.TURBO_MULTIPLIER
                    ball.speed_y /= self.TURBO_MULTIPLIER
                    ball.color = (255, 100, 100)
            
            print("⏳ Turbo en cooldown...")
    
    def update_turbo(self):
        """Actualiza estado del turbo."""
        if self.turbo_active:
            elapsed = time.time() - self.turbo_start_time
            if elapsed >= self.TURBO_DURATION:
                self.deactivate_turbo()
    
    def spawn_powerup(self, x, y):
        """Genera un power-up con probabilidad aleatoria (máximo 6 por partida)."""
        # Verificar límite de power-ups
        if self.powerups_spawned >= self.max_powerups_per_game:
            return
        
        if random.random() < self.powerup_spawn_chance:
            # x2 tiene 80% probabilidad, x5 tiene 20% probabilidad
            if random.random() < 0.80:
                powerup_type = 'multi_2'
            else:
                powerup_type = 'multi_5'
            
            powerup = PowerUp(x, y, powerup_type)
            self.powerups.append(powerup)
            self.powerups_spawned += 1
            print(f"✨ Power-up spawneado: {powerup_type} ({self.powerups_spawned}/{self.max_powerups_per_game})")
    
    def apply_powerup(self, powerup):
        """Aplica el efecto del power-up."""
        if powerup.type == 'multi_2':
            self.multiply_balls(2)
        elif powerup.type == 'multi_5':
            self.multiply_balls(5)
    
    def multiply_balls(self, multiplier):
        """Multiplica las bolas activas."""
        new_balls = []
        active_balls = [b for b in self.balls if b.active and b.launched]
        
        if not active_balls:
            return
        
        for ball in active_balls:
            for i in range(multiplier - 1):
                angle_offset = (i + 1) * (np.pi / 6)
                
                current_angle = np.arctan2(ball.speed_y, ball.speed_x)
                current_speed = np.sqrt(ball.speed_x**2 + ball.speed_y**2)
                
                new_angle = current_angle + angle_offset * (1 if i % 2 == 0 else -1)
                new_vx = current_speed * np.cos(new_angle)
                new_vy = current_speed * np.sin(new_angle)
                
                new_ball = Ball(ball.x, ball.y, new_vx, new_vy, ball.radius)
                new_ball.launched = True
                new_ball.color = ball.color
                new_balls.append(new_ball)
        
        self.balls.extend(new_balls)
        print(f"🎱 Bolas multiplicadas: {len(self.balls)} activas")
    
    def process_gesture_input(self):
        """Procesa entrada gestual."""
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        processed_frame, landmarks_data = self.gesture_controller.calculate_hand_landmarks(frame)
        
        if landmarks_data['hands_detected'] and landmarks_data['palm_center'] is not None:
            palm_x = self.gesture_controller.get_smoothed_position(
                landmarks_data['palm_center'], self.WIDTH
            )
            
            self.paddle['x'] = max(0, min(palm_x - self.paddle['width'] // 2, 
                                         self.WIDTH - self.paddle['width']))
            
            static_gesture = landmarks_data['static_gesture']
            dynamic_gesture = landmarks_data['dynamic_gesture']
            
            if dynamic_gesture != 'none' and self.gesture_cooldown <= 0:
                if dynamic_gesture != self.last_processed_gesture:
                    self._handle_dynamic_gesture(dynamic_gesture)
                    self.gesture_cooldown = self.gesture_cooldown_duration
                    self.last_processed_gesture = dynamic_gesture
                    self.current_detected_gesture = dynamic_gesture
                    self.gesture_display_timer = 60
            elif static_gesture != 'none' and self.gesture_cooldown <= 0:
                if static_gesture != self.last_processed_gesture:
                    self._handle_static_gesture(static_gesture)
                    self.gesture_cooldown = self.gesture_cooldown_duration // 2
                    self.last_processed_gesture = static_gesture
                    self.current_detected_gesture = static_gesture
                    self.gesture_display_timer = 60
        
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
        
        if self.gesture_display_timer > 0:
            self.gesture_display_timer -= 1
            if self.gesture_display_timer == 0:
                self.current_detected_gesture = 'none'
        
        return processed_frame
    
    def _handle_static_gesture(self, gesture):
        """Maneja gestos estáticos."""
        if gesture == 'fist':
            if self.game_state == 'playing':
                self.toggle_pause()
        elif gesture == 'open_palm':
            if self.game_state == 'paused':
                self.toggle_pause()
            elif not any(b.launched for b in self.balls):
                for ball in self.balls:
                    ball.launched = True
        elif gesture == 'pinch':
            if not any(b.launched for b in self.balls):
                for ball in self.balls:
                    ball.launched = True
        elif gesture == 'thumb_up':
            self.activate_turbo()
    
    def _handle_dynamic_gesture(self, gesture):
        """Maneja gestos dinámicos."""
        if gesture == 'swipe_left':
            self.paddle['x'] = max(0, self.paddle['x'] - 150)
        elif gesture == 'swipe_right':
            self.paddle['x'] = min(self.WIDTH - self.paddle['width'], self.paddle['x'] + 150)
        elif gesture == 'swipe_up':
            self.activate_turbo()
    
    def update_game_physics(self):
        """Actualiza física del juego con rebotes mejorados."""
        if self.game_state != 'playing':
            return
        
        self.game_time = self.get_elapsed_time()
        self.update_turbo()
        
        # Actualizar power-ups
        for powerup in self.powerups[:]:
            powerup.update()
            
            # Colisión con paddle
            paddle_rect = pygame.Rect(self.paddle['x'], self.paddle['y'],
                                      self.paddle['width'], self.paddle['height'])
            powerup_rect = pygame.Rect(powerup.x, powerup.y, powerup.width, powerup.height)
            
            if paddle_rect.colliderect(powerup_rect):
                self.apply_powerup(powerup)
                self.powerups.remove(powerup)
            elif powerup.y > self.HEIGHT:
                self.powerups.remove(powerup)
        
        # Verificar si hay bolas activas
        active_balls = [b for b in self.balls if b.active]
        if not active_balls:
            self.lives -= 1
            if self.lives <= 0:
                self._end_game(victory=False)
            else:
                self._reset_ball()
            return
        
        for ball in self.balls:
            if not ball.active:
                continue
            
            if not ball.launched:
                ball.x = self.paddle['x'] + self.paddle['width'] // 2
                ball.y = self.paddle['y'] - ball.radius - 5
                continue
            
            # Movimiento suavizado
            ball.x += ball.speed_x
            ball.y += ball.speed_y
            
            # REBOTE EN BORDES - velocidad conservada
            # Borde izquierdo
            if ball.x - ball.radius <= 0:
                ball.x = ball.radius + 1
                ball.speed_x = abs(ball.speed_x)
            
            # Borde derecho
            if ball.x + ball.radius >= self.WIDTH:
                ball.x = self.WIDTH - ball.radius - 1
                ball.speed_x = -abs(ball.speed_x)
            
            # Borde superior
            if ball.y - ball.radius <= 0:
                ball.y = ball.radius + 1
                ball.speed_y = abs(ball.speed_y)
            
            # Borde inferior - pierde bola
            if ball.y + ball.radius >= self.HEIGHT:
                ball.active = False
                continue
            
            # Colisión con paddle
            paddle_rect = pygame.Rect(self.paddle['x'], self.paddle['y'],
                                      self.paddle['width'], self.paddle['height'])
            ball_rect = pygame.Rect(ball.x - ball.radius, ball.y - ball.radius,
                                    ball.radius * 2, ball.radius * 2)
            
            if paddle_rect.colliderect(ball_rect) and ball.speed_y > 0:
                ball.y = self.paddle['y'] - ball.radius - 1
                
                impact_point = (ball.x - self.paddle['x']) / self.paddle['width']
                angle_factor = (impact_point - 0.5) * 2
                
                current_speed = np.sqrt(ball.speed_x**2 + ball.speed_y**2)
                new_angle = -np.pi/2 + angle_factor * (np.pi/3)
                
                new_vx = current_speed * np.cos(new_angle + np.pi/2)
                new_vy = -abs(current_speed * np.sin(new_angle + np.pi/2))
                
                ball.update_velocity(new_vx, new_vy)
            
            # Colisión con bloques
            for block in self.blocks:
                if not block['active']:
                    continue
                
                block_rect = pygame.Rect(block['x'], block['y'],
                                         block['width'], block['height'])
                
                if block_rect.colliderect(ball_rect):
                    block['active'] = False
                    
                    # Puntuación según color
                    block_color = tuple(block['color'])
                    points = self.BLOCK_SCORES.get(block_color, 10)
                    self.score += points
                    
                    # Spawn power-up
                    self.spawn_powerup(
                        block['x'] + block['width'] // 2 - 15,
                        block['y'] + block['height']
                    )
                    
                    # Determinar lado de colisión para rebote
                    ball_center = np.array([ball.x, ball.y])
                    block_center = np.array([block['x'] + block['width']/2,
                                            block['y'] + block['height']/2])
                    
                    diff = ball_center - block_center
                    
                    if abs(diff[0]) / block['width'] > abs(diff[1]) / block['height']:
                        ball.update_velocity(-ball.speed_x, ball.speed_y)
                    else:
                        ball.update_velocity(ball.speed_x, -ball.speed_y)
                    
                    break
        
        # Victoria
        if all(not block['active'] for block in self.blocks):
            self._end_game(victory=True)
    
    def _end_game(self, victory):
        """Finaliza el juego."""
        self.game_state = 'game_over' if not victory else 'victory'
        self.victory = victory
        self.final_time = self.get_elapsed_time()
        self.final_score = self.score
        
        self.new_record, self.records_broken = self.records_manager.update_records(
            self.final_score, self.final_time, victory
        )
        
        if self.new_record:
            print(f"\n🎉 ¡NUEVO RÉCORD! 🎉")
    
    def _reset_ball(self):
        """Reinicia las bolas."""
        main_ball = Ball(self.WIDTH // 2, self.HEIGHT // 2, 6, -6, 10)
        self.balls = [main_ball]
        self.powerups = []
        
        if self.turbo_active:
            self.deactivate_turbo()
        
        self.paddle['width'] = self.paddle['base_width']
    
    def toggle_pause(self):
        """Alterna pausa."""
        if self.game_state == 'playing':
            self.game_state = 'paused'
            self.pause_time = time.time()
        elif self.game_state == 'paused':
            self.game_state = 'playing'
            self.total_pause_duration += time.time() - self.pause_time
    
    def render_frame(self, camera_frame=None):
        """Renderiza frame completo."""
        self.screen.fill((20, 20, 40))
        
        if camera_frame is not None:
            cam_surface = self._convert_cv_to_pygame(camera_frame)
            cam_surface = pygame.transform.scale(cam_surface, (320, 240))
            cam_surface.set_alpha(200)
            self.screen.blit(cam_surface, (self.WIDTH - 330, 10))
            
            detection_ok = self.gesture_controller.detection_stats['consecutive_failures'] == 0
            pygame.draw.rect(self.screen, (100, 255, 100) if detection_ok else (255, 100, 100),
                           (self.WIDTH - 330, 10, 320, 240), 3)
        
        # Paddle
        paddle_color = (255, 255, 100) if self.turbo_active else self.paddle['color']
        pygame.draw.rect(self.screen, paddle_color,
                        (self.paddle['x'], self.paddle['y'],
                         self.paddle['width'], self.paddle['height']))
        
        # Bolas
        for ball in self.balls:
            if ball.active:
                pygame.draw.circle(self.screen, ball.color,
                                 (int(ball.x), int(ball.y)), ball.radius)
        
        # Bloques
        for block in self.blocks:
            if block['active']:
                pygame.draw.rect(self.screen, block['color'],
                               (block['x'], block['y'], block['width'], block['height']))
        
        # Power-ups
        for powerup in self.powerups:
            powerup.draw(self.screen)
        
        self._render_ui()
        pygame.display.flip()
    
    def _convert_cv_to_pygame(self, cv_frame):
        cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        cv_frame = np.rot90(cv_frame)
        return pygame.surfarray.make_surface(cv_frame)
    
    def _render_ui(self):
        """Renderiza interfaz de usuario."""
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 24)
        
        # Score y vidas
        score_text = font_medium.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        lives_text = font_medium.render(f"Vidas: {self.lives}", True, (255, 255, 255))
        self.screen.blit(lives_text, (10, 50))
        
        # Cronómetro
        time_text = font_medium.render(f"Tiempo: {self.format_time(self.game_time)}", True, (255, 255, 100))
        self.screen.blit(time_text, (10, 90))
        
        # Bolas activas
        active_count = len([b for b in self.balls if b.active])
        balls_text = font_small.render(f"Bolas: {active_count}", True, (0, 255, 255))
        self.screen.blit(balls_text, (10, 130))
        
        # Estado del Turbo
        turbo_status, turbo_time = self.get_turbo_status()
        y_turbo = 160
        
        if turbo_status == 'active':
            turbo_text = font_small.render(f"⚡ TURBO: {turbo_time:.1f}s", True, (255, 255, 0))
            # Barra de progreso
            bar_width = 150
            progress = turbo_time / self.TURBO_DURATION
            pygame.draw.rect(self.screen, (100, 100, 100), (10, y_turbo + 25, bar_width, 10))
            pygame.draw.rect(self.screen, (255, 255, 0), (10, y_turbo + 25, int(bar_width * progress), 10))
        elif turbo_status == 'cooldown':
            turbo_text = font_small.render(f"⏳ Cooldown: {turbo_time:.1f}s", True, (255, 100, 100))
            bar_width = 150
            progress = 1 - (turbo_time / self.TURBO_COOLDOWN)
            pygame.draw.rect(self.screen, (100, 100, 100), (10, y_turbo + 25, bar_width, 10))
            pygame.draw.rect(self.screen, (100, 100, 255), (10, y_turbo + 25, int(bar_width * progress), 10))
        else:
            turbo_text = font_small.render("⚡ Turbo: LISTO (👍)", True, (0, 255, 0))
        
        self.screen.blit(turbo_text, (10, y_turbo))
        
        # Récords
        records = self.records_manager.get_records()
        y_records = 210
        best_score_text = font_small.render(f"Mejor: {records['best_score']}", True, (255, 200, 100))
        self.screen.blit(best_score_text, (10, y_records))
        
        # Leyenda de puntuación
        y_legend = 250
        legend_title = font_small.render("Puntos por color:", True, (200, 200, 200))
        self.screen.blit(legend_title, (10, y_legend))
        
        color_points = [("🔴", 50), ("🟠", 40), ("🟡", 30), ("🟢", 20), ("🔵", 10)]
        for i, (emoji, pts) in enumerate(color_points):
            txt = font_small.render(f"{emoji} {pts}pts", True, (180, 180, 180))
            self.screen.blit(txt, (10 + (i % 3) * 70, y_legend + 25 + (i // 3) * 20))
        
        # Gesto detectado
        if self.current_detected_gesture != 'none':
            gesture_names = {
                'swipe_left': '← SWIPE', 'swipe_right': 'SWIPE →',
                'swipe_up': '↑ TURBO',
                'fist': '✊ PAUSA', 'open_palm': '✋ REANUDAR',
                'pinch': '🤏 LANZAR', 'thumb_up': '👍 TURBO'
            }
            display_name = gesture_names.get(self.current_detected_gesture, self.current_detected_gesture)
            gesture_text = font_large.render(display_name, True, (0, 255, 255))
            text_rect = gesture_text.get_rect(center=(self.WIDTH//2, 150))
            
            bg_rect = pygame.Surface((text_rect.width + 40, text_rect.height + 20))
            bg_rect.set_alpha(180)
            bg_rect.fill((0, 0, 0))
            self.screen.blit(bg_rect, (text_rect.x - 20, text_rect.y - 10))
            self.screen.blit(gesture_text, text_rect)
        
        # Instrucciones
        instructions = [
            "GESTOS: ✊Pausa | ✋Reanudar | 👍/↑Turbo(10s)",
            "TECLADO: SPACE=Pausa | R=Reiniciar | ESC=Salir"
        ]
        y_inst = self.HEIGHT - 60
        for inst in instructions:
            inst_text = font_small.render(inst, True, (180, 220, 255))
            self.screen.blit(inst_text, (10, y_inst))
            y_inst += 25
        
        # Estados especiales
        if self.game_state == 'paused':
            self._render_overlay("⏸ PAUSADO", (255, 255, 0), "Mano abierta para continuar")
        elif self.game_state == 'game_over':
            self._render_end_screen("💀 GAME OVER", (255, 0, 0))
        elif self.game_state == 'victory':
            self._render_end_screen("🎉 ¡VICTORIA!", (0, 255, 0))
        
        if not any(b.launched for b in self.balls) and self.game_state == 'playing':
            launch_text = font_medium.render("PINZA o MANO ABIERTA para lanzar", True, (255, 255, 100))
            text_rect = launch_text.get_rect(center=(self.WIDTH//2, self.HEIGHT//2 + 100))
            self.screen.blit(launch_text, text_rect)
    
    def _render_overlay(self, title, color, subtitle):
        """Renderiza overlay semi-transparente."""
        font_large = pygame.font.Font(None, 54)
        font_small = pygame.font.Font(None, 28)
        
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        title_text = font_large.render(title, True, color)
        title_rect = title_text.get_rect(center=(self.WIDTH//2, self.HEIGHT//2))
        self.screen.blit(title_text, title_rect)
        
        sub_text = font_small.render(subtitle, True, (255, 255, 255))
        sub_rect = sub_text.get_rect(center=(self.WIDTH//2, self.HEIGHT//2 + 50))
        self.screen.blit(sub_text, sub_rect)
    
    def _render_end_screen(self, title, color):
        """Renderiza pantalla de fin de juego."""
        font_large = pygame.font.Font(None, 54)
        font_medium = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 28)
        
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        y = self.HEIGHT//2 - 150
        
        title_text = font_large.render(title, True, color)
        self.screen.blit(title_text, title_text.get_rect(center=(self.WIDTH//2, y)))
        y += 80
        
        score_text = font_medium.render(f"Puntuación: {self.final_score}", True, (255, 255, 100))
        self.screen.blit(score_text, score_text.get_rect(center=(self.WIDTH//2, y)))
        y += 50
        
        time_text = font_medium.render(f"Tiempo: {self.format_time(self.final_time)}", True, (255, 255, 100))
        self.screen.blit(time_text, time_text.get_rect(center=(self.WIDTH//2, y)))
        y += 60
        
        if self.new_record:
            record_text = font_medium.render("🎉 ¡NUEVO RÉCORD! 🎉", True, (255, 215, 0))
            self.screen.blit(record_text, record_text.get_rect(center=(self.WIDTH//2, y)))
            y += 50
        
        restart_text = font_small.render("Presiona R para reiniciar", True, (255, 255, 255))
        self.screen.blit(restart_text, restart_text.get_rect(center=(self.WIDTH//2, self.HEIGHT - 80)))
    
    def run(self):
        """Loop principal del juego."""
        running = True
        
        print("\n" + "="*70)
        print("  BREAKOUT - POWER-UPS Y TURBO MEJORADO")
        print("="*70)
        print("\n🆕 CARACTERÍSTICAS:")
        print("  • 🎱 Power-ups x2 y x5 bolas (aleatorio)")
        print("  • ⚡ Turbo: 10s activo, 20s cooldown")
        print("  • 🎨 Puntuación por color: 🔴50 🟠40 🟡30 🟢20 🔵10")
        print("  • 🏓 Rebotes suavizados")
        print("="*70 + "\n")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_pause()
                    elif event.key == pygame.K_r and self.game_state in ['game_over', 'victory']:
                        self._initialize_game_elements()
                        self.score = 0
                        self.lives = 3
                        self.game_state = 'playing'
                        self.turbo_active = False
                        self.turbo_on_cooldown = False
                        self.powerups_spawned = 0  # Reiniciar contador
                        self.start_time = time.time()
                        self.total_pause_duration = 0
                        self.new_record = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            camera_frame = self.process_gesture_input()
            self.update_game_physics()
            self.render_frame(camera_frame)
            self.clock.tick(60)
        
        self.camera.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("¡Gracias por jugar! 👋\n")


def main():
    try:
        print("\n🎮 Inicializando Breakout...")
        
        test_cam = cv2.VideoCapture(0)
        if not test_cam.isOpened():
            print("❌ ERROR: No se puede acceder a la cámara")
            return
        test_cam.release()
        
        print("✅ Cámara detectada")
        print("🚀 Iniciando juego...\n")
        
        game = EnhancedBreakoutGame(width=1200, height=900)
        game.run()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    main()
