import cv2
import pygame

pygame.mixer.init()

try:
    pygame.mixer.music.load("foto_kita_blur.mp3")
except pygame.error:
    print("File 'foto_kita_blur.mp3' tidak ditemukan. Pastikan file lagu berada di folder yang sama.")

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
except ImportError:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_draw

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def check_v_sign(landmarks):
    lm = landmarks.landmark
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y
    return index_up and middle_up and ring_down and pinky_down

cap = cv2.VideoCapture(0)

# Kontrol audio
is_playing = False

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    v_sign_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if check_v_sign(hand_landmarks):
                v_sign_detected = True

    # + Audio Toggle
    if v_sign_detected:
        # Blur layar
        frame = cv2.GaussianBlur(frame, (99, 99), 0)
        
        # Putar lagu jika belum berjalan
        if not is_playing:
            pygame.mixer.music.play(-1)  # loop lagu
            is_playing = True

        cv2.putText(frame, "FOTO KITA BLUR", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        # stop lagu
        if is_playing:
            pygame.mixer.music.stop()
            is_playing = False

        cv2.putText(frame, "Tunjukkan Jari 'V' untuk Memblurkan Layar", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Foto Kita Blur + Audio", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()