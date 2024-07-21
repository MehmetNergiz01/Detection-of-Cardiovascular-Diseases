import cv2
import numpy as np
import dlib
import time

# Kamera ve yüz algılayıcıyı başlat
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

# Yazı tipi ve metin özellikleri
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 0.7
color = (255, 0, 0)
thickness = 2

# Nabız ve BPM değişkenlerini başlat
pulse = []
last_bpm = 0
start_time = time.time()

# Göz algılama değişkenlerini başlat
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eye_detected = False
eye_status = "Normal"

# Cilt tonu algılama değişkenlerini başlat
skin_tone_lower = np.array([0, 48, 80], dtype=np.uint8)
skin_tone_upper = np.array([20, 255, 255], dtype=np.uint8)
skin_tone_detected = False
skin_tone_status = "Normal"

# Kan basıncı ve kalp atış hızı değişkenlerini başlat
blood_pressure = 120  # Örnek değer
heart_rate = 75  # Örnek değer
breath_rate = 20  # Örnek değer

# Nefes alıp verme oranını takip etmek için değişkenler
breath_count = 0
last_breath_time = time.time()
breath_start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera okunamıyor!")
        break

    # Frame'i gri tonlamaya çevir ve yüzleri algıla
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        x1, y1, x2, y2, w, h = face.left(), face.top(), face.right(), face.bottom(), face.width(), face.height()

        # ROI'yi (İlgi Bölgesi) yüzden çıkar
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Gözleri algıla
        eyes = eye_cascade.detectMultiScale(roi)
        if len(eyes) > 0:
            eye_detected = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eye_roi = roi[ey:ey + eh, ex:ex + ew]
                eye_status = "Tespit Edildi"
        else:
            eye_detected = False
            eye_status = "Tespit Edilmedi"

        # Cilt tonunu algıla
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(roi_hsv, skin_tone_lower, skin_tone_upper)
        skin = cv2.bitwise_and(roi, roi, mask=mask)
        if np.mean(skin) > 100:
            skin_tone_detected = True
            skin_tone_status = "Normal"
        else:
            skin_tone_detected = False
            skin_tone_status = "Anormal"

        # Nabzı hesapla
        pulse.append(np.mean(roi))

        # BPM'yi her saniye hesapla
        current_time = time.time()
        time_diff = current_time - start_time
        if time_diff >= 1:
            bpm = len(pulse) * 60
            last_bpm = bpm
            start_time = current_time
            pulse = []

        # Nefes alıp verme oranını hesapla (basit bir örnek)
        if np.mean(roi) > 105:  # Nefes almayı tespit etmek için basit bir eşik değeri
            if current_time - last_breath_time > 2:  # Nefes aralığı en az 2 saniye
                breath_count += 1
                last_breath_time = current_time

        # Toplam zamanın sıfır olup olmadığını kontrol et
        total_breath_time = current_time - breath_start_time
        if total_breath_time > 0:
            breath_rate = breath_count * (60 / total_breath_time)

        # Sağlık metriklerini görüntüle
        cv2.putText(frame, "Nabiz: {}".format(last_bpm), org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "Kan Basinci: {}".format(blood_pressure), (50, 80), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "Kalp Atis Hizi: {}".format(heart_rate), (50, 110), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "Goz Bulgulari: {}".format(eye_status), (50, 140), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "Cilt Tonu: {}".format(skin_tone_status), (50, 170), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "Nefes Alis: {:.2f}".format(breath_rate), (50, 200), font, fontScale, color, thickness, cv2.LINE_AA)

        # ROI ve cilt tonunu görüntüle
        cv2.imshow("ROI", roi)
        cv2.imshow("Cilt Tonu", skin)

    # Ana frame'i görüntüle
    cv2.imshow('Frame', frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
