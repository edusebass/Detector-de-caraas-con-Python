import cv2

# Cargar el clasificador pre-entrenado de Haar Cascade para detección de caras
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inicializar la cámara web
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir la camara web")
    exit()

while True:
    # Capturar un frame de la cámara web
    ret, frame = cap.read()

    if not ret:
        continue
    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ejecutar la detección de caras utilizando el clasificador de Haar Cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Dibujar un rectángulo alrededor de cada cara detectada
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostrar el frame con las caras detectadas
    cv2.imshow('Detector de caras con Haar Cascade', frame)

    # Esperar a que se presione la tecla 'q' para salir
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()