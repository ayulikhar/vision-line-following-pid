import cv2
import numpy as np

# PID constants
Kp = 0.3
Ki = 0.0003
Kd = 0.15

prev_error = 0
integral = 0

prev_cx = 170   # center of ROI
alpha = 0.85

last_seen_direction = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Draw ROI and center line
    cv2.rectangle(frame, (150, 350), (490, 480), (255, 0, 0), 2)
    cv2.line(frame, (320, 350), (320, 480), (255,255,0), 2)

    # Convert to HSV (for color filtering)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect black color only
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])

    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Blur to reduce noise
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    # ROI (very focused)
    roi = mask[350:480, 150:490]

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1500]

    if len(valid_contours) > 0:
        c = max(valid_contours, key=cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)

        # Ignore thin shapes
        if w > 30:
            M = cv2.moments(c)
            if M["m00"] != 0:
                raw_cx = int(M["m10"] / M["m00"])
            else:
                raw_cx = prev_cx

            # Prevent sudden jumps
            if abs(raw_cx - prev_cx) > 100:
                cx = prev_cx
            else:
                cx = int(alpha * prev_cx + (1 - alpha) * raw_cx)

            prev_cx = cx

            # Draw contour
            cv2.drawContours(frame[350:480, 150:490], [c], -1, (0,255,0), 2)
            cv2.circle(frame, (cx + 150, 420), 6, (0,0,255), -1)

            # Error calculation
            error = cx - 170
            if abs(error) < 10:
                error = 0

            # PID
            integral += error
            derivative = error - prev_error
            control = Kp*error + Ki*integral + Kd*derivative
            prev_error = error

            control *= 0.5

            # Direction memory
            last_seen_direction = 1 if error > 0 else -1

            # Adaptive speed
            speed = max(0, 100 - abs(error))

            # Display info
            cv2.putText(frame, "Tracking", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(frame, f"Error: {error}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(frame, f"Speed: {int(speed)}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        else:
            cv2.putText(frame, "Noise Ignored", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    else:
        # SEARCH MODE
        cv2.putText(frame, "Searching...", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        control = 40 * last_seen_direction

    cv2.imshow("Autonomous Line Follower", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
